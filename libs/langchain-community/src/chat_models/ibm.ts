import {
  AIMessage,
  AIMessageChunk,
  ChatMessageChunk,
  FunctionMessageChunk,
  HumanMessage,
  HumanMessageChunk,
  isAIMessage,
  MessageContent,
  MessageType,
  ToolMessageChunk,
  type BaseMessage,
} from "@langchain/core/messages";
import {
  BaseLanguageModelInput,
  type BaseLanguageModelCallOptions,
} from "@langchain/core/language_models/base";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  BaseChatModel,
  BindToolsInput,
  type BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import {
  ChatGeneration,
  ChatGenerationChunk,
  ChatResult,
} from "@langchain/core/outputs";
import { AsyncCaller } from "@langchain/core/utils/async_caller";
import {
  TextChatMessagesTextChatMessageAssistant,
  TextChatParameterTools,
  TextChatParams,
  TextChatResponse,
  TextChatResultChoice,
  TextChatResultMessage,
  TextChatToolCall,
  TextChatUsage,
} from "@ibm-cloud/watsonx-ai/dist/watsonx-ai-ml/vml_v1.js";
import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import zodToJsonSchema from "zod-to-json-schema";
import {
  JsonOutputKeyToolsParser,
  convertLangChainToolCallToOpenAI,
  makeInvalidToolCall,
  parseToolCall,
} from "@langchain/core/output_parsers/openai_tools";
import { WatsonxAuth, WatsonxParams } from "../types/watsonx_ai.js";
import { authenticateAndSetInstance } from "../utils/ibm.js";
import { ToolCallChunk } from "@langchain/core/messages/tool";
import { Runnable } from "@langchain/core/runnables";

export interface WatsonxDeltaStream {
  role?: string;
  content?: string;
  tool_calls?: TextChatToolCall[];
  refusal?: string;
}
export interface WatsonxCallOptionsChat
  extends BaseLanguageModelCallOptions,
    Omit<TextChatParams, "tools"> {
  maxRetries?: number;
  tools?: ChatWatsonxToolType[];
}

type ChatWatsonxToolType = BindToolsInput | TextChatParameterTools;

export interface ChatWatsonxInput extends BaseChatModelParams, WatsonxParams {}

function _convertToolToWatsonxTool(
  tools: ChatWatsonxToolType[]
): WatsonXAI.TextChatParameterTools[] {
  return tools.map((tool) => {
    if ("type" in tool) {
      return tool as WatsonXAI.TextChatParameterTools;
    }
    const description = tool.description ?? `Tool: ${tool.name}`;
    return {
      type: "function",
      function: {
        name: tool.name,
        description,
        parameters: zodToJsonSchema(tool.schema),
      },
    };
  });
}

function convertMessagesToWatsonxMessages(
  messages: BaseMessage[]
): TextChatResultMessage[] {
  const getRole = (role: MessageType) => {
    switch (role) {
      case "human":
        return "user";
      case "ai":
        return "assistant";
      case "system":
        return "system";
      case "tool":
        return "tool";
      case "function":
        return "assistant";
      default:
        throw new Error(`Unknown message type: ${role}`);
    }
  };

  const getContent = (content: MessageContent): string => {
    if (typeof content === "string") {
      return content;
    }
    throw new Error(
      `WatsonxChat does not support non text message content. Received: ${JSON.stringify(
        content,
        null,
        2
      )}`
    );
  };

  const getTools = (message: BaseMessage): TextChatToolCall[] | undefined => {
    if (isAIMessage(message) && !!message.tool_calls?.length) {
      return message.tool_calls
        .map((toolCall) => ({
          ...toolCall,
          id: toolCall.id ?? "",
        }))
        .map(convertLangChainToolCallToOpenAI) as TextChatToolCall[];
    }
    if (!message.additional_kwargs.tool_calls?.length) {
      return undefined;
    }
    const toolCalls = message.additional_kwargs.tool_calls;
    return toolCalls?.map((toolCall) => ({
      id: toolCall.id,
      type: "function",
      function: toolCall.function,
    }));
  };

  return messages.map((message) => {
    const toolCalls = getTools(message);
    const content = toolCalls === undefined ? getContent(message.content) : "";
    if ("tool_call_id" in message && typeof message.tool_call_id === "string") {
      return {
        role: getRole(message._getType()),
        content,
        name: message.name,
        tool_call_id: message.tool_call_id,
      };
    }

    return {
      role: getRole(message._getType()),
      content,
      tool_calls: toolCalls,
    };
  }) as TextChatResultMessage[];
}

function watsonxResponseToChatMessage(
  choice: TextChatResultChoice,
  rawDataId: string,
  usage?: TextChatUsage
): BaseMessage {
  const { message } = choice;
  if (!message) throw new Error("No message presented");
  let rawToolCalls: TextChatToolCall[] = message.tool_calls ?? [];

  switch (message.role) {
    case "assistant": {
      const toolCalls = [];
      const invalidToolCalls = [];
      for (const rawToolCall of rawToolCalls) {
        try {
          const parsed = parseToolCall(rawToolCall, { returnId: true });
          toolCalls.push({
            ...parsed,
            id: parsed.id,
          });
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } catch (e: any) {
          invalidToolCalls.push(makeInvalidToolCall(rawToolCall, e.message));
        }
      }
      return new AIMessage({
        id: rawDataId,
        content: message.content ?? "",
        tool_calls: toolCalls,
        invalid_tool_calls: invalidToolCalls,
        additional_kwargs: {
          tool_calls: rawToolCalls.length
            ? rawToolCalls.map((toolCall) => ({
                ...toolCall,
                type: "function",
              }))
            : undefined,
        },
        usage_metadata: usage
          ? {
              input_tokens: usage.prompt_tokens ?? 0,
              output_tokens: usage.completion_tokens ?? 0,
              total_tokens: usage.total_tokens ?? 0,
            }
          : undefined,
      });
    }
    default:
      return new HumanMessage(message.content ?? "");
  }
}

function _convertDeltaToMessageChunk(
  delta: WatsonxDeltaStream,
  rawData: TextChatResponse,
  usage?: TextChatUsage,
  defaultRole?: TextChatMessagesTextChatMessageAssistant.Constants.Role
) {
  if (delta.refusal) throw new Error(delta.refusal);
  const rawToolCalls = delta.tool_calls?.length
    ? delta.tool_calls?.map(
        (
          toolCall,
          index
        ): TextChatToolCall & {
          index: number;
          type: "function";
        } => ({
          ...toolCall,
          index,
          id: toolCall.id,
          type: "function",
        })
      )
    : undefined;

  let role = "assistant";
  if (delta.role) {
    role = delta.role;
  } else if (defaultRole) {
    role = defaultRole;
  }
  const content = delta.content ?? "";
  let additional_kwargs;
  if (rawToolCalls) {
    additional_kwargs = {
      tool_calls: rawToolCalls,
    };
  } else {
    additional_kwargs = {};
  }

  if (role === "user") {
    return new HumanMessageChunk({ content });
  } else if (role === "assistant") {
    const toolCallChunks: ToolCallChunk[] = [];
    if (rawToolCalls)
      for (const rawToolCallChunk of rawToolCalls) {
        toolCallChunks.push({
          name: rawToolCallChunk.function?.name,
          args: rawToolCallChunk.function?.arguments,
          id: rawToolCallChunk.id,
          index: rawToolCallChunk.index,
          type: "tool_call_chunk",
        });
      }

    return new AIMessageChunk({
      content,
      tool_call_chunks: toolCallChunks,
      additional_kwargs,
      usage_metadata: {
        input_tokens: usage?.prompt_tokens ?? 0,
        output_tokens: usage?.completion_tokens ?? 0,
        total_tokens: usage?.total_tokens ?? 0,
      },
      id: rawData.id,
    });
  } else if (role === "tool") {
    if (rawToolCalls)
      return new ToolMessageChunk({
        content,
        additional_kwargs,
        tool_call_id: rawToolCalls?.[0].id,
      });
  } else if (role === "function") {
    return new FunctionMessageChunk({
      content,
      additional_kwargs,
    });
  } else {
    return new ChatMessageChunk({ content, role });
  }
  return null;
}

export class ChatWatsonx<
    CallOptions extends WatsonxCallOptionsChat = WatsonxCallOptionsChat
  >
  extends BaseChatModel<CallOptions>
  implements ChatWatsonxInput
{
  static lc_name() {
    return "ChatWatsonx";
  }

  lc_serializable = true;

  get lc_secrets(): { [key: string]: string } {
    return {
      authenticator: "AUTHENTICATOR",
      apiKey: "WATSONX_AI_APIKEY",
      apikey: "WATSONX_AI_APIKEY",
      watsonxAIAuthType: "WATSONX_AI_AUTH_TYPE",
      watsonxAIApikey: "WATSONX_AI_APIKEY",
      watsonxAIBearerToken: "WATSONX_AI_BEARER_TOKEN",
      watsonxAIUsername: "WATSONX_AI_USERNAME",
      watsonxAIPassword: "WATSONX_AI_PASSWORD",
      watsonxAIUrl: "WATSONX_AI_URL",
    };
  }

  get lc_aliases(): { [key: string]: string } {
    return {
      authenticator: "authenticator",
      apikey: "watsonx_ai_apikey",
      apiKey: "watsonx_ai_apikey",
      watsonxAIAuthType: "watsonx_ai_auth_type",
      watsonxAIApikey: "watsonx_ai_apikey",
      watsonxAIBearerToken: "watsonx_ai_bearer_token",
      watsonxAIUsername: "watsonx_ai_username",
      watsonxAIPassword: "watsonx_ai_password",
      watsonxAIUrl: "watsonx_ai_url",
    };
  }
  modelId = "mistralai/mistral-large";

  version: "2024-05-31";

  serviceUrl: string;

  spaceId?: string;

  projectId?: string;

  frequency_penalty?: number;

  logprobs?: boolean;

  top_logprobs?: number;

  max_new_tokens?: number;

  n?: number;

  presence_penalty?: number;

  temperature?: number;

  top_p?: number;

  time_limit?: number;

  maxRetries?: number;

  maxConcurrency?: number;

  stop_sequences: string;

  service: WatsonXAI;

  constructor(fields: ChatWatsonxInput & WatsonxAuth) {
    super(fields);

    if (
      (fields.projectId && fields.spaceId) ||
      (fields.idOrName && fields.projectId) ||
      (fields.spaceId && fields.idOrName)
    )
      throw new Error("Maximum 1 id type can be specified per instance");

    if (!fields.projectId && !fields.spaceId && !fields.idOrName)
      throw new Error(
        "No id specified! At least ide of 1 type has to be specified"
      );
    this.projectId = fields?.projectId;
    this.spaceId = fields?.spaceId;

    this.serviceUrl = fields?.serviceUrl;
    const {
      watsonxAIApikey,
      watsonxAIAuthType,
      watsonxAIBearerToken,
      watsonxAIUsername,
      watsonxAIPassword,
      watsonxAIUrl,
      version,
      serviceUrl,
    } = fields;

    const auth = authenticateAndSetInstance({
      watsonxAIApikey,
      watsonxAIAuthType,
      watsonxAIBearerToken,
      watsonxAIUsername,
      watsonxAIPassword,
      watsonxAIUrl,
      version,
      serviceUrl,
    });
    if (auth) this.service = auth;
    else throw new Error("You have not provided one type of authentication");
  }

  _llmType() {
    return "watsonx";
  }

  invocationParams(options: this["ParsedCallOptions"]) {
    return {
      maxTokens: options.maxTokens ?? this.max_new_tokens,
      temperature: options?.temperature ?? this.temperature,
      timeLimit: options?.timeLimit ?? this.time_limit,
      topP: options?.topP ?? this.top_p,
      presencePenalty: options?.presencePenalty ?? this.presence_penalty,
      n: options?.n ?? this.n,
      topLogprobs: options?.topLogprobs ?? this.top_logprobs,
      logprobs: options?.logprobs ?? this?.logprobs,
      frequencyPenalty: options?.frequencyPenalty ?? this.frequency_penalty,
      tools: options.tools
        ? _convertToolToWatsonxTool(options.tools)
        : undefined,
      toolChoice: options.toolChoice,
      responseFormat: options.responseFormat,
      toolChoiceOption: options.toolChoiceOption,
    };
  }

  override bindTools(
    tools: ChatWatsonxToolType[],
    kwargs?: Partial<CallOptions>
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, CallOptions> {
    return this.bind({
      tools: _convertToolToWatsonxTool(tools),
      ...kwargs,
    } as CallOptions);
  }

  scopeId() {
    if (this.projectId)
      return { projectId: this.projectId, modelId: this.modelId };
    else return { spaceId: this.spaceId, modelId: this.modelId };
  }

  async completionWithRetry<T>(
    callback: () => T,
    options?: this["ParsedCallOptions"]
  ) {
    const caller = new AsyncCaller({
      maxConcurrency: options?.maxConcurrency || this.maxConcurrency,
      maxRetries: this.maxRetries,
    });
    const result = options
      ? caller.callWithOptions(
          {
            signal: options.signal,
          },
          async () => callback()
        )
      : caller.call(async () => callback());

    return result;
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const params = { ...this.invocationParams(options), ...this.scopeId() };
    const watsonxMessages = convertMessagesToWatsonxMessages(messages);

    const callback = () =>
      this.service.textChat({
        ...params,
        messages: watsonxMessages,
      });

    const { result } = await this.completionWithRetry(callback, options);
    const generations: ChatGeneration[] = [];
    for (const part of result.choices) {
      const generation: ChatGeneration = {
        text: part.message?.content ?? "",
        message: watsonxResponseToChatMessage(part, result.id, result?.usage),
      };
      if (part.finish_reason) {
        generation.generationInfo = { finish_reason: part.finish_reason };
      }
      generations.push(generation);
    }
    return {
      generations,
      llmOutput: {
        tokenUsage: result?.usage,
      },
    };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const params = { ...this.invocationParams(options), ...this.scopeId() };
    const watsonxMessages = convertMessagesToWatsonxMessages(messages);
    const callback = () =>
      this.service.textChatStream({
        ...params,
        messages: watsonxMessages,
        returnObject: true,
      });
    const stream = await this.completionWithRetry(callback, options);
    let defaultRole;
    for await (const chunk of stream) {
      if (options.signal?.aborted) {
        throw new Error("AbortError");
      }
      const { data } = chunk;

      const choice = data.choices[0] as TextChatResultChoice &
        Record<"delta", TextChatResultMessage>;
      if (choice && !("delta" in choice)) {
        continue;
      }
      const delta = choice?.delta;
      if (!delta) {
        continue;
      }

      const newTokenIndices = {
        prompt: 0,
        completion: choice.index ?? 0,
      };

      const generationInfo = {
        ...newTokenIndices,
        finish_reason: choice.finish_reason,
      };

      const message = _convertDeltaToMessageChunk(
        delta,
        data,
        chunk.data.usage,
        defaultRole
      );

      defaultRole =
        (delta.role as TextChatMessagesTextChatMessageAssistant.Constants.Role) ??
        defaultRole;
      if (message === null) {
        continue;
      }

      const generationChunk = new ChatGenerationChunk({
        message,
        text: delta?.content ?? "",
        generationInfo,
      });

      yield generationChunk;

      void _runManager?.handleLLMNewToken(
        generationChunk.text ?? "",
        newTokenIndices,
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk }
      );
    }
  }

  /** @ignore */
  _combineLLMOutput() {
    return [];
  }
}
