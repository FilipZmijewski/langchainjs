import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseLLM, BaseLLMParams } from "@langchain/core/language_models/llms";
import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import {
  DeploymentsTextGenerationParams,
  DeploymentsTextGenerationStreamParams,
  DeploymentTextGenProperties,
  ReturnOptionProperties,
  TextGenerationParams,
  TextGenerationStreamParams,
  TextGenLengthPenalty,
  TextGenParameters,
  TextTokenizationParams,
  TextTokenizeParameters,
} from "@ibm-cloud/watsonx-ai/dist/watsonx-ai-ml/vml_v1.js";
import { Generation, LLMResult } from "@langchain/core/outputs";
import { GenerationChunk } from "@langchain/core/outputs";
import { authenticateAndSetInstance } from "./utilis/authentication.js";
import {
  GenerationInfo,
  ResponseChunk,
  TokenUsage,
  WatsonXAuth,
  WatsonXParams,
} from "./types.js";
import { BaseLanguageModelCallOptions } from "@langchain/core/language_models/base";
import { AsyncCaller } from "@langchain/core/utils/async_caller";

/**
 * Input to LLM class.
 */

export interface WatsonXCallOptionsLLM
  extends BaseLanguageModelCallOptions,
    Omit<
      Partial<
        TextGenerationParams &
          TextGenerationStreamParams &
          DeploymentsTextGenerationParams &
          DeploymentsTextGenerationStreamParams
      >,
      "input"
    > {
  maxRetries?: number;
}

export interface WatsonXInputLLM
  extends TextGenParameters,
    WatsonXParams,
    BaseLLMParams {
  streaming?: boolean;
}

/**
 * Integration with an LLM.
 */
export class WatsonXLLM<
    CallOptions extends WatsonXCallOptionsLLM = WatsonXCallOptionsLLM
  >
  extends BaseLLM<CallOptions>
  implements WatsonXInputLLM
{
  // Used for tracing, replace with the same name as your class
  static lc_name() {
    return "WatsonX";
  }

  lc_serializable = true;
  streaming = false;
  modelId = "ibm/granite-13b-chat-v2";
  maxRetries = 0;
  version = "2024-05-31";

  serviceUrl: string;
  max_new_tokens?: number;
  spaceId?: string;
  projectId?: string;
  idOrName?: string;
  decoding_method?: TextGenParameters.Constants.DecodingMethod | string;
  length_penalty?: TextGenLengthPenalty;
  min_new_tokens?: number;
  random_seed?: number;
  stop_sequences?: string[];
  temperature?: number;
  time_limit?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  truncate_input_tokens?: number;
  return_options?: ReturnOptionProperties;
  include_stop_sequence?: boolean;
  maxConcurrency?: number;

  private service: WatsonXAI;

  constructor(fields: WatsonXInputLLM & WatsonXAuth) {
    super(fields);
    this.modelId = fields.modelId ?? this.modelId;
    this.version = fields.version;
    this.max_new_tokens = fields.max_new_tokens ?? this.max_new_tokens;
    this.serviceUrl = fields.serviceUrl;
    this.decoding_method = fields.decoding_method;
    this.length_penalty = fields.length_penalty;
    this.min_new_tokens = fields.min_new_tokens;
    this.random_seed = fields.random_seed;
    this.stop_sequences = fields.stop_sequences;
    this.temperature = fields.temperature;
    this.time_limit = fields.time_limit;
    this.top_k = fields.top_k;
    this.top_p = fields.top_p;
    this.repetition_penalty = fields.repetition_penalty;
    this.truncate_input_tokens = fields.truncate_input_tokens;
    this.return_options = fields.return_options;
    this.include_stop_sequence = fields.include_stop_sequence;
    this.maxRetries = fields.maxRetries || this.maxRetries;
    this.maxConcurrency = fields.maxConcurrency;
    this.streaming = fields.streaming || this.streaming;
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
    this.idOrName = fields?.idOrName;

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

    this.service = authenticateAndSetInstance({
      watsonxAIApikey,
      watsonxAIAuthType,
      watsonxAIBearerToken,
      watsonxAIUsername,
      watsonxAIPassword,
      watsonxAIUrl,
      version,
      serviceUrl,
    });
  }

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

  invocationParams(
    options: this["ParsedCallOptions"]
  ): TextGenParameters | DeploymentTextGenProperties {
    const { parameters } = options;

    return {
      max_new_tokens: parameters?.max_new_tokens ?? this.max_new_tokens,
      decoding_method: parameters?.decoding_method ?? this.decoding_method,
      length_penalty: parameters?.length_penalty ?? this.length_penalty,
      min_new_tokens: parameters?.min_new_tokens ?? this.min_new_tokens,
      random_seed: parameters?.random_seed ?? this.random_seed,
      stop_sequences: options?.stop ?? this.stop_sequences,
      temperature: parameters?.temperature ?? this.temperature,
      time_limit: parameters?.time_limit ?? this.time_limit,
      top_k: parameters?.top_k ?? this.top_k,
      top_p: parameters?.top_p ?? this.top_p,
      repetition_penalty:
        parameters?.repetition_penalty ?? this.repetition_penalty,
      truncate_input_tokens:
        parameters?.truncate_input_tokens ?? this.truncate_input_tokens,
      return_options: parameters?.return_options ?? this.return_options,
      include_stop_sequence:
        parameters?.include_stop_sequence ?? this.include_stop_sequence,
    };
  }

  scopeId() {
    if (this.projectId)
      return { projectId: this.projectId, modelId: this.modelId };
    else if (this.spaceId)
      return { spaceId: this.spaceId, modelId: this.modelId };
    else if (this.idOrName)
      return { idOrName: this.idOrName, modelId: this.modelId };
    else return { spaceId: this.spaceId, modelId: this.modelId };
  }

  async listModels() {
    const listModelParams = {
      filters: "function_text_generation",
    };
    const listModels = await this.completionWithRetry(() =>
      this.service.listFoundationModelSpecs(listModelParams)
    );
    return listModels.result.resources?.map((item) => item.model_id);
  }

  private async generateSingleMessage(
    input: string,
    options: this["ParsedCallOptions"],
    stream: true,
    tokenUsage?: TokenUsage
  ): Promise<AsyncIterable<string>>;

  private async generateSingleMessage(
    input: string,
    options: this["ParsedCallOptions"],
    stream: false,
    tokenUsage?: TokenUsage
  ): Promise<Generation[]>;

  private async generateSingleMessage(
    input: string,
    options: this["ParsedCallOptions"],
    stream: boolean,
    tokenUsage = { generated_token_count: 0, input_token_count: 0 }
  ) {
    const {
      signal,
      stop,
      maxRetries,
      maxConcurrency,
      timeout,
      ...requestOptions
    } = options;

    const idOrName = options?.idOrName ?? this.idOrName;
    const parameters = this.invocationParams(options);
    if (stream) {
      const textStream = idOrName
        ? await this.service.deploymentGenerateTextStream({
            idOrName: idOrName,
            ...requestOptions,
            parameters: {
              ...parameters,
              prompt_variables: {
                input,
              },
            },
          })
        : await this.service.generateTextStream({
            input,
            parameters,
            ...this.scopeId(),
            ...requestOptions,
          });
      return textStream;
    } else {
      const textGenerationPromise = idOrName
        ? this.service.deploymentGenerateText({
            ...requestOptions,
            idOrName,
            parameters: {
              ...parameters,
              prompt_variables: {
                input,
              },
            },
          })
        : this.service.generateText({
            input,
            parameters,
            ...this.scopeId(),
            ...requestOptions,
          });

      try {
        const textGeneration = await textGenerationPromise;
        const singleGeneration: Generation[] =
          textGeneration.result.results.map((result) => {
            tokenUsage.generated_token_count += result.generated_token_count
              ? result.generated_token_count
              : 0;
            tokenUsage.input_token_count += result.input_token_count
              ? result.input_token_count
              : 0;
            return {
              text: result.generated_text,
              generationInfo: {
                stop_reason: result.stop_reason,
                ...tokenUsage,
              },
            };
          });
        return singleGeneration;
      } catch (err) {
        throw err;
      }
    }
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
    prompts: string[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): Promise<LLMResult> {
    const tokenUsage: TokenUsage = {
      generated_token_count: 0,
      input_token_count: 0,
    };
    if (this.streaming) {
      const generations: Generation[][] = await Promise.all(
        prompts.map(async (prompt, promptIdx) => {
          if (options.signal?.aborted) {
            throw new Error("AbortError");
          }
          const callback = () =>
            this.generateSingleMessage(prompt, options, true, tokenUsage);

          type ReturnMessage = ReturnType<typeof callback>;
          const stream = await this.completionWithRetry<ReturnMessage>(
            callback,
            options
          );

          const responseChunk: ResponseChunk = {
            id: 0,
            event: "",
            data: {
              results: [],
            },
          };
          const messages: ResponseChunk[] = [];
          type ResponseChunkKeys = keyof ResponseChunk;
          for await (const chunk of stream) {
            if (chunk.length > 0) {
              const index = chunk.indexOf(": ");
              const [key, value] = [
                chunk.substring(0, index) as ResponseChunkKeys,
                chunk.substring(index + 2),
              ];
              if (key === "id") {
                responseChunk[key] = Number(value);
              } else if (key === "event") {
                responseChunk[key] = String(value);
              } else {
                responseChunk[key] = JSON.parse(value);
              }
            } else if (chunk.length === 0) {
              messages.push(JSON.parse(JSON.stringify(responseChunk)));
              Object.assign(responseChunk, { id: 0, event: "", data: {} });
            }
          }

          const geneartionsArray: GenerationInfo[] = [];
          for (const message of messages) {
            message.data.results.forEach((item, index) => {
              const generationInfo: GenerationInfo = {
                text: "",
                stop_reason: "",
                generated_token_count: 0,
                input_token_count: 0,
              };
              void _runManager?.handleLLMNewToken(item.generated_text, {
                prompt: promptIdx,
                completion: 1,
              });
              geneartionsArray[index] ??= generationInfo;
              geneartionsArray[index].generated_token_count =
                item.generated_token_count;
              geneartionsArray[index].input_token_count +=
                item.input_token_count;
              geneartionsArray[index].stop_reason = item.stop_reason;
              geneartionsArray[index].text += item.generated_text;
            });
          }

          return geneartionsArray.map((item) => {
            const { text, ...rest } = item;
            tokenUsage.generated_token_count += rest.generated_token_count;
            tokenUsage.input_token_count += rest.input_token_count;
            return {
              text,
              generationInfo: rest,
            };
          });
        })
      );
      const result: LLMResult = { generations, llmOutput: { tokenUsage } };
      return result;
    } else {
      const generations: Generation[][] = await Promise.all(
        prompts.map(async (prompt) => {
          if (options.signal?.aborted) {
            throw new Error("AbortError");
          }

          const callback = () =>
            this.generateSingleMessage(prompt, options, false, tokenUsage);
          type ReturnMessage = ReturnType<typeof callback>;

          const response = await this.completionWithRetry<ReturnMessage>(
            callback,
            options
          );
          return response;
        })
      );
      const result: LLMResult = { generations, llmOutput: { tokenUsage } };
      return result;
    }
  }

  async getNumTokens(
    content: string,
    options?: TextTokenizeParameters
  ): Promise<number> {
    const params: TextTokenizationParams = {
      ...this.scopeId(),
      input: content,
      parameters: options,
    };
    const callback = () => this.service.tokenizeText(params);
    type ReturnTokens = ReturnType<typeof callback>;

    const response = await this.completionWithRetry<ReturnTokens>(callback);
    return response.result.result.token_count;
  }

  async *_streamResponseChunks(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    const callback = () => this.generateSingleMessage(prompt, options, true);
    type ReturnStream = ReturnType<typeof callback>;
    const streamInferDeployedPrompt =
      await this.completionWithRetry<ReturnStream>(callback);
    const responseChunk: ResponseChunk = {
      id: 0,
      event: "",
      data: {
        results: [],
      },
    };
    for await (const chunk of streamInferDeployedPrompt) {
      if (options.signal?.aborted) {
        throw new Error("AbortError");
      }

      type Keys = keyof typeof responseChunk;
      if (chunk.length > 0) {
        const index = chunk.indexOf(": ");
        const [key, value] = [
          chunk.substring(0, index) as Keys,
          chunk.substring(index + 2),
        ];
        if (key === "id") {
          responseChunk[key] = Number(value);
        } else if (key === "event") {
          responseChunk[key] = String(value);
        } else {
          responseChunk[key] = JSON.parse(value);
        }
      } else if (
        chunk.length === 0 &&
        responseChunk.data?.results?.length > 0
      ) {
        for (const item of responseChunk.data.results) {
          yield new GenerationChunk({
            text: item.generated_text,
            generationInfo: {
              stop_reason: item.stop_reason,
            },
          });
          await runManager?.handleLLMNewToken(item.generated_text ?? "");
        }
        Object.assign(responseChunk, { id: 0, event: "", data: {} });
      }
    }
  }

  _llmType() {
    return "watsonx";
  }
}
