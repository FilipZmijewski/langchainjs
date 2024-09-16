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
} from "@ibm-cloud/watsonx-ai/dist/watsonx-ai-ml/vml_v1.js";
import { Generation, LLMResult } from "@langchain/core/outputs";
import { GenerationChunk } from "@langchain/core/outputs";
import { authenticateAndSetInstance } from "./utilis/authentication.js";
import {
  TokenUsage,
  WatsonXAuth,
  WatsonXCallOptions,
  WatsonXParams,
} from "./types.js";

/**
 * Input to LLM class.
 */
interface WatsonXCallOptionsLLM extends WatsonXCallOptions {
  options?: Omit<
    TextGenerationParams &
      TextGenerationStreamParams &
      DeploymentsTextGenerationParams &
      DeploymentsTextGenerationStreamParams,
    "parameters"
  >;
}

export interface WatsonXInputLLM
  extends TextGenParameters,
    WatsonXParams,
    BaseLLMParams {}

/**
 * Integration with an LLM.
 */
export class WatsonX<
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
  max_new_tokens = 100;
  modelId = "ibm/granite-13b-chat-v2";
  serviceUrl: string;
  version: string;
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

  private serviceInstance: WatsonXAI;

  constructor(fields: WatsonXInputLLM & WatsonXAuth) {
    super(fields);
    this.modelId = fields.modelId ? fields.modelId : this.modelId;
    this.version = fields.version;
    this.max_new_tokens = fields.max_new_tokens
      ? fields.max_new_tokens
      : this.max_new_tokens;
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

    if (fields?.projectId) this.projectId = fields?.projectId;
    else if (fields?.spaceId) this.spaceId = fields?.spaceId;
    else if (fields?.idOrName) this.idOrName = fields?.idOrName;
    else
      throw new Error(
        "You need to pass either spaceId or projectId to proceed"
      );
    this.serviceUrl = fields?.serviceUrl;
    this.serviceInstance = authenticateAndSetInstance(fields);
  }

  get lc_secrets(): { [key: string]: string } | undefined {
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

  get lc_aliases(): { [key: string]: string } | undefined {
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

  ivocationParams(
    options: this["ParsedCallOptions"]
  ): TextGenParameters | DeploymentTextGenProperties {
    return {
      max_new_tokens: this.max_new_tokens,
      decoding_method: this.decoding_method,
      length_penalty: this.length_penalty,
      min_new_tokens: this.min_new_tokens,
      random_seed: this.random_seed,
      stop_sequences: options?.stop ?? this.stop_sequences,
      temperature: this.temperature,
      time_limit: this.time_limit,
      top_k: this.top_k,
      top_p: this.top_p,
      repetition_penalty: this.repetition_penalty,
      truncate_input_tokens: this.truncate_input_tokens,
      return_options: this.return_options,
      include_stop_sequence: this.include_stop_sequence,
    };
  }

  scopeId() {
    if (this.projectId) return { projectId: this.projectId };
    else if (this.spaceId) return { spaceId: this.spaceId };
    else return undefined;
  }

  private async generateSingleMessage(
    input: string,
    options: this["ParsedCallOptions"],
    tokenUsage: TokenUsage
  ) {
    const requestOptions = options.options ?? undefined;
    const idOrName = options.options?.idOrName ?? this.idOrName;
    const textGeneration = idOrName
      ? await this.serviceInstance.deploymentGenerateText({
          ...requestOptions,
          idOrName: idOrName,
          parameters: {
            ...this.invocationParams(),
            prompt_variables: {
              input,
            },
          },
        })
      : await this.serviceInstance.generateText({
          input,
          parameters: this.invocationParams(),
          modelId: this.modelId,
          ...requestOptions,
          ...this.scopeId(),
        });

    const singleGeneration: Generation[] = textGeneration.result.results.map(
      (result) => {
        tokenUsage.generated_token_count += result.generated_token_count
          ? result.generated_token_count
          : 0;
        tokenUsage.input_token_count += result.input_token_count
          ? result.input_token_count
          : 0;
        return {
          text: result.generated_text,
          generationInfo: { stop_reason: result.stop_reason },
        };
      }
    );
    return singleGeneration;
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
    const generations = await Promise.all(
      prompts.map(async (prompt) => {
        if (options.signal?.aborted) {
          throw new Error("AbortError");
        }
        return await this.generateSingleMessage(prompt, options, tokenUsage);
      })
    );
    const result: LLMResult = { generations };
    return result;
  }

  async *_streamResponseChunks(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    const requestOptions = options.options ?? {};
    const idOrName = options.options?.idOrName ?? this.idOrName;

    const streamInferDeployedPrompt = idOrName
      ? await this.serviceInstance.deploymentGenerateTextStream({
          input: prompt,
          parameters: {
            ...this.invocationParams(),
            prompt_variables: {
              input: prompt,
            },
          },
          ...this.scopeId(),
          idOrName: idOrName,
          ...requestOptions,
        })
      : await this.serviceInstance.generateTextStream({
          input: prompt,
          parameters: this.invocationParams(),
          modelId: this.modelId,
          ...this.scopeId(),
          ...requestOptions,
        });
    for await (const chunk of streamInferDeployedPrompt as any) {
      if (options.signal?.aborted) {
        throw new Error("AbortError");
      }
      yield new GenerationChunk({
        text: chunk,
      });
      await runManager?.handleLLMNewToken(chunk.response ?? "");
    }
  }

  _llmType() {
    return "watsonx";
  }
}
