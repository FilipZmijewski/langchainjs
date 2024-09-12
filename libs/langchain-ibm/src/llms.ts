import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  BaseLLM,
  type BaseLLMParams,
} from "@langchain/core/language_models/llms";
import { type BaseLanguageModelCallOptions } from "@langchain/core/language_models/base";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import {
  ReturnOptionProperties,
  TextGenerationParams,
  TextGenLengthPenalty,
  TextGenParameters,
} from "@ibm-cloud/watsonx-ai/dist/watsonx-ai-ml/vml_v1.js";
import { Generation, LLMResult } from "@langchain/core/outputs";
import { GenerationChunk } from "@langchain/core/outputs";

/**
 * Input to LLM class.
 */

export interface WatsonXAuth {
  apiKey?: string;
  watsonxAIApikey?: string;
  watsonxAIBearerToken?: string;
  watsonxAIUsername?: string;
  watsonxAIPassword?: string;
  watsonxAIUrl?: string;
  watsonxAIAuthType?: string;
}

export interface WatsonXInput extends BaseLLMParams, TextGenParameters {
  modelId: string;
  serviceUrl: string;
  version: string;
  spaceId?: string;
  projectId?: string;
}

/**
 * Integration with an LLM.
 */
export class WatsonX
  extends BaseLLM<BaseLanguageModelCallOptions>
  implements WatsonXInput
{
  // Used for tracing, replace with the same name as your class
  static lc_name() {
    return "WatsonX";
  }
  lc_serializable = true;
  max_new_tokens = 100;
  modelId: string;
  serviceUrl: string;
  version: string;
  spaceId?: string;
  projectId?: string;
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

  constructor(fields: WatsonXInput & WatsonXAuth) {
    super(fields);
    this.modelId = fields.modelId;
    this.version = fields.version;
    console.log(fields);
    if (fields?.projectId) this.projectId = fields?.projectId;
    else if (fields?.spaceId) this.spaceId = fields?.spaceId;
    else
      throw new Error(
        "You need to pass either spaceId or projectId to proceed"
      );
    this.serviceUrl = fields?.serviceUrl;
    const authType = getEnvironmentVariable("WATSONX_AI_AUTH_TYPE");
    const bearerToken = getEnvironmentVariable("WATSONX_AI_BEARER_TOKEN");
    const apiKey = getEnvironmentVariable("WATSONX_AI_APIKEY");
    const credentials =
      getEnvironmentVariable("WATSONX_AI_USERNAME") &&
      getEnvironmentVariable("WATSONX_AI_PASSWORD") &&
      getEnvironmentVariable("WATSONX_AI_URL");
    if (!authType || (!credentials && !bearerToken && !apiKey)) {
      const chosenAuthType = fields?.watsonxAIAuthType;
      if (!chosenAuthType)
        throw new Error(
          "No authentication type chosen. Check your enviromental variables or passed options"
        );
      process.env["WATSONX_AI_AUTH_TYPE"] = chosenAuthType;

      const chosenBearerToken = fields?.watsonxAIBearerToken;
      const chosenApiKey = fields?.watsonxAIApikey;
      const chosenUsername = fields?.watsonxAIUsername;
      const chosenPassword = fields?.watsonxAIPassword;
      const chosenUrl = fields?.watsonxAIUrl;
      if (chosenBearerToken)
        process.env["WATSONX_AI_BEARER_TOKEN"] = chosenBearerToken;
      else if (chosenApiKey) process.env["WATSONX_AI_APIKEY"] = chosenApiKey;
      else if (chosenUsername && chosenPassword && chosenUrl) {
        process.env["WATSONX_AI_USERNAME"] = chosenUsername;
        process.env["WATSONX_AI_PASSWORD"] = chosenPassword;
        process.env["WATSONX_AI_URL"] = chosenUrl;
      } else
        throw new Error(
          "You have not provided any form of authentication, please check passed arguments"
        );
    }
    this.serviceInstance = WatsonXAI.newInstance({
      version: this.version,
      serviceUrl: this.serviceUrl,
    });
  }

  /**
   * Replace with any secrets this class passes to `super`.
   * See {@link ../../langchain-cohere/src/chat_model.ts} for
   * an example.
   */
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

  /**
   * For some given input string and options, return a string output.
   */

  ivocationParams(): TextGenParameters {
    return {
      max_new_tokens: this.max_new_tokens,
      decoding_method: this.decoding_method,
      length_penalty: this.length_penalty,
      min_new_tokens: this.min_new_tokens,
      random_seed: this.random_seed,
      stop_sequences: this.stop_sequences,
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

  //one of these will awlays exist
  scopeId(): { projectId: string } | { spaceId: string } {
    return this.projectId
      ? { projectId: this.projectId }
      : { spaceId: this.spaceId as string };
  }

  private async generateSingleMessage(input: string) {
    //one of these always exists, it is guaranteed upper

    const params: TextGenerationParams = {
      input,
      parameters: this.invocationParams(),
      modelId: this.modelId,
      ...this.scopeId(),
    };

    try {
      const textGeneration = await this.serviceInstance.generateText(params);
      const singleGeneration: Generation[] = textGeneration.result.results.map(
        (result) => {
          return { text: result.generated_text, generationInfo: result };
        }
      );
      return singleGeneration;
    } catch (err) {
      console.warn(err);
      throw new Error("There was an error generating your text.");
    }
  }

  async _generate(
    prompts: string[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): Promise<LLMResult> {
    const generations = await Promise.all(
      prompts.map(async (prompt) => await this.generateSingleMessage(prompt))
    );
    const result: LLMResult = { generations };
    return result;
  }

  /**
   * Implement to support streaming.
   * Should yield chunks iteratively.
   */
  async *_streamResponseChunks(
    prompt: string,
    _options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    const streamInferDeployedPrompt =
      await this.serviceInstance.generateTextStream({
        input: prompt,
        parameters: this.invocationParams(),
        modelId: this.modelId,
        ...this.scopeId(),
      });

    for await (const chunk of streamInferDeployedPrompt as any) {
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
