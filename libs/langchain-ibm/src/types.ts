import { BaseLanguageModelCallOptions } from "@langchain/core/language_models/base";

export interface WatsonXCallOptions extends BaseLanguageModelCallOptions {
  options?: { [key: string]: any };
}
export interface TokenUsage {
  generated_token_count: number;
  input_token_count: number;
}
export interface WatsonXAuth {
  watsonxAIApikey?: string;
  watsonxAIBearerToken?: string;
  watsonxAIUsername?: string;
  watsonxAIPassword?: string;
  watsonxAIUrl?: string;
  watsonxAIAuthType?: string;
}

export interface WatsonXInit {
  authenticator?: string;
  serviceUrl: string;
  version: string;
}

export interface WatsonXParams extends WatsonXInit {
  modelId?: string;
  spaceId?: string;
  projectId?: string;
  idOrName?: string;
}
