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
