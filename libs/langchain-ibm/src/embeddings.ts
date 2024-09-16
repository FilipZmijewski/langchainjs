import { Embeddings } from "@langchain/core/embeddings";
import { WatsonXAuth, WatsonXParams } from "./types.js";
import {
  EmbeddingParameters,
  EmbeddingReturnOptions,
  TextEmbeddingsParams,
} from "@ibm-cloud/watsonx-ai/dist/watsonx-ai-ml/vml_v1.js";
import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import { authenticateAndSetInstance } from "./utilis/authentication.js";

interface WatsonXEmbeddingsParams
  extends WatsonXParams,
    EmbeddingParameters,
    Pick<TextEmbeddingsParams, "headers"> {
  maxConcurrency?: number;
}

export class WatsonXEmbeddings
  extends Embeddings
  implements WatsonXEmbeddingsParams
{
  modelId = "ibm/slate-125m-english-rtrvr";
  serviceUrl: string;
  version: string;
  spaceId?: string;
  projectId?: string;
  truncate_input_tokens?: number;
  return_options?: EmbeddingReturnOptions;
  private serviceInstance: WatsonXAI;

  constructor(fields: WatsonXEmbeddingsParams & WatsonXAuth) {
    const superProps = { maxConcurrency: 2, ...fields };
    super(superProps);
    this.modelId = fields?.modelId ? fields.modelId : this.modelId;
    this.version = fields.version;
    this.serviceUrl = fields.serviceUrl;
    this.truncate_input_tokens = fields.truncate_input_tokens;
    this.return_options = fields.return_options;
    if (fields?.projectId) this.projectId = fields?.projectId;
    else if (fields?.spaceId) this.spaceId = fields?.spaceId;
    else
      throw new Error(
        "You need to pass either spaceId or projectId to proceed"
      );
    this.serviceUrl = fields?.serviceUrl;

    this.serviceInstance = authenticateAndSetInstance(fields);
  }
  scopeId() {
    if (this.projectId) return { projectId: this.projectId };
    else return { spaceId: this.spaceId };
  }

  ivocationParams(): EmbeddingParameters {
    return {
      truncate_input_tokens: this.truncate_input_tokens,
      return_options: this.return_options,
    };
  }

  async listModels() {
    const listModelParams = {
      filters: "function_embedding",
    };
    const listModels = await this.serviceInstance.listFoundationModelSpecs(
      listModelParams
    );
    return listModels.result.resources?.map((item) => item.model_id);
  }

  private async embedSingleText(inputs: string[]) {
    const textEmbeddingParams: TextEmbeddingsParams = {
      inputs,
      modelId: this.modelId,
      ...this.scopeId(),
      parameters: this.ivocationParams(),
    };
    const embeddings = await this.serviceInstance.embedText(
      textEmbeddingParams
    );
    return embeddings.result.results.map((item) => item.embedding);
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    const data = await this.embedSingleText(documents);
    return data;
  }
  async embedQuery(document: string): Promise<number[]> {
    const data = await this.embedSingleText([document]);
    return data[0];
  }
}
