import { WatsonxEmbeddings } from "@langchain/community/embeddings/ibm";

const instance = new WatsonxEmbeddings({
  version: "YYYY-MM-DD",
  serviceUrl: process.env.API_URL,
  projectId: "<PROJECT_ID>",
  spaceId: "<SPACE_ID>",
  idOrName: "<DEPLOYMENT_ID>",
  modelId: "<MODEL_ID>",
});

const result = await instance.embedQuery("Hello world!");
console.log(result);
