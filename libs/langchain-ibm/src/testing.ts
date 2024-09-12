import { WatsonX } from "./llms.js";
import dotenv from "dotenv";

dotenv.config();

const newInstance = new WatsonX({
  version: "2024-05-31",
  modelId: "google/flan-ul2",
  serviceUrl: "https://yp-qa.ml.cloud.ibm.com",
  projectId: process.env.PROJECT_ID,
});
(async () => {
  const result = await newInstance.stream("Hey what are your names?");
  for await (const chunk of result) {
    console.log(">" + chunk);
  }
})();
