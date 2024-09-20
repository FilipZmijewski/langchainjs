import { WatsonXEmbeddings, WatsonXEmbeddingsParams } from "../embeddings.js";
import { testProperties } from "./utilis.js";

describe("Embeddings unit tests", () => {
  describe("Positive tests", () => {
    test("Basic properties", () => {
      const testProps: WatsonXEmbeddingsParams = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      };
      const instance = new WatsonXEmbeddings(testProps);
      testProperties(instance, testProps);
    });

    test("Basic properties", () => {
      const testProps: WatsonXEmbeddingsParams = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        truncate_input_tokens: 10,
        maxConcurrency: 2,
        maxRetries: 2,
        modelId: "ibm/slate-125m-english-rtrvr",
      };
      const instance = new WatsonXEmbeddings(testProps);

      testProperties(instance, testProps);
    });
  });

  describe("Negative tests", () => {
    test("Missing id", async () => {
      const testProps: WatsonXEmbeddingsParams = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
      };
      expect(
        () =>
          new WatsonXEmbeddings({
            ...testProps,
          })
      ).toThrowError();
    });

    test("Missing other props", async () => {
      // @ts-ignore
      const testPropsProjectId: WatsonXInputLLM = {
        projectId: process.env.PROJECT_ID,
      };
      expect(
        () =>
          new WatsonXEmbeddings({
            ...testPropsProjectId,
          })
      ).toThrowError();
      // @ts-ignore
      const testPropsServiceUrl: WatsonXInputLLM = {
        serviceUrl: process.env.API_URL as string,
      };
      expect(
        () =>
          new WatsonXEmbeddings({
            ...testPropsServiceUrl,
          })
      ).toThrowError();
      const testPropsVersion = {
        version: "2024-05-31",
      };
      expect(
        () =>
          new WatsonXEmbeddings({
            // @ts-ignore
            testPropsVersion,
          })
      ).toThrowError();
    });

    test("Passing more than one id", async () => {
      const testProps: WatsonXEmbeddingsParams = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        spaceId: process.env.PROJECT_ID,
      };
      expect(
        () =>
          new WatsonXEmbeddings({
            ...testProps,
          })
      ).toThrowError();
    });

    test("Invalid properties", () => {
      const testProps: WatsonXEmbeddingsParams = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      };
      const notExTestProps = {
        notExisting: 12,
        notExObj: {
          notExProp: 12,
        },
      };
      const instance = new WatsonXEmbeddings({
        ...testProps,
        ...notExTestProps,
      });

      testProperties(instance, testProps, notExTestProps);
    });
  });
});
