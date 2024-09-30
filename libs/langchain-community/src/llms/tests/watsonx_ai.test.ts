/* eslint-disable no-process-env */
import WatsonxAiMlVml_v1 from "@ibm-cloud/watsonx-ai/dist/watsonx-ai-ml/vml_v1.js";
import { WatsonxLLM, WatsonxInputLLM } from "../watsonx_ai.js";
import { authenticateAndSetInstance } from "../../utils/watsonx_ai.js";
import {
  WatsonxEmbeddings,
  WatsonxEmbeddingsParams,
} from "../../embeddings/watsonx_ai.js";

export function getKey<K>(key: K): K {
  return key;
}
export const testProperties = (
  instance: WatsonxLLM | WatsonxEmbeddings,
  testProps: WatsonxInputLLM,
  notExTestProps?: { [key: string]: any }
) => {
  const checkProperty = <T extends { [key: string]: any }>(
    testProps: T,
    instance: T,
    existing = true
  ) => {
    Object.keys(testProps).forEach((key) => {
      const keys = getKey<keyof T>(key);
      type Type = Pick<T, typeof keys>;

      if (typeof testProps[key as keyof T] === "object")
        checkProperty<Type>(testProps[key as keyof T], instance[key], existing);
      else {
        if (existing)
          expect(instance[key as keyof T]).toBe(testProps[key as keyof T]);
        else if (instance) expect(instance[key as keyof T]).toBeUndefined();
      }
    });
  };
  checkProperty<WatsonxEmbeddingsParams>(testProps, instance);
  if (notExTestProps)
    checkProperty<typeof notExTestProps>(notExTestProps, instance, false);
};

describe("LLM unit tests", () => {
  describe("Positive tests", () => {
    test("Test authentication function", () => {
      const instance = authenticateAndSetInstance({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
      });
      expect(instance).toBeInstanceOf(WatsonxAiMlVml_v1);
    });

    test("Test basic properties after init", async () => {
      const testProps: WatsonxInputLLM = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      };
      const instance = new WatsonxLLM(testProps);

      testProperties(instance, testProps);
    });

    test("Test methods after init", () => {
      const testProps: WatsonxInputLLM = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      };
      const instance = new WatsonxLLM({
        ...testProps,
      });
      expect(instance.getNumTokens).toBeDefined();
      expect(instance._generate).toBeDefined();
      expect(instance._streamResponseChunks).toBeDefined();
      expect(instance.invocationParams).toBeDefined();
    });

    test("Test properties after init", async () => {
      const testProps = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        modelId: "ibm/granite-13b-chat-v2",
        max_new_tokens: 100,
        decoding_method: "sample",
        length_penalty: { decay_factor: 1, start_index: 1 },
        min_new_tokens: 10,
        random_seed: 1,
        stop_sequences: ["hello"],
        temperature: 0.1,
        time_limit: 10000,
        top_k: 1,
        top_p: 1,
        repetition_penalty: 1,
        truncate_input_tokens: 1,
        return_options: {
          input_text: true,
          generated_tokens: true,
          input_tokens: true,
          token_logprobs: true,
          token_ranks: true,

          top_n_tokens: 2,
        },
        include_stop_sequence: false,
        maxRetries: 3,
        maxConcurrency: 3,
      };
      const instance = new WatsonxLLM(testProps);

      testProperties(instance, testProps);
    });
  });

  describe("Negative tests", () => {
    test("Missing id", async () => {
      const testProps: WatsonxInputLLM = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
      };
      expect(
        () =>
          new WatsonxLLM({
            ...testProps,
          })
      ).toThrowError();
    });

    test("Missing other props", async () => {
      // @ts-expect-error Intentionally passing not enough parameters
      const testPropsProjectId: WatsonxInputLLM = {
        projectId: process.env.PROJECT_ID,
      };
      expect(
        () =>
          new WatsonxLLM({
            ...testPropsProjectId,
          })
      ).toThrowError();
      // @ts-expect-error Intentionally passing not enough parameters
      const testPropsServiceUrl: WatsonxInputLLM = {
        serviceUrl: process.env.API_URL as string,
      };
      expect(
        () =>
          new WatsonxLLM({
            ...testPropsServiceUrl,
          })
      ).toThrowError();
      const testPropsVersion = {
        version: "2024-05-31",
      };
      expect(
        () =>
          new WatsonxLLM({
            // @ts-expect-error Intentionally passing wrong type of an object
            testPropsVersion,
          })
      ).toThrowError();
    });

    test("Passing more than one id", async () => {
      const testProps: WatsonxInputLLM = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        spaceId: process.env.PROJECT_ID,
      };
      expect(
        () =>
          new WatsonxLLM({
            ...testProps,
          })
      ).toThrowError();
    });

    test("Not existing property passed", async () => {
      const testProps = {
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
      const instance = new WatsonxLLM({ ...testProps, ...notExTestProps });
      testProperties(instance, testProps, notExTestProps);
    });
  });
});
