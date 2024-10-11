/* eslint-disable no-process-env */
import { test, expect } from "@jest/globals";
import { ChatModelIntegrationTests } from "@langchain/standard-tests";
import { AIMessageChunk } from "@langchain/core/messages";
import {
  ChatWatsonx,
  ChatWatsonxInput,
  WatsonxCallOptionsChat,
  WatsonxCallParams,
} from "../ibm.js";
import { WatsonxAuth } from "../../types/watsonx_ai.js";

class ChatWatsonxStandardIntegrationTests extends ChatModelIntegrationTests<
  WatsonxCallOptionsChat,
  AIMessageChunk,
  ChatWatsonxInput &
    WatsonxAuth &
    Partial<Omit<WatsonxCallParams, "tool_choice">>
> {
  constructor() {
    if (!process.env.WATSONX_AI_APIKEY) {
      throw new Error("Cannot run tests. Api key not provided");
    }
    super({
      Cls: ChatWatsonx,
      chatModelHasToolCalling: true,
      chatModelHasStructuredOutput: true,
      constructorArgs: {
        version: "2024-05-31",
        serviceUrl: process.env.WATSONX_AI_SERVICE_URL as string,
        projectId: process.env.WATSONX_AI_PROJECT_ID,
        temperature: 0,
      },
    });
  }

  async testCacheComplexMessageTypes() {
    this.skipTestMessage(
      "testStructuredFewShotExamples",
      "ChatWatsonx",
      "Model does not support non-string content"
    );
  }
}

const testClass = new ChatWatsonxStandardIntegrationTests();

test("ChatWatsonxStandardIntegrationTests", async () => {
  const testResults = await testClass.runTests();
  expect(testResults).toBe(true);
});
