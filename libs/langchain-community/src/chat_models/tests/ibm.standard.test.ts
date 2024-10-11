/* eslint-disable no-process-env */
import { test, expect } from "@jest/globals";
import { ChatModelUnitTests } from "@langchain/standard-tests";
import { AIMessageChunk } from "@langchain/core/messages";
import { LangSmithParams } from "@langchain/core/language_models/chat_models";
import {
  ChatWatsonx,
  ChatWatsonxInput,
  WatsonxCallOptionsChat,
  WatsonxCallParams,
} from "../ibm.js";
import { WatsonxAuth } from "../../types/watsonx_ai.js";

class ChatOpenAIStandardUnitTests extends ChatModelUnitTests<
  WatsonxCallOptionsChat,
  AIMessageChunk,
  ChatWatsonxInput &
    WatsonxAuth &
    Partial<Omit<WatsonxCallParams, "tool_choice">>
> {
  constructor() {
    super({
      Cls: ChatWatsonx,
      chatModelHasToolCalling: true,
      chatModelHasStructuredOutput: true,
      constructorArgs: {
        watsonxAIApikey: "testString",
        version: "2024-05-31",
        serviceUrl: process.env.WATSONX_AI_SERVICE_URL ?? "testString",
        projectId: process.env.WATSONX_AI_PROJECT_ID ?? "testString",
      },
    });
  }

  expectedLsParams(): Partial<LangSmithParams> {
    console.warn(
      "ChatWatsonx does not support stop sequences. Overwrite params."
    );
    return {
      ls_provider: "watsonx",
      ls_model_name: "string",
      ls_model_type: "chat",
      ls_temperature: 0,
      ls_max_tokens: 0,
    };
  }
}

const testClass = new ChatOpenAIStandardUnitTests();

test("ChatOpenAIStandardUnitTests", () => {
  const testResults = testClass.runTests();
  expect(testResults).toBe(true);
});
