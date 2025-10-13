/* eslint-disable no-process-env */
import { test, expect } from "@jest/globals";
import { ChatModelIntegrationTests } from "@langchain/standard-tests";
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
} from "@langchain/core/messages";
import { RunnableLambda } from "@langchain/core/runnables";
import z from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Serialized } from "@langchain/core/load/serializable";
import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import { WatsonxAuth } from "../../types/ibm.js";
import {
  ChatWatsonx,
  ChatWatsonxInput,
  WatsonxCallOptionsChat,
  WatsonxCallParams,
} from "../ibm.js";

const MATH_ADDITION_PROMPT = /* #__PURE__ */ ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are bad at math and must ALWAYS call the {toolName} function.",
  ],
  ["human", "What is the sum of 1836281973 and 19973286?"],
]);

const adderSchema = /* #__PURE__ */ z
  .object({
    a: z.number().int().describe("The first integer to add."),
    b: z.number().int().describe("The second integer to add."),
  })
  .describe("Add two integers");

export class TestCallbackHandler extends BaseCallbackHandler {
  name = "TestCallbackHandler";

  /**
   * Internal array to store extra parameters from each chat model start event.
   * @internal
   */
  _extraParams: Array<Record<string, unknown>> = [];

  /**
   * Returns a single object containing all accumulated extra parameters,
   * merged together. If multiple runs provide extra parameters, later
   * values will overwrite earlier ones for the same keys.
   *
   * @returns {Record<string, unknown>} The merged extra parameters.
   */
  get extraParams(): Record<string, unknown> {
    return this._extraParams.reduce(Object.assign, {});
  }

  handleChatModelStart(
    _llm: Serialized,
    _messages: BaseMessage[][],
    _runId: string,
    _parentRunId?: string,
    extraParams?: Record<string, unknown>,
    _tags?: string[],
    _metadata?: Record<string, unknown>,
    _runName?: string
  ) {
    console.log(extraParams);
    if (extraParams) this._extraParams.push(extraParams);
  }
}

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
        model: "meta-llama/llama-3-3-70b-instruct",
        version: "2024-05-31",
        serviceUrl: process.env.WATSONX_AI_SERVICE_URL ?? "testString",
        projectId: process.env.WATSONX_AI_PROJECT_ID ?? "testString",
        temperature: 0,
      },
    });
  }

  async testInvokeMoreComplexTools() {
    this.skipTestMessage(
      "testInvokeMoreComplexTools",
      "ChatWatsonx",
      "Watsonx does not support tool schemas which contain object with unknown/any parameters." +
        "Watsonx only supports objects in schemas when the parameters are defined."
    );
  }

  async testBindToolsWithRunnableToolLike(callOptions?: any) {
    if (!this.chatModelHasToolCalling) {
      console.log("Test requires tool calling. Skipping...");
      return;
    }

    const model = new this.Cls(this.constructorArgs);
    if (!model.bindTools) {
      throw new Error(
        "bindTools undefined. Cannot test Runnable-like tool calls."
      );
    }

    const runnableLike = RunnableLambda.from((_) => {}).asTool({
      name: "math_addition",
      description: adderSchema.description,
      schema: adderSchema,
    });

    const modelWithTools = model.bindTools([runnableLike]);

    const result: AIMessage = await MATH_ADDITION_PROMPT.pipe(
      modelWithTools
    ).invoke(
      {
        toolName: "math_addition",
      },
      callOptions
    );

    expect(result.tool_calls?.[0]).toBeDefined();
    if (!result.tool_calls?.[0]) {
      throw new Error("result.tool_calls is undefined");
    }
    const { tool_calls } = result;

    expect(tool_calls).toHaveLength(1);

    const toolCall = tool_calls[0];

    expect(toolCall.name).toBe("math_addition");

    expect(toolCall.args).toEqual({
      a: expect.any(Number),
      b: expect.any(Number),
    });

    expect(toolCall.id).toBeDefined();

    expect(toolCall.type).toBe("tool_call");
  }

  async testWithStructuredOutput() {
    this.skipTestMessage(
      "testWithStructuredOutput",
      "ChatWatsonx",
      "Assertion ```expect(handler.extraParams)``` is not valid in ChatWatsonx"
    );
  }

  async testWithStructuredOutputIncludeRaw() {
    this.skipTestMessage(
      "testWithStructuredOutputIncludeRaw",
      "ChatWatsonx",
      "Assertion ```expect(handler.extraParams)``` is not valid in ChatWatsonx"
    );
  }
}

const testClass = new ChatWatsonxStandardIntegrationTests();

test("ChatWatsonxStandardIntegrationTests", async () => {
  const testResults = await testClass.runTests();
  expect(testResults).toBe(true);
});
