/* eslint-disable @typescript-eslint/no-explicit-any */
import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import {
  IamAuthenticator,
  BearerTokenAuthenticator,
  CloudPakForDataAuthenticator,
} from "ibm-cloud-sdk-core";
import {
  JsonOutputKeyToolsParserParams,
  JsonOutputToolsParser,
} from "@langchain/core/output_parsers/openai_tools";
import { OutputParserException } from "@langchain/core/output_parsers";
import { z } from "zod";
import { ChatGeneration } from "@langchain/core/outputs";
import { AIMessageChunk } from "@langchain/core/messages";
import { ToolCall } from "@langchain/core/messages/tool";
import {
  CallbackManager,
  CallbackManagerForLLMRun,
} from "@langchain/core/callbacks/manager";
import {
  InvokeRequestCallback,
  RecieveResponseCallback,
  RequestCallbacks,
} from "@ibm-cloud/watsonx-ai/dist/watsonx-ai-ml/vml_v1.js";
import {
  BaseCallbackHandler,
  CallbackHandlerMethods,
} from "@langchain/core/callbacks/base";
import { WatsonxAuth, WatsonxInit } from "../types/ibm.js";

export const authenticateAndSetInstance = ({
  watsonxAIApikey,
  watsonxAIAuthType,
  watsonxAIBearerToken,
  watsonxAIUsername,
  watsonxAIPassword,
  watsonxAIUrl,
  version,
  serviceUrl,
}: WatsonxAuth & Omit<WatsonxInit, "authenticator">): WatsonXAI | undefined => {
  if (watsonxAIAuthType === "iam" && watsonxAIApikey) {
    return WatsonXAI.newInstance({
      version,
      serviceUrl,
      authenticator: new IamAuthenticator({
        apikey: watsonxAIApikey,
      }),
    });
  } else if (watsonxAIAuthType === "bearertoken" && watsonxAIBearerToken) {
    return WatsonXAI.newInstance({
      version,
      serviceUrl,
      authenticator: new BearerTokenAuthenticator({
        bearerToken: watsonxAIBearerToken,
      }),
    });
  } else if (watsonxAIAuthType === "cp4d" && watsonxAIUrl) {
    if (watsonxAIUsername && watsonxAIPassword && watsonxAIApikey)
      return WatsonXAI.newInstance({
        version,
        serviceUrl,
        authenticator: new CloudPakForDataAuthenticator({
          username: watsonxAIUsername,
          password: watsonxAIPassword,
          url: watsonxAIUrl,
          apikey: watsonxAIApikey,
        }),
      });
  } else
    return WatsonXAI.newInstance({
      version,
      serviceUrl,
    });
  return undefined;
};

// Mistral enforces a specific pattern for tool call IDs
// Thanks to Mistral for implementing this, I was unable to import which is why this is copied 1:1
const TOOL_CALL_ID_PATTERN = /^[a-zA-Z0-9]{9}$/;

export function _isValidMistralToolCallId(toolCallId: string): boolean {
  return TOOL_CALL_ID_PATTERN.test(toolCallId);
}

function _base62Encode(num: number): string {
  let numCopy = num;
  const base62 =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  if (numCopy === 0) return base62[0];
  const arr: string[] = [];
  const base = base62.length;
  while (numCopy) {
    arr.push(base62[numCopy % base]);
    numCopy = Math.floor(numCopy / base);
  }
  return arr.reverse().join("");
}

function _simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i += 1) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash &= hash; // Convert to 32-bit integer
  }
  return Math.abs(hash);
}

export function _convertToolCallIdToMistralCompatible(
  toolCallId: string
): string {
  if (_isValidMistralToolCallId(toolCallId)) {
    return toolCallId;
  } else {
    const hash = _simpleHash(toolCallId);
    const base62Str = _base62Encode(hash);
    if (base62Str.length >= 9) {
      return base62Str.slice(0, 9);
    } else {
      return base62Str.padStart(9, "0");
    }
  }
}

interface WatsonxToolsOutputParserParams<T extends Record<string, any>>
  extends JsonOutputKeyToolsParserParams<T> {}

export class WatsonxToolsOutputParser<
  T extends Record<string, any> = Record<string, any>
> extends JsonOutputToolsParser<T> {
  static lc_name() {
    return "WatsonxToolsOutputParser";
  }

  lc_namespace = ["langchain", "watsonx", "output_parsers"];

  returnId = false;

  keyName: string;

  returnSingle = false;

  zodSchema?: z.ZodType<T>;

  latestCorrect?: ToolCall;

  constructor(params: WatsonxToolsOutputParserParams<T>) {
    super(params);
    this.keyName = params.keyName;
    this.returnSingle = params.returnSingle ?? this.returnSingle;
    this.zodSchema = params.zodSchema;
  }

  protected async _validateResult(result: unknown): Promise<T> {
    let parsedResult = result;
    if (typeof result === "string") {
      try {
        parsedResult = JSON.parse(result);
      } catch (e: any) {
        throw new OutputParserException(
          `Failed to parse. Text: "${JSON.stringify(
            result,
            null,
            2
          )}". Error: ${JSON.stringify(e.message)}`,
          result
        );
      }
    } else {
      parsedResult = result;
    }
    if (this.zodSchema === undefined) {
      return parsedResult as T;
    }
    const zodParsedResult = await this.zodSchema.safeParseAsync(parsedResult);
    if (zodParsedResult.success) {
      return zodParsedResult.data;
    } else {
      throw new OutputParserException(
        `Failed to parse. Text: "${JSON.stringify(
          result,
          null,
          2
        )}". Error: ${JSON.stringify(zodParsedResult.error.errors)}`,
        JSON.stringify(result, null, 2)
      );
    }
  }

  async parsePartialResult(generations: ChatGeneration[]): Promise<T> {
    const tools = generations.flatMap((generation) => {
      const message = generation.message as AIMessageChunk;
      if (!Array.isArray(message.tool_calls)) {
        return [];
      }
      const tool = message.tool_calls;
      return tool;
    });
    if (tools[0] === undefined) {
      if (this.latestCorrect) tools.push(this.latestCorrect);
    }
    const [tool] = tools;
    this.latestCorrect = tool;
    return tool.args as T;
  }
}

type RequestCallbackKeys = keyof RequestCallbacks;
export class WatsonxCallbackManager extends CallbackManager {
  constructor(
    parentRunId: string | undefined,
    callbackManager: CallbackManager
  ) {
    super(parentRunId, { ...callbackManager });
  }

  static fromHandlers(
    handlers: CallbackHandlerMethods & RequestCallbacks
  ): WatsonxCallbackManager {
    const watsonxCallbackNames: RequestCallbackKeys[] = [
      "responseCallback",
      "requestCallback",
    ];
    const watsonxHandlers: Partial<
      Record<
        RequestCallbackKeys,
        InvokeRequestCallback | RecieveResponseCallback
      >
    > = {};
    watsonxCallbackNames.forEach((item) => {
      watsonxHandlers[item] = handlers[item];
      Reflect.deleteProperty(watsonxHandlers, item);
    });
    const manager = super.fromHandlers(handlers);
    class WatsonxHandler extends BaseCallbackHandler {
      name = "watsonxHandler";

      requestCallback: InvokeRequestCallback;

      responseCallback: RecieveResponseCallback;

      constructor() {
        super();
        Object.assign(this, watsonxHandlers);
      }

      get watsonxHandlers() {
        return {
          requestCallback: this.requestCallback,
          responseCallback: this.responseCallback,
        };
      }
    }

    manager.addHandler(new WatsonxHandler());
    return manager;
  }
}

export class WatsonxBaseCallbackHandler extends BaseCallbackHandler {
  name: string;

  constructor() {
    super();
  }

  requestCallback: InvokeRequestCallback;

  responseCallback: RecieveResponseCallback;

  get watsonxHandlers() {
    if (this.name === "watsonxHandler")
      return {
        requestCallback: this.requestCallback,
        responseCallback: this.responseCallback,
      };
    else return undefined;
  }
}
export class WatsonxCallbackManagerForLLMRun extends CallbackManagerForLLMRun {
  declare handlers: WatsonxBaseCallbackHandler[];
}