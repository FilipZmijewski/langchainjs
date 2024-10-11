import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import {
  IamAuthenticator,
  BearerTokenAuthenticator,
  CloudPakForDataAuthenticator,
} from "ibm-cloud-sdk-core";
import { WatsonxAuth, WatsonxInit } from "../types/watsonx_ai.js";

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
