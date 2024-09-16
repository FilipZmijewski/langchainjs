import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { WatsonXInit } from "../llms.js";
import { WatsonXAuth } from "../types.js";
// @ts-ignore
import { IamAuthenticator } from "ibm-cloud-sdk-core";

export const authenticateAndSetInstance = ({
  watsonxAIApikey,
  watsonxAIAuthType,
  watsonxAIBearerToken,
  watsonxAIUsername,
  watsonxAIPassword,
  watsonxAIUrl,
  version,
  serviceUrl,
  authenticator,
}: WatsonXAuth & WatsonXInit): WatsonXAI => {
  if (authenticator)
    return WatsonXAI.newInstance({
      version,
      serviceUrl,
      authenticator: new IamAuthenticator({ apikey: authenticator }),
    });
  const authType = getEnvironmentVariable("WATSONX_AI_AUTH_TYPE");
  const bearerToken = getEnvironmentVariable("WATSONX_AI_BEARER_TOKEN");
  const apiKey = getEnvironmentVariable("WATSONX_AI_APIKEY");
  const credentials =
    getEnvironmentVariable("WATSONX_AI_USERNAME") &&
    getEnvironmentVariable("WATSONX_AI_PASSWORD") &&
    getEnvironmentVariable("WATSONX_AI_URL");
  if (!authType || (!credentials && !bearerToken && !apiKey)) {
    const chosenAuthType = watsonxAIAuthType;
    if (!chosenAuthType)
      throw new Error(
        "No authentication type chosen. Check your enviromental variables or passed options"
      );
    process.env["WATSONX_AI_AUTH_TYPE"] = chosenAuthType;

    if (watsonxAIBearerToken)
      process.env["WATSONX_AI_BEARER_TOKEN"] = watsonxAIBearerToken;
    else if (watsonxAIApikey)
      process.env["WATSONX_AI_APIKEY"] = watsonxAIApikey;
    else if (watsonxAIUsername && watsonxAIPassword && watsonxAIUrl) {
      process.env["WATSONX_AI_USERNAME"] = watsonxAIUsername;
      process.env["WATSONX_AI_PASSWORD"] = watsonxAIPassword;
      process.env["WATSONX_AI_URL"] = watsonxAIUrl;
    } else
      throw new Error(
        "You have not provided any form of authentication, please check passed arguments"
      );
  }
  return WatsonXAI.newInstance({
    version,
    serviceUrl,
  });
};
