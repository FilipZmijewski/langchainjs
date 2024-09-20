import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import { WatsonXAuth, WatsonXInit } from "../types.js";
import {
  // @ts-ignore
  IamAuthenticator,
  // @ts-ignore
  BearerTokenAuthenticator,
  // @ts-ignore
  CloudPakForDataAuthenticator,
} from "ibm-cloud-sdk-core";

export const authenticateAndSetInstance = ({
  watsonxAIApikey,
  watsonxAIAuthType,
  watsonxAIBearerToken,
  watsonxAIUsername,
  watsonxAIPassword,
  watsonxAIUrl,
  version,
  serviceUrl,
}: WatsonXAuth & Omit<WatsonXInit, "authenticator">): WatsonXAI => {
  if (watsonxAIAuthType === "iam" && watsonxAIApikey) {
    return WatsonXAI.newInstance({
      version,
      serviceUrl,
      authenticator: new IamAuthenticator({
        apikey: watsonxAIApikey,
        url: watsonxAIUrl,
      }),
    });
  } else if (watsonxAIAuthType === "bearertoken" && watsonxAIBearerToken) {
    return WatsonXAI.newInstance({
      version,
      serviceUrl,
      authenticator: new BearerTokenAuthenticator({
        bearerToken: watsonxAIBearerToken,
        url: watsonxAIUrl,
      }),
    });
  } else if (
    (watsonxAIAuthType === "cp4d" && watsonxAIUsername && watsonxAIPassword) ||
    watsonxAIApikey ||
    watsonxAIBearerToken
  ) {
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
};
