import {
  WatsonXAI,
  convertUtilityToolToWatsonxTool,
} from "@ibm-cloud/watsonx-ai";
import {
  BaseToolkit,
  StructuredToolInterface,
  Tool,
  ToolInterface,
} from "@langchain/core/tools";
import { authenticateAndSetInstance, jsonSchemaToZod } from "../utils/ibm.js";
import { WatsonxAuth, WatsonxInit } from "../types/ibm.js";

export interface WatsonxToolParams {
  name: string;
  description: string;
  agent_description?: string;
  schema?: Record<string, any>;
  service?: WatsonXAI;
}
export class WatsonxTool extends Tool implements WatsonxToolParams {
  name: string;

  description: string;

  agent_description?: string;

  service?: WatsonXAI;

  constructor(fields: WatsonXAI.TextChatParameterFunction, service: WatsonXAI) {
    super();

    this.name = fields?.name;
    this.description = fields?.description || "";
    this.schema = jsonSchemaToZod(fields?.parameters);
    this.service = service;
  }

  protected async _call(inputObject: Record<string, any>): Promise<string> {
    const { input, ...args } = inputObject;
    const response = await this.service?.runUtilityAgentToolByName({
      toolId: this.name,
      wxUtilityAgentToolsRunRequest: {
        input: input ?? args,
        tool_name: this.name,
        config: args,
      },
    });

    const result = response?.result.output;
    return new Promise((resolve) => {
      resolve(result ?? "Sorry, the tool did not work as expected");
    });
  }
}

export class WatsonxToolkit extends BaseToolkit {
  tools: ToolInterface[];

  service: WatsonXAI;

  constructor(fields: WatsonxAuth & WatsonxInit) {
    super();
    const {
      watsonxAIApikey,
      watsonxAIAuthType,
      watsonxAIBearerToken,
      watsonxAIUsername,
      watsonxAIPassword,
      watsonxAIUrl,
      version,
      serviceUrl,
    } = fields;

    const auth = authenticateAndSetInstance({
      watsonxAIApikey,
      watsonxAIAuthType,
      watsonxAIBearerToken,
      watsonxAIUsername,
      watsonxAIPassword,
      watsonxAIUrl,
      version,
      serviceUrl,
    });
    if (auth) this.service = auth;
  }

  protected async loadTools() {
    const { result: tools } = await this.service.listUtilityAgentTools();

    this.tools = tools.resources
      .map((tool) => {
        const { function: watsonxTool } = convertUtilityToolToWatsonxTool(tool);
        if (watsonxTool) return new WatsonxTool(watsonxTool, this.service);
        else return undefined;
      })
      .filter((item) => !!item);
  }

  static async init(props: WatsonxAuth & WatsonxInit) {
    const instance = new WatsonxToolkit({ ...props });
    await instance.loadTools();
    return instance;
  }

  getTools(): StructuredToolInterface[] {
    return this.tools;
  }

  getTool(toolName: string): StructuredToolInterface | undefined {
    return this.tools.find((item) => item.name === toolName);
  }
}
