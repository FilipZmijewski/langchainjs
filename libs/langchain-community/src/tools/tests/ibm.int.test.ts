/* eslint-disable no-process-env */
import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import { WatsonxTool, WatsonxToolkit } from "../ibm.js";

const serviceUrl = process.env.WATSONX_AI_SERVICE_URL as string;
describe("Tool class tests", () => {
  test("Init tool", async () => {
    const service = WatsonXAI.newInstance({
      serviceUrl,
      version: "2024-05-31",
    });
    const tool = new WatsonxTool(
      {
        name: "Weather",
        description: "Get the weather for a given location",
        parameters: {
          type: "object",
          properties: {
            name: {
              title: "name",
              description: "Name of the location",
              type: "string",
            },
            country: {
              title: "country",
              description: "Name of the state or country",
              type: "string",
            },
          },
          required: ["name"],
        },
      },
      service
    );
    expect(tool).toBeInstanceOf(WatsonxTool);
  });
  test("Invoke tool with json", async () => {
    const service = WatsonXAI.newInstance({
      serviceUrl,
      version: "2024-05-31",
    });
    const tool = new WatsonxTool(
      {
        name: "Weather",
        description: "Get the weather for a given location",
        parameters: {
          type: "object",
          properties: {
            name: {
              title: "name",
              description: "Name of the location",
              type: "string",
            },
            country: {
              title: "country",
              description: "Name of the state or country",
              type: "string",
            },
          },
          required: ["name"],
        },
      },
      service
    );

    const res = await tool.invoke({ name: "Krakow" });
    expect(res).toBeDefined();
  });
  test("Invoke tool with tool_call", async () => {
    const service = WatsonXAI.newInstance({
      serviceUrl,
      version: "2024-05-31",
    });
    const tool = new WatsonxTool(
      {
        name: "Weather",
        description: "Get the weather for a given location",
        parameters: {
          type: "object",
          properties: {
            name: {
              title: "name",
              description: "Name of the location",
              type: "string",
            },
            country: {
              title: "country",
              description: "Name of the state or country",
              type: "string",
            },
          },
          required: ["name"],
        },
      },
      service
    );
    const toolCall = {
      name: "Weather",
      args: {
        name: "Krakow",
      },
      type: "tool_call",
      id: "ABCD12345",
    };
    const res = await tool.invoke(toolCall);
    expect(res).toBeDefined();
  });
});

describe("Toolkit class tests", () => {
  test("Toolkit init", async () => {
    const toolkit = await WatsonxToolkit.init({
      version: "2024-05-31",
      serviceUrl,
    });
    expect(toolkit).toBeInstanceOf(WatsonxToolkit);
  });
  test("Test method getTools", async () => {
    const toolkit = await WatsonxToolkit.init({
      version: "2024-05-31",
      serviceUrl,
    });
    const testTools = toolkit.getTools();
    testTools.map((tool) => expect(tool).toBeInstanceOf(WatsonxTool));
  });

  test("Test method getTool", async () => {
    const toolkit = await WatsonxToolkit.init({
      version: "2024-05-31",
      serviceUrl,
    });
    const tool = toolkit.getTool("Weather");
    expect(tool).toBeInstanceOf(WatsonxTool);
  });
});
