import { ChatWatsonx } from "../ibm.js";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
const parser = new StringOutputParser();
describe("Tests for chat", () => {
  //   test("Check if working", async () => {
  //     const service = new ChatWatsonx({
  //       version: "2024-05-31",
  //       serviceUrl: process.env.WATSONX_AI_SERVICE_URL as string,
  //       projectId: process.env.WATSONX_AI_PROJECT_ID,
  //     });

  //     const messages = [
  //       new SystemMessage("Translate the following from English into Italian"),
  //       new HumanMessage("hi!"),
  //     ];
  //     const chain = service.pipe(parser);
  //     const result = await chain.invoke(messages);
  //   });

  //   test("Propmt", async () => {
  //     const model = new ChatWatsonx({
  //       version: "2024-05-31",
  //       serviceUrl: process.env.WATSONX_AI_SERVICE_URL as string,
  //       projectId: process.env.WATSONX_AI_PROJECT_ID,
  //     });
  //     const systemTemplate = "Translate the following into {language}:";
  //     const promptTemplate = ChatPromptTemplate.fromMessages([
  //       ["system", systemTemplate],
  //       ["user", "{text}"],
  //     ]);
  //     const promptValue = await promptTemplate.invoke({
  //       language: "italian",
  //       text: "hi",
  //     });
  //     promptValue.toChatMessages();

  //     const llmChain = promptTemplate.pipe(model).pipe(parser);
  //     const result = await llmChain.invoke({ language: "italian", text: "hi" });
  //     console.log(result);
  //   });

  // test("Propmt", async () => {
  //   const model = new ChatWatsonx({
  //     version: "2024-05-31",
  //     serviceUrl: process.env.WATSONX_AI_SERVICE_URL as string,
  //     projectId: process.env.WATSONX_AI_PROJECT_ID,
  //   });

  //   const messages = [
  //     new SystemMessage("Translate the following from English into Italian"),
  //     new HumanMessage("hi!"),
  //   ];

  //   const stream = await model.stream(messages, { maxTokens: 5 });

  //   for await (const chunk of stream) {
  //     // console.log(chunk);
  //   }
  // });

  test("Tools", async () => {
    /**
     * Note that the descriptions here are crucial, as they will be passed along
     * to the model along with the class name.
     */
    const model = new ChatWatsonx({
      version: "2024-05-31",
      serviceUrl: process.env.WATSONX_AI_SERVICE_URL as string,
      projectId: process.env.WATSONX_AI_PROJECT_ID,
    });

    const calculatorSchema = z.object({
      operation: z
        .enum(["add", "subtract", "multiply", "divide"])
        .describe("The type of operation to execute."),
      number1: z.number().describe("The first number to operate on."),
      number2: z.number().describe("The second number to operate on."),
    });

    const calculatorTool = tool(
      async ({ operation, number1, number2 }) => {
        // Functions must return strings
        if (operation === "add") {
          return `${number1 + number2}`;
        } else if (operation === "subtract") {
          return `${number1 - number2}`;
        } else if (operation === "multiply") {
          return `${number1 * number2}`;
        } else if (operation === "divide") {
          return `${number1 / number2}`;
        } else {
          throw new Error("Invalid operation.");
        }
      },
      {
        name: "calculator",
        description: "Can perform mathematical operations.",
        schema: calculatorSchema,
      }
    );

    const llmWithTools = model.bindTools([calculatorTool]);

    const res = await llmWithTools.invoke("What is 3 * 12");

    console.log(res);
  });
});
