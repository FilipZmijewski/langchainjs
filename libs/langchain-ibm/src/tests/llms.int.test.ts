import { WatsonXLLM, WatsonXInputLLM } from "../llms.js";
import { CallbackManager } from "@langchain/core/callbacks/manager";
import { LLMResult } from "@langchain/core/outputs";
import { TokenUsage } from "../types.js";
import { StringPromptValue } from "@langchain/core/prompt_values";
const originalBackground = process.env.LANGCHAIN_CALLBACKS_BACKGROUND;

describe("Text generation", () => {
  describe("Test invoke method", () => {
    test("Correct value", async () => {
      const watsonXInstance = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      });
      await watsonXInstance.invoke("Hello world?");
    });

    test("Invalid projectId", async () => {
      const watsonXInstance = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: "Test wrong value",
      });
      expect(watsonXInstance.invoke("Hello world?")).rejects.toThrow();
    });

    test("Invalid credentials", async () => {
      const watsonXInstance = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: "Test wrong value",
        watsonxAIAuthType: "iam",
        watsonxAIApikey: "WrongApiKey",
        watsonxAIUrl: "https://wrong.wrong/",
      });
      expect(watsonXInstance.invoke("Hello world?")).rejects.toThrow();
    });

    test("Wrong value", async () => {
      const watsonXInstance = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      });
      // @ts-expect-error
      await watsonXInstance.invoke({});
    });

    test("Stop", async () => {
      const watsonXInstance = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      });
      await watsonXInstance.invoke("Hello, how are you?", {
        stop: ["Hello"],
      });
    }, 5000);

    test("Stop with timeout", async () => {
      const watsonXInstance = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: "sdadasdas" as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 5,
        maxRetries: 3,
      });

      await expect(() =>
        watsonXInstance.invoke("Print hello world", { timeout: 10 })
      ).rejects.toThrowError("AbortError");
    }, 5000);

    test("Signal in call options", async () => {
      const watsonXInstance = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 5,
        maxRetries: 3,
      });
      const controllerNoAbortion = new AbortController();
      expect(
        watsonXInstance.invoke("Print hello world", {
          signal: controllerNoAbortion.signal,
        })
      ).resolves.toBeDefined();
      const controllerToAbort = new AbortController();
      await expect(async () => {
        const ret = watsonXInstance.invoke("Print hello world", {
          signal: controllerToAbort.signal,
        });
        controllerToAbort.abort();
        return ret;
      }).rejects.toThrowError("AbortError");
    }, 5000);

    test("Concurenccy", async () => {
      const model = new WatsonXLLM({
        maxConcurrency: 1,
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      });
      await Promise.all([
        model.invoke("Print hello world"),
        model.invoke("Print hello world"),
      ]);
    });

    test("Token usage", async () => {
      process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "false";
      try {
        let tokenUsage: TokenUsage = {
          generated_token_count: 0,
          input_token_count: 0,
        };
        const model = new WatsonXLLM({
          maxConcurrency: 1,
          version: "2024-05-31",
          max_new_tokens: 1,
          serviceUrl: process.env.API_URL as string,
          projectId: process.env.PROJECT_ID,
          callbacks: CallbackManager.fromHandlers({
            async handleLLMEnd(output: LLMResult) {
              const singleTokenUsage: TokenUsage | undefined =
                output.llmOutput?.tokenUsage;
              if (singleTokenUsage) {
                tokenUsage.generated_token_count +=
                  singleTokenUsage.generated_token_count;
                tokenUsage.input_token_count +=
                  singleTokenUsage.input_token_count;
              }
            },
          }),
        });
        await model.invoke("Hello");
        expect(tokenUsage.generated_token_count).toBe(1);
        expect(tokenUsage.input_token_count).toBe(1);
      } finally {
        process.env.LANGCHAIN_CALLBACKS_BACKGROUND = originalBackground;
      }
    });

    test("Streaming mode", async () => {
      let countedTokens = 0;
      let streamedText = "";
      let usedTokens = 0;
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 5,
        streaming: true,

        callbacks: CallbackManager.fromHandlers({
          async handleLLMNewToken(token: string) {
            countedTokens += 1;
            streamedText += token;
          },
          async handleLLMEnd(output) {
            usedTokens = output.llmOutput?.tokenUsage.generated_token_count;
          },
        }),
      });
      const res = await model.invoke("Print hello world");

      expect(countedTokens).toBe(usedTokens);
      expect(res).toBe(streamedText);
    });
  });

  describe("Test generate methods", () => {
    test("Basic usage", async () => {
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 5,
      });
      const res = await model.generate([
        "Print hello world!",
        "Print hello universe!",
      ]);
      expect(res.generations.length).toBe(2);
    });

    test("Stop", async () => {
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 100,
      });

      const res = await model.generate(
        ["Print hello world!", "Print hello world hello!"],
        {
          stop: ["Hello"],
        }
      );
      for (const g of res.generations) {
        g.forEach((item) => {
          expect(item.text.indexOf("world")).toBe(-1);
        });
      }
    });

    test("Streaming mode with multiple prompts", async () => {
      let nrNewTokens = [0, 0, 0];
      const completions = ["", "", ""];
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 5,
        streaming: true,
        callbacks: CallbackManager.fromHandlers({
          async handleLLMNewToken(token: string, idx) {
            nrNewTokens[idx.prompt] += 1;
            completions[idx.prompt] += token;
          },
        }),
      });
      const res = await model.generate([
        "Print hello world!",
        "Print hello otter!",
        "Print hello butter!",
      ]);
      res.generations.forEach((generation, index) => {
        generation.forEach((g) => {
          expect(g.generationInfo?.generated_token_count).toBe(
            nrNewTokens[index]
          );
        });
      });
      nrNewTokens.forEach((tokens) => expect(tokens > 0).toBe(true));
      expect(res.generations.length).toBe(3);
    });

    test("Prompt value", async () => {
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 5,
      });
      const res = await model.generatePrompt([
        new StringPromptValue("Print hello world!"),
      ]);
      for (const generation of res.generations) {
        expect(generation.length).toBe(1);
      }
    });
  });

  describe("Test stream method", () => {
    test("Basic usage", async () => {
      let countedTokens = 0;
      let streamedText = "";
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 100,
        callbacks: CallbackManager.fromHandlers({
          async handleLLMNewToken(token: string) {
            countedTokens += 1;
            streamedText += token;
          },
        }),
      });
      const stream = await model.stream("Print hello world.");
      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }
      expect(chunks.length).toBeGreaterThan(1);
      expect(chunks.join("")).toBe(streamedText);
    });

    test("Stop", async () => {
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 100,
      });

      const stream = await model.stream("Print hello world!", {
        stop: ["Hello"],
      });
      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }
      expect(chunks.join("").indexOf("world")).toBe(-1);
    });

    test("Timeout", async () => {
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 1000,
      });
      await expect(async () => {
        const stream = await model.stream(
          "How is your day going? Be precise and tell me a lot about it/",
          {
            signal: AbortSignal.timeout(750),
          }
        );
        const chunks = [];
        for await (const chunk of stream) {
          chunks.push(chunk);
        }
      }).rejects.toThrowError();
    });

    test("Signal in call options", async () => {
      const model = new WatsonXLLM({
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        max_new_tokens: 1000,
      });
      const controller = new AbortController();
      await expect(async () => {
        const stream = await model.stream(
          "How is your day going? Be precise and tell me a lot about it",
          {
            signal: controller.signal,
          }
        );
        const chunks = [];
        let i = 0;
        for await (const chunk of stream) {
          i++;
          chunks.push(chunk);
          if ((i = 5)) {
            controller.abort();
          }
        }
      }).rejects.toThrowError();
    });
  });

  describe("Test getNumToken method", () => {
    test("Passing correct value", async () => {
      const testProps: WatsonXInputLLM = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
      };
      const instance = new WatsonXLLM({
        ...testProps,
      });
      expect(instance.getNumTokens("Hello")).resolves.toBeGreaterThanOrEqual(0);
      expect(
        instance.getNumTokens("Hello", { return_tokens: true })
      ).resolves.toBeGreaterThanOrEqual(0);
    });

    test("Passing wrong value", async () => {
      const testProps: WatsonXInputLLM = {
        version: "2024-05-31",
        serviceUrl: process.env.API_URL as string,
        projectId: process.env.PROJECT_ID,
        maxRetries: 3,
      };
      const instance = new WatsonXLLM({
        ...testProps,
      });

      // @ts-expect-error
      expect(instance.getNumTokens(12)).rejects.toThrowError();
      expect(
        // @ts-expect-error
        instance.getNumTokens(12, { wrong: "Wrong" })
      ).rejects.toThrowError();
    });
  });
});
