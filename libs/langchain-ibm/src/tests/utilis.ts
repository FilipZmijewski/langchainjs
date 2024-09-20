import { WatsonXEmbeddings, WatsonXEmbeddingsParams } from "../embeddings.js";
import { WatsonX, WatsonXInputLLM } from "../llms.js";

export function getKey<K>(key: K): K {
  return key;
}
export const testProperties = (
  instance: WatsonX | WatsonXEmbeddings,
  testProps: WatsonXInputLLM,
  notExTestProps?: { [key: string]: any }
) => {
  const checkProperty = <T extends { [key: string]: any }>(
    testProps: T,
    instance: T,
    existing = true
  ) => {
    Object.keys(testProps).forEach((key) => {
      const keys = getKey<keyof T>(key);
      type Type = Pick<T, typeof keys>;
      typeof testProps[key as keyof T] === "object"
        ? checkProperty<Type>(
            testProps[key as keyof T],
            instance[key],
            existing
          )
        : existing
        ? expect(instance[key as keyof T]).toBe(testProps[key as keyof T])
        : instance && expect(instance[key as keyof T]).toBeUndefined();
    });
  };
  checkProperty<WatsonXEmbeddingsParams>(testProps, instance);
  notExTestProps &&
    checkProperty<typeof notExTestProps>(notExTestProps, instance, false);
};
