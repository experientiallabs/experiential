import { describe, expect, it } from "vitest";
import { errorCode, MODEL_UNAVAILABLE_MESSAGE, modelErrorEvent } from "./errors.js";

describe("error codes", () => {
  it("maps gateway refusals to codes", () => {
    expect(errorCode("403 model not granted")).toBe("model_unavailable");
    expect(errorCode("404 The model does not exist")).toBe("model_unavailable");
    expect(errorCode("402 Payment required")).toBe("insufficient_credits");
    expect(errorCode("Insufficient credits")).toBe("insufficient_credits");
    expect(errorCode("429 slow down")).toBe("rate_limited");
    expect(errorCode("500 boom")).toBeUndefined();
    expect(errorCode(undefined)).toBeUndefined();
  });

  it("replaces a refused model's message with plain words", () => {
    expect(modelErrorEvent("403 model alias not granted")).toEqual({ message: MODEL_UNAVAILABLE_MESSAGE, code: "model_unavailable" });
    expect(modelErrorEvent("500 boom")).toEqual({ message: "500 boom" });
  });
});
