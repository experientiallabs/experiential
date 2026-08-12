import { afterEach, describe, expect, it, vi } from "vitest";

import { localReviewApi } from "@/lib/review-api";

describe("localReviewApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses the loopback proxy routes and surfaces the adapter error detail", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify({ project_id: "support" }), { status: 200 })
    );
    vi.stubGlobal("fetch", fetchMock);

    await localReviewApi.getSnapshot();

    expect(fetchMock).toHaveBeenCalledWith(
      "/review-api/api/review",
      expect.objectContaining({ headers: { "Content-Type": "application/json" } })
    );
  });

  it("returns the local validation reason instead of replacing it in the UI", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "finalization needs confirmation" }), { status: 400 })
      )
    );

    await expect(localReviewApi.mutateRubric("finalize", { confirmed: false })).rejects.toThrow(
      "finalization needs confirmation"
    );
  });
});
