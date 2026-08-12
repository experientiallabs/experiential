import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReviewWorkbench } from "@/components/review-workbench";
import { draftReviewSnapshot, finalizedReviewSnapshot, taskSuccessDimension } from "@/lib/review-fixture";
import type { ReviewApi } from "@/lib/review-api";
import type { ReviewMutationResponse, ReviewSnapshot } from "@/lib/review-types";

describe("ReviewWorkbench", () => {
  afterEach(() => cleanup());

  it("delegates proposal acceptance and confirmed finalization to the local adapter", async () => {
    let state = draftReviewSnapshot;
    const mutateRubric = vi.fn(async (action: string) => {
      if (action === "accept") {
        state = {
          ...state,
          rubric_review: { ...state.rubric_review, dimensions: [taskSuccessDimension] }
        };
      }
      if (action === "finalize") {
        state = finalizedReviewSnapshot;
      }
      return state;
    });
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => state),
      mutateRubric,
      overrideScore: vi.fn()
    };

    render(<ReviewWorkbench api={api} />);

    await screen.findAllByText("Resolve a customer refund request after a duplicate subscription charge.");
    fireEvent.click(screen.getByRole("button", { name: "Rubric scales" }));
    fireEvent.click(screen.getByRole("button", { name: "Accept" }));

    await waitFor(() =>
      expect(mutateRubric).toHaveBeenCalledWith("accept", { dimension_id: "task-success" })
    );
    fireEvent.click(screen.getByRole("button", { name: "Finalize rubric" }));
    expect(screen.getByRole("dialog")).toHaveTextContent("Finalize this rubric?");
    fireEvent.click(screen.getByRole("button", { name: "Confirm finalization" }));

    await waitFor(() => expect(mutateRubric).toHaveBeenCalledWith("finalize", { confirmed: true }));
    expect(await screen.findByText("Finalized scales")).toBeInTheDocument();
  });

  it("records a zero-to-five score override against selected rollout evidence", async () => {
    const overrideScore = vi.fn(async () => scoreResponse(finalizedReviewSnapshot));
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => finalizedReviewSnapshot),
      mutateRubric: vi.fn(),
      overrideScore
    };

    render(<ReviewWorkbench api={api} />);

    await screen.findByText("Rollout evidence");
    fireEvent.click(screen.getByRole("button", { name: "Set Task success score to 4" }));

    await waitFor(() =>
      expect(overrideScore).toHaveBeenCalledWith({
        rollout_id: "rollout-refund",
        lineage_id: "lineage-refund",
        dimension_id: "task-success",
        score: 4
      })
    );
    expect(await screen.findByText("Score saved locally.")).toBeInTheDocument();
  });

  it("moves selected tasks with keyboard arrows outside an input", async () => {
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => draftReviewSnapshot),
      mutateRubric: vi.fn(),
      overrideScore: vi.fn()
    };

    render(<ReviewWorkbench api={api} />);

    await screen.findAllByText("Resolve a customer refund request after a duplicate subscription charge.");
    fireEvent.keyDown(window, { key: "ArrowDown" });

    expect(screen.getByRole("option", { selected: true })).toHaveTextContent(
      "Explain a delayed shipment and give the customer a verified next step."
    );
  });
});

function scoreResponse(snapshot: ReviewSnapshot): ReviewMutationResponse {
  return { notice: "Score saved locally.", snapshot };
}
