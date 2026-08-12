import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReviewWorkbench } from "@/components/review-workbench";
import { draftReviewSnapshot, finalizedReviewSnapshot, taskSuccessDimension } from "@/lib/review-fixture";
import type { ReviewApi } from "@/lib/review-api";
import type { ReviewMutationResponse, ReviewSnapshot } from "@/lib/review-types";

const clarityDimension = {
  ...taskSuccessDimension,
  dimension_id: "customer-clarity",
  name: "Customer clarity",
  description: "Whether the response gives the customer a clear next step."
};

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
      overrideScore: vi.fn(),
      approveCalibration: vi.fn()
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
      overrideScore,
      approveCalibration: vi.fn()
    };

    render(<ReviewWorkbench api={api} />);

    await screen.findByText("Rollout evidence");
    fireEvent.click(screen.getByRole("button", { name: "Set Task success score to 4" }));

    await waitFor(() =>
      expect(overrideScore).toHaveBeenCalledWith({
        rollout_id: "rollout-refund",
        lineage_id: "lineage-refund",
        dimension_id: "task-success",
        score: 4,
        submission_id: expect.any(String)
      })
    );
    expect(await screen.findByText("Score saved locally.")).toBeInTheDocument();
  });

  it("moves selected tasks with keyboard arrows outside an input", async () => {
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => draftReviewSnapshot),
      mutateRubric: vi.fn(),
      overrideScore: vi.fn(),
      approveCalibration: vi.fn()
    };

    render(<ReviewWorkbench api={api} />);

    await screen.findAllByText("Resolve a customer refund request after a duplicate subscription charge.");
    fireEvent.keyDown(window, { key: "ArrowDown" });

    expect(screen.getByRole("option", { selected: true })).toHaveTextContent(
      "Explain a delayed shipment and give the customer a verified next step."
    );
  });

  it("submits the exact selected complete replacement set", async () => {
    const snapshot: ReviewSnapshot = {
      ...draftReviewSnapshot,
      rubric_review: {
        ...draftReviewSnapshot.rubric_review,
        dimensions: [taskSuccessDimension],
        proposals: [
          {
            ...draftReviewSnapshot.rubric_review.proposals[0],
            dimensions: [
              {
                ...draftReviewSnapshot.rubric_review.proposals[0].dimensions[0],
                dimension: clarityDimension
              }
            ]
          }
        ]
      }
    };
    const mutateRubric = vi.fn(async () => snapshot);
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => snapshot),
      mutateRubric,
      overrideScore: vi.fn(),
      approveCalibration: vi.fn()
    };

    render(<ReviewWorkbench api={api} />);
    await screen.findByText("Rollout evidence");
    fireEvent.click(screen.getByRole("button", { name: "Rubric scales" }));
    fireEvent.click(screen.getByRole("button", { name: "Design replacement set" }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Customer clarity/ }));
    fireEvent.click(screen.getByRole("button", { name: "Replace all scales" }));

    await waitFor(() =>
      expect(mutateRubric).toHaveBeenCalledWith("replace_all", {
        dimensions: [clarityDimension]
      })
    );
  });

  it("blocks background proposal shortcuts while a modal is open", async () => {
    const mutateRubric = vi.fn(async () => draftReviewSnapshot);
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => draftReviewSnapshot),
      mutateRubric,
      overrideScore: vi.fn(),
      approveCalibration: vi.fn()
    };

    render(<ReviewWorkbench api={api} />);
    await screen.findByText("Rollout evidence");
    fireEvent.click(screen.getByRole("button", { name: "Rubric scales" }));
    const proposal = screen.getByRole("article", { name: "Task success proposal" });
    fireEvent.click(screen.getByRole("button", { name: "Design replacement set" }));
    await screen.findByRole("dialog", { name: "Design the complete replacement set" });
    fireEvent.keyDown(proposal, { key: "a" });

    expect(mutateRubric).not.toHaveBeenCalled();
  });

  it("dismisses rubric finalization with Escape without finalizing", async () => {
    const snapshot: ReviewSnapshot = {
      ...draftReviewSnapshot,
      rubric_review: {
        ...draftReviewSnapshot.rubric_review,
        dimensions: [taskSuccessDimension]
      }
    };
    const mutateRubric = vi.fn(async () => snapshot);
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => snapshot),
      mutateRubric,
      overrideScore: vi.fn(),
      approveCalibration: vi.fn()
    };

    render(<ReviewWorkbench api={api} />);
    await screen.findByText("Rollout evidence");
    fireEvent.click(screen.getByRole("button", { name: "Rubric scales" }));
    const finalize = screen.getByRole("button", { name: "Finalize rubric" });
    finalize.focus();
    fireEvent.click(finalize);
    fireEvent.keyDown(screen.getByRole("dialog", { name: "Finalize this rubric?" }), {
      key: "Escape"
    });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(finalize).toHaveFocus();
    expect(mutateRubric).not.toHaveBeenCalled();
  });

  it("disables reorder controls at the first and last boundaries", async () => {
    const snapshot: ReviewSnapshot = {
      ...draftReviewSnapshot,
      rubric_review: {
        ...draftReviewSnapshot.rubric_review,
        dimensions: [taskSuccessDimension, clarityDimension]
      }
    };
    const mutateRubric = vi.fn(async () => snapshot);
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => snapshot),
      mutateRubric,
      overrideScore: vi.fn(),
      approveCalibration: vi.fn()
    };

    render(<ReviewWorkbench api={api} />);
    await screen.findByText("Rollout evidence");
    fireEvent.click(screen.getByRole("button", { name: "Rubric scales" }));

    expect(screen.getByRole("button", { name: "Move Task success up" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Move Task success down" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Move Customer clarity up" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Move Customer clarity down" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Move Task success up" }));
    fireEvent.click(screen.getByRole("button", { name: "Move Customer clarity down" }));
    expect(mutateRubric).not.toHaveBeenCalled();
  });

  it("requires low-sample risk confirmation and renders the written calibration", async () => {
    const insufficientSnapshot: ReviewSnapshot = {
      ...finalizedReviewSnapshot,
      calibration_reports: finalizedReviewSnapshot.calibration_reports.map((report) => ({
        ...report,
        status: "insufficient"
      }))
    };
    const approvedSnapshot: ReviewSnapshot = {
      ...insufficientSnapshot,
      calibrations: [
        {
          calibration_id: "human-calibration-fixture",
          rubric_id: "rubric-fixture",
          out_of_fold_report_id: "calibration-fixture",
          label_count: 2,
          recommended_label_count: 8,
          status: "human_calibrated",
          approved_at: "2026-08-12T00:00:00Z",
          risk_acceptance: { artifact_id: "risk-fixture", sha256: "a".repeat(64) }
        }
      ]
    };
    const approveCalibration = vi.fn(async () => scoreResponse(approvedSnapshot));
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => insufficientSnapshot),
      mutateRubric: vi.fn(),
      overrideScore: vi.fn(),
      approveCalibration
    };

    render(<ReviewWorkbench api={api} />);
    await screen.findByText("Rollout evidence");
    fireEvent.click(screen.getByRole("button", { name: "Calibration" }));
    fireEvent.click(screen.getByRole("button", { name: "Review low-sample approval" }));
    const confirm = screen.getByRole("button", { name: "Confirm calibration approval" });
    expect(confirm).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: /explicitly accept/ }));
    fireEvent.click(confirm);

    await waitFor(() =>
      expect(approveCalibration).toHaveBeenCalledWith("calibration-fixture", {
        confirmed: true,
        accept_insufficient_risk: true
      })
    );
    expect(await screen.findByText("Human-calibrated artifact")).toBeInTheDocument();
    expect(screen.getByText(/risk-fixture/)).toBeInTheDocument();
  });

  it("dismisses calibration approval with Escape without approving", async () => {
    const insufficientSnapshot: ReviewSnapshot = {
      ...finalizedReviewSnapshot,
      calibration_reports: finalizedReviewSnapshot.calibration_reports.map((report) => ({
        ...report,
        status: "insufficient"
      }))
    };
    const approveCalibration = vi.fn();
    const api: ReviewApi = {
      getSnapshot: vi.fn(async () => insufficientSnapshot),
      mutateRubric: vi.fn(),
      overrideScore: vi.fn(),
      approveCalibration
    };

    render(<ReviewWorkbench api={api} />);
    await screen.findByText("Rollout evidence");
    fireEvent.click(screen.getByRole("button", { name: "Calibration" }));
    const reviewApproval = screen.getByRole("button", { name: "Review low-sample approval" });
    reviewApproval.focus();
    fireEvent.click(reviewApproval);
    const dialog = screen.getByRole("dialog", { name: "Approve this judge calibration?" });
    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(reviewApproval).toHaveFocus();
    expect(approveCalibration).not.toHaveBeenCalled();
  });
});

function scoreResponse(snapshot: ReviewSnapshot): ReviewMutationResponse {
  return { notice: "Score saved locally.", snapshot };
}
