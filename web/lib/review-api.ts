import type {
  ReviewMutationResponse,
  ReviewSnapshot,
  RubricAction,
  ScoreOverride
} from "@/lib/review-types";

export type ReviewApi = {
  getSnapshot: () => Promise<ReviewSnapshot>;
  mutateRubric: (action: RubricAction, payload: Record<string, unknown>) => Promise<ReviewSnapshot>;
  overrideScore: (payload: ScoreOverride) => Promise<ReviewMutationResponse>;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/review-api${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers
    }
  });
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const detail =
      typeof body === "object" && body !== null && "detail" in body && typeof body.detail === "string"
        ? body.detail
        : `Local review request failed with ${response.status}.`;
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const localReviewApi: ReviewApi = {
  getSnapshot: () => request<ReviewSnapshot>("/api/review"),
  mutateRubric: (action, payload) =>
    request<ReviewSnapshot>(`/api/review/rubric/${action}`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  overrideScore: (payload) =>
    request<ReviewMutationResponse>("/api/review/score", {
      method: "POST",
      body: JSON.stringify(payload)
    })
};
