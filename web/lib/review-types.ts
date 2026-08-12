export type JsonPrimitive = boolean | number | string | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

export type Score = 0 | 1 | 2 | 3 | 4 | 5;
export type RubricAction =
  | "accept"
  | "reject"
  | "edit"
  | "add"
  | "replace_all"
  | "order"
  | "finalize";

export type ScoreAnchor = {
  score: Score;
  description: string;
};

export type RubricDimension = {
  dimension_id: string;
  name: string;
  description: string;
  anchors: ScoreAnchor[];
};

export type ProposedDimension = {
  dimension: RubricDimension;
  source_rollout_ids: string[];
  evidence_span_ids: string[];
  overlap_with_dimension_ids: string[];
};

export type RubricProposal = {
  proposal_id: string;
  dimensions: ProposedDimension[];
  successful_rollout_ids: string[];
  failed_rollout_ids: string[];
};

export type RubricReview = {
  source_task_set_id: string;
  proposals: RubricProposal[];
  dimensions: RubricDimension[];
  rejected_dimension_ids: string[];
  status: "draft" | "finalized";
  finalized_rubric: {
    rubric_id: string;
    dimensions: RubricDimension[];
    status: "provisional" | "human_approved";
  } | null;
};

export type Task = {
  task_id: string;
  lineage_group_id: string;
  partition: "fit" | "held_out";
  instruction: string;
  initial_context: JsonObject;
  tools: Array<{ name: string; description: string; input_schema: JsonObject }>;
  workload_weight: number;
  source_trace_ids: string[];
};

export type SelectionCoverage = {
  task_id: string;
  representative_trace_id: string;
  partition: "fit" | "held_out";
  lineage_group_id: string;
  cluster_id: number;
  selection_reasons: string[];
  source_trace_ids: string[];
  workload_mass: number;
  workload_weight: number;
};

export type CoverageReport = {
  input_trace_count: number;
  invalid_trace_count: number;
  eligible_trace_count: number;
  duplicate_trace_count: number;
  selected_task_count: number;
  split_separation_verified: boolean;
  selections: SelectionCoverage[];
};

export type RolloutSpan = {
  span_id: string;
  kind: string;
  payload: JsonObject;
  tool_name: string | null;
  failure: { message: string } | null;
};

export type Rollout = {
  rollout_id: string;
  task_id: string;
  evidence_source: "production" | "world_model" | "sandbox";
  candidate: { model_id: string };
  stop_reason: string;
  final_output: JsonObject | null;
  failure: { message: string } | null;
  spans: RolloutSpan[];
};

export type DimensionJudgment = {
  dimension_id: string;
  raw_score: Score;
  calibrated_score: number;
  evidence_span_ids: string[];
  feedback: string;
};

export type Judgment = {
  judgment_id: string;
  rollout_id: string;
  rubric_id: string;
  dimensions: DimensionJudgment[];
  overall_score: number;
};

export type ReviewRollout = {
  rollout: Rollout;
  lineage_id: string | null;
  judgment: Judgment | null;
};

export type WorstDisagreement = {
  direction: "optimistic" | "pessimistic" | "exact";
  prediction: {
    rollout_id: string;
    lineage_id: string;
    dimension_id: string;
    raw_score: Score;
    human_score: Score;
    calibrated_score: number;
    absolute_error: number;
  };
};

export type CalibrationReport = {
  report_id: string;
  status: "provisional" | "insufficient" | "ready_for_approval";
  eligible_label_count: number;
  recommended_label_count: number;
  dimension_metrics: Array<{
    dimension_id: string;
    mae: number | null;
    rank_agreement: number | null;
    mean_optimistic_error: number | null;
  }>;
  worst_disagreements: WorstDisagreement[];
};

export type HumanScore = {
  label_id: string;
  rubric_id: string;
  rollout_id: string;
  lineage_id: string;
  dimension_id: string;
  score: Score;
  supersedes_label_id: string | null;
};

export type ReviewSnapshot = {
  project_id: string;
  local_data_notice: string;
  task_set: {
    task_set: { task_set_id: string; code_revision: string };
    tasks: Task[];
  };
  coverage: CoverageReport | null;
  rubric_review: RubricReview;
  human_score_history: { scores: HumanScore[] };
  rollouts: ReviewRollout[];
  calibration_reports: CalibrationReport[];
};

export type ReviewMutationResponse = {
  snapshot: ReviewSnapshot;
  notice: string;
};

export type ScoreOverride = {
  rollout_id: string;
  lineage_id: string;
  dimension_id: string;
  score: Score;
};
