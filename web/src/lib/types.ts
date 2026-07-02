/**
 * Domain types, mirrored by hand from the Python models (wmh/config/card.py ModelCard,
 * wmh/core/types.py Action/Observation/Session, wmh/serving BuildSnapshot/RunRecord).
 * Keep in sync when the serving API changes — every page renders through these.
 */

export type CardCorpus = {
  traces: number | null;
  steps: number;
  source?: string | null;
};

export type CardFidelity = {
  suite: string;
  score: number;
  std?: number | null;
  run_id?: string | null;
};

export type ModelCard = {
  schema_version: number;
  name: string;
  title: string;
  description: string;
  task?: string | null;
  corpus: CardCorpus;
  provider: string;
  model_id: string;
  fidelity?: CardFidelity | null;
  cost_per_step_usd?: number | null;
  latency_per_step_s?: number | null;
  built_at?: string | null;
  license?: string | null;
  tags: string[];
};

/** A sample (action -> observation) from the model's replay index — the card's "screenshot". */
export type PreviewStep = {
  action: string;
  observation: string;
};

/** One gallery entry in the generated index: the card plus build metrics from metrics.json. */
export type IndexEntry = {
  card: ModelCard;
  dir: string;
  held_out_accuracy: number | null;
  serve_root: string;
  preview: PreviewStep[];
};

export type SiteIndex = {
  generated_at: string;
  models: IndexEntry[];
};

// ---- live serving API ----

export type Action =
  | { kind: "tool_call"; name: string; arguments: Record<string, unknown> }
  | { kind: "message"; content: string };

export type Observation = {
  content: string;
  is_error: boolean;
  reward?: number | null;
  metadata: Record<string, unknown>;
};

export type EnvState = {
  structured: Record<string, unknown>;
  scratchpad: string;
};

export type SessionStep = {
  action: Action;
  observation: Observation;
};

export type Session = {
  id: string;
  task?: string | null;
  state: EnvState;
  history: SessionStep[];
};

export type UsageTotals = {
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
};

export type RunRecord = {
  run_id: string;
  kind: string;
  duration_seconds: number;
  total: UsageTotals;
};

export type ModelsResponse = {
  world_models: string[];
  models: { name: string; card: ModelCard | null }[];
};

export type BuildEvent = {
  type: string;
  traces?: number;
  steps?: number;
  train?: number;
  test?: number;
  budget?: number;
  done?: number;
  score?: number | null;
  held_out_accuracy?: number;
  frontier_size?: number;
  rollouts?: number;
  error?: string;
  name?: string;
};

export type BuildSnapshot = {
  build_id: string;
  name: string;
  status: "running" | "succeeded" | "failed";
  error?: string | null;
  events: BuildEvent[];
};
