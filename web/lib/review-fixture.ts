import type { ReviewSnapshot, RubricDimension, ScoreAnchor } from "@/lib/review-types";

const anchors: ScoreAnchor[] = [
  { score: 0, description: "The requested outcome is absent or harmful." },
  { score: 1, description: "The response misses the core request." },
  { score: 2, description: "The response makes limited, incomplete progress." },
  { score: 3, description: "The response resolves the request with material gaps." },
  { score: 4, description: "The response resolves the request clearly and correctly." },
  { score: 5, description: "The response resolves the request with excellent evidence and care." }
];

export const taskSuccessDimension: RubricDimension = {
  dimension_id: "task-success",
  name: "Task success",
  description: "Whether the customer received the requested outcome.",
  anchors
};

const baseSnapshot: ReviewSnapshot = {
  project_id: "support",
  local_data_notice:
    "This review stays on this machine. WMO reads local artifacts and writes only the project review draft plus immutable approved artifacts. Browser restarts resume review.json; remove the local project directory yourself to clear all project data.",
  task_set: {
    task_set: { task_set_id: "task-set-fixture", code_revision: "fixture" },
    tasks: [
      {
        task_id: "task-refund",
        lineage_group_id: "lineage-refund",
        partition: "fit",
        instruction: "Resolve a customer refund request after a duplicate subscription charge.",
        initial_context: { channel: "email", account_tier: "standard" },
        tools: [
          {
            name: "billing_lookup",
            description: "Find a subscription charge by account and invoice.",
            input_schema: { type: "object", required: ["account_id"] }
          }
        ],
        workload_weight: 0.62,
        source_trace_ids: ["trace-refund-01", "trace-refund-07"]
      },
      {
        task_id: "task-shipping",
        lineage_group_id: "lineage-shipping",
        partition: "held_out",
        instruction: "Explain a delayed shipment and give the customer a verified next step.",
        initial_context: { channel: "chat", locale: "en-US" },
        tools: [
          {
            name: "shipment_lookup",
            description: "Retrieve a shipment status and carrier event history.",
            input_schema: { type: "object", required: ["order_id"] }
          }
        ],
        workload_weight: 0.38,
        source_trace_ids: ["trace-shipping-03"]
      }
    ]
  },
  coverage: {
    input_trace_count: 8,
    invalid_trace_count: 0,
    eligible_trace_count: 8,
    duplicate_trace_count: 1,
    selected_task_count: 2,
    split_separation_verified: true,
    selections: [
      {
        task_id: "task-refund",
        representative_trace_id: "trace-refund-01",
        partition: "fit",
        lineage_group_id: "lineage-refund",
        cluster_id: 0,
        selection_reasons: ["coverage representative", "high workload mass"],
        source_trace_ids: ["trace-refund-01", "trace-refund-07"],
        workload_mass: 0.62,
        workload_weight: 0.62
      },
      {
        task_id: "task-shipping",
        representative_trace_id: "trace-shipping-03",
        partition: "held_out",
        lineage_group_id: "lineage-shipping",
        cluster_id: 1,
        selection_reasons: ["held-out lineage", "coverage representative"],
        source_trace_ids: ["trace-shipping-03"],
        workload_mass: 0.38,
        workload_weight: 0.38
      }
    ]
  },
  rubric_review: {
    source_task_set_id: "task-set-fixture",
    proposals: [
      {
        proposal_id: "proposal-task-success",
        successful_rollout_ids: ["rollout-refund"],
        failed_rollout_ids: ["rollout-shipping"],
        dimensions: [
          {
            dimension: taskSuccessDimension,
            source_rollout_ids: ["rollout-refund", "rollout-shipping"],
            evidence_span_ids: ["span-refund-answer", "span-shipping-answer"],
            overlap_with_dimension_ids: []
          }
        ]
      }
    ],
    dimensions: [],
    rejected_dimension_ids: [],
    status: "draft",
    finalized_rubric: null
  },
  human_score_history: { scores: [] },
  rollouts: [
    {
      lineage_id: "lineage-refund",
      rollout: {
        rollout_id: "rollout-refund",
        task_id: "task-refund",
        evidence_source: "world_model",
        candidate: { model_id: "candidate-a" },
        stop_reason: "completed",
        final_output: { message: "The duplicate charge was refunded and confirmation was sent." },
        failure: null,
        spans: [
          {
            span_id: "span-refund-lookup",
            kind: "tool_call",
            payload: { account_id: "acct_104", invoice: "inv_900" },
            tool_name: "billing_lookup",
            failure: null
          },
          {
            span_id: "span-refund-answer",
            kind: "agent_message",
            payload: { text: "I found the duplicate charge and submitted the refund." },
            tool_name: null,
            failure: null
          }
        ]
      },
      judgment: {
        judgment_id: "judgment-refund",
        rollout_id: "rollout-refund",
        rubric_id: "rubric-fixture",
        dimensions: [
          {
            dimension_id: "task-success",
            raw_score: 3,
            calibrated_score: 3.2,
            evidence_span_ids: ["span-refund-answer"],
            feedback: "The refund is completed but the confirmation timing is not explicit."
          }
        ],
        overall_score: 0.64
      }
    },
    {
      lineage_id: "lineage-shipping",
      rollout: {
        rollout_id: "rollout-shipping",
        task_id: "task-shipping",
        evidence_source: "world_model",
        candidate: { model_id: "candidate-a" },
        stop_reason: "completed",
        final_output: { message: "The package is delayed. Please wait." },
        failure: null,
        spans: [
          {
            span_id: "span-shipping-lookup",
            kind: "tool_call",
            payload: { order_id: "order_233" },
            tool_name: "shipment_lookup",
            failure: null
          },
          {
            span_id: "span-shipping-answer",
            kind: "agent_message",
            payload: { text: "The package is delayed. Please wait." },
            tool_name: null,
            failure: null
          }
        ]
      },
      judgment: {
        judgment_id: "judgment-shipping",
        rollout_id: "rollout-shipping",
        rubric_id: "rubric-fixture",
        dimensions: [
          {
            dimension_id: "task-success",
            raw_score: 1,
            calibrated_score: 1.1,
            evidence_span_ids: ["span-shipping-answer"],
            feedback: "The response gives no verified carrier event or usable next step."
          }
        ],
        overall_score: 0.22
      }
    }
  ],
  calibration_reports: [],
  calibrations: []
};

export const draftReviewSnapshot: ReviewSnapshot = baseSnapshot;

export const finalizedReviewSnapshot: ReviewSnapshot = {
  ...baseSnapshot,
  rubric_review: {
    ...baseSnapshot.rubric_review,
    dimensions: [taskSuccessDimension],
    status: "finalized",
    finalized_rubric: {
      rubric_id: "rubric-fixture",
      dimensions: [taskSuccessDimension],
      status: "human_approved"
    }
  },
  calibration_reports: [
    {
      report_id: "calibration-fixture",
      status: "provisional",
      eligible_label_count: 2,
      recommended_label_count: 8,
      dimension_metrics: [
        {
          dimension_id: "task-success",
          mae: 0.8,
          rank_agreement: 0.5,
          mean_optimistic_error: 0.6
        }
      ],
      worst_disagreements: [
        {
          direction: "optimistic",
          prediction: {
            rollout_id: "rollout-shipping",
            lineage_id: "lineage-shipping",
            dimension_id: "task-success",
            raw_score: 3,
            human_score: 1,
            calibrated_score: 2.5,
            absolute_error: 2
          }
        }
      ]
    }
  ],
  calibrations: []
};
