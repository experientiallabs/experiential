import { ArrowDown, ArrowUp, CheckCircle2, CircleAlert, Layers3 } from "lucide-react";

import { Button, Card, Chip, EmptyState } from "@/components/ui";
import type {
  HumanScore,
  ReviewRollout,
  RubricDimension,
  Score,
  SelectionCoverage,
  Task
} from "@/lib/review-types";

type TaskReviewProps = {
  activeDimensions: RubricDimension[];
  coverage: SelectionCoverage[];
  humanScores: HumanScore[];
  onScore: (input: {
    rolloutId: string;
    lineageId: string;
    dimensionId: string;
    score: Score;
  }) => void;
  onSelectTask: (taskId: string) => void;
  rollouts: ReviewRollout[];
  selectedTaskId: string;
  tasks: Task[];
};

export function TaskReview({
  activeDimensions,
  coverage,
  humanScores,
  onScore,
  onSelectTask,
  rollouts,
  selectedTaskId,
  tasks
}: TaskReviewProps) {
  const selectedTask = tasks.find((task) => task.task_id === selectedTaskId) ?? tasks[0];
  const selectedCoverage = coverage.find((item) => item.task_id === selectedTask?.task_id);
  const taskRollouts = rollouts.filter((item) => item.rollout.task_id === selectedTask?.task_id);

  if (!selectedTask) {
    return <EmptyState title="No selected tasks" body="Run wmo build for this project, then reopen local review." />;
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[290px_minmax(0,1fr)]">
      <Card className="h-fit p-2">
        <div className="flex items-center justify-between px-2 py-2">
          <span className="text-xs font-semibold uppercase tracking-[0.1em] text-muted">Task set</span>
          <span className="font-mono text-xs text-muted">{tasks.length}</span>
        </div>
        <div aria-label="Task list" className="space-y-1" role="listbox">
          {tasks.map((task, index) => {
            const selected = task.task_id === selectedTask.task_id;
            return (
              <button
                aria-selected={selected}
                className={`w-full rounded-[var(--radius-md)] px-3 py-3 text-left transition-colors ${
                  selected ? "bg-active text-ink" : "text-[#474747] hover:bg-hover"
                }`}
                key={task.task_id}
                onClick={() => onSelectTask(task.task_id)}
                role="option"
                type="button"
              >
                <span className="mb-2 flex items-center justify-between gap-3">
                  <span className="font-mono text-[11px] text-muted">{String(index + 1).padStart(2, "0")}</span>
                  <Chip label={task.partition === "held_out" ? "held out" : "fit"} tone="neutral" />
                </span>
                <span className="line-clamp-3 block text-sm font-medium leading-snug">{task.instruction}</span>
              </button>
            );
          })}
        </div>
        <p className="mb-1 mt-3 px-2 text-xs leading-relaxed text-muted">
          <ArrowUp aria-hidden="true" className="mr-1 inline size-3" />
          <ArrowDown aria-hidden="true" className="mr-1 inline size-3" />
          Use arrow keys outside a field to move between tasks.
        </p>
      </Card>

      <div className="space-y-4">
        <TaskContext coverage={selectedCoverage} task={selectedTask} />
        {taskRollouts.length === 0 ? (
          <EmptyState
            body="This selected task has no stored rollout evidence yet. The task and coverage provenance remain reviewable."
            title="No rollout evidence for this task"
          />
        ) : (
          taskRollouts.map((item) => (
            <RolloutEvidence
              activeDimensions={activeDimensions}
              humanScores={humanScores}
              key={item.rollout.rollout_id}
              onScore={onScore}
              reviewRollout={item}
            />
          ))
        )}
      </div>
    </div>
  );
}

function TaskContext({ coverage, task }: { coverage?: SelectionCoverage; task: Task }) {
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="m-0 font-mono text-[11px] uppercase tracking-[0.12em] text-muted">Selected task</p>
          <h2 className="mb-0 mt-2 max-w-3xl text-xl font-semibold tracking-tight text-ink">
            {task.instruction}
          </h2>
        </div>
        <Chip label={task.partition === "held_out" ? "held-out evidence" : "fit evidence"} tone="purple" />
      </div>
      <dl className="mt-5 grid gap-4 border-t border-line pt-4 sm:grid-cols-3">
        <Definition label="Lineage" value={task.lineage_group_id} />
        <Definition label="Workload weight" value={`${Math.round(task.workload_weight * 100)}%`} />
        <Definition label="Source traces" value={task.source_trace_ids.join(", ")} />
      </dl>
      {coverage ? (
        <div className="mt-4 rounded-[var(--radius-md)] bg-surface-subtle p-3">
          <p className="m-0 flex items-center gap-2 text-sm font-medium text-[#363636]">
            <Layers3 aria-hidden="true" className="size-4 text-muted" />
            Coverage selection
          </p>
          <p className="mb-0 mt-1 text-sm leading-relaxed text-muted">
            {coverage.selection_reasons.join(" · ")} · representative trace {coverage.representative_trace_id}
          </p>
        </div>
      ) : null}
    </Card>
  );
}

function RolloutEvidence({
  activeDimensions,
  humanScores,
  onScore,
  reviewRollout
}: {
  activeDimensions: RubricDimension[];
  humanScores: HumanScore[];
  onScore: TaskReviewProps["onScore"];
  reviewRollout: ReviewRollout;
}) {
  const { judgment, lineage_id: lineageId, rollout } = reviewRollout;
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="m-0 font-mono text-[11px] uppercase tracking-[0.12em] text-muted">Rollout evidence</p>
          <h2 className="mb-0 mt-2 text-base font-semibold text-ink">{rollout.candidate.model_id}</h2>
          <p className="mb-0 mt-1 text-sm text-muted">
            {rollout.evidence_source.replaceAll("_", " ")} · stopped {rollout.stop_reason.replaceAll("_", " ")}
          </p>
        </div>
        {rollout.failure ? (
          <Chip label="Failure captured" tone="danger" />
        ) : (
          <Chip label="Completed" tone="success" />
        )}
      </div>
      <div className="mt-5 space-y-3 border-t border-line pt-4">
        {rollout.spans.map((span, index) => (
          <article className="grid gap-2 sm:grid-cols-[28px_minmax(0,1fr)]" key={span.span_id}>
            <span className="font-mono text-xs text-muted">{String(index + 1).padStart(2, "0")}</span>
            <div className="rounded-[var(--radius-md)] border border-line bg-[#fcfcfc] p-3">
              <p className="m-0 flex flex-wrap items-center gap-2 text-sm font-semibold text-[#363636]">
                {span.kind.replaceAll("_", " ")}
                {span.tool_name ? <Chip label={span.tool_name} tone="blue" /> : null}
              </p>
              <pre className="mb-0 mt-2 overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-muted">
                {JSON.stringify(span.payload, null, 2)}
              </pre>
              {span.failure ? <p className="mb-0 mt-2 text-sm text-danger">{span.failure.message}</p> : null}
            </div>
          </article>
        ))}
      </div>
      {rollout.final_output ? (
        <div className="mt-4 rounded-[var(--radius-md)] border border-line bg-surface p-3">
          <p className="m-0 text-xs font-semibold uppercase tracking-[0.1em] text-muted">Final output</p>
          <pre className="mb-0 mt-2 overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-[#474747]">
            {JSON.stringify(rollout.final_output, null, 2)}
          </pre>
        </div>
      ) : null}
      <JudgmentReview
        activeDimensions={activeDimensions}
        humanScores={humanScores}
        judgment={judgment}
        lineageId={lineageId}
        onScore={onScore}
        rolloutId={rollout.rollout_id}
      />
    </Card>
  );
}

function JudgmentReview({
  activeDimensions,
  humanScores,
  judgment,
  lineageId,
  onScore,
  rolloutId
}: {
  activeDimensions: RubricDimension[];
  humanScores: HumanScore[];
  judgment: ReviewRollout["judgment"];
  lineageId: string | null;
  onScore: TaskReviewProps["onScore"];
  rolloutId: string;
}) {
  if (!judgment) {
    return (
      <div className="mt-5 rounded-[var(--radius-md)] bg-warning-soft p-3 text-sm text-warning">
        No stored judgment is attached to this rollout yet.
      </div>
    );
  }
  return (
    <div className="mt-5 border-t border-line pt-4">
      <div className="flex items-center gap-2">
        <CheckCircle2 aria-hidden="true" className="size-4 text-success" />
        <h3 className="m-0 text-sm font-semibold text-ink">Judgment review</h3>
      </div>
      <div className="mt-3 space-y-3">
        {judgment.dimensions.map((dimension) => {
          const rubricDimension = activeDimensions.find(
            (item) => item.dimension_id === dimension.dimension_id
          );
          const activeHumanScore = humanScores
            .filter(
              (item) =>
                item.rollout_id === rolloutId &&
                item.dimension_id === dimension.dimension_id &&
                item.lineage_id === lineageId
            )
            .at(-1);
          return (
            <div className="rounded-[var(--radius-md)] bg-surface-subtle p-3" key={dimension.dimension_id}>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="m-0 text-sm font-semibold text-[#363636]">
                    {rubricDimension?.name ?? dimension.dimension_id}
                  </p>
                  <p className="mb-0 mt-1 text-sm leading-relaxed text-muted">{dimension.feedback}</p>
                </div>
                <span className="font-mono text-xs text-muted">
                  raw {dimension.raw_score} · calibrated {dimension.calibrated_score.toFixed(1)}
                </span>
              </div>
              {activeDimensions.some((item) => item.dimension_id === dimension.dimension_id) && lineageId ? (
                <div className="mt-3 flex flex-wrap items-center gap-2" aria-label={`Set ${rubricDimension?.name ?? dimension.dimension_id} score`}>
                  <span className="mr-1 text-xs text-muted">
                    Human {activeHumanScore ? `override: ${activeHumanScore.score}` : "score"}
                  </span>
                  {([0, 1, 2, 3, 4, 5] as Score[]).map((score) => (
                    <Button
                      aria-label={`Set ${rubricDimension?.name ?? dimension.dimension_id} score to ${score}`}
                      className="min-h-8 min-w-8 px-2 font-mono text-xs"
                      key={score}
                      onClick={() =>
                        onScore({
                          rolloutId,
                          lineageId,
                          dimensionId: dimension.dimension_id,
                          score
                        })
                      }
                      type="button"
                      variant={activeHumanScore?.score === score ? "primary" : "default"}
                    >
                      {score}
                    </Button>
                  ))}
                </div>
              ) : (
                <p className="mb-0 mt-3 flex items-center gap-2 text-xs text-muted">
                  <CircleAlert aria-hidden="true" className="size-3" />
                  Finalize this rubric before recording a human score.
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Definition({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="font-mono text-[11px] uppercase tracking-[0.1em] text-muted">{label}</dt>
      <dd className="mb-0 mt-1 break-words text-sm text-[#474747]">{value}</dd>
    </div>
  );
}
