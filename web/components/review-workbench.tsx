"use client";

import { ChevronRight, Database, Gauge, ListChecks, RefreshCw, Scale } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { CalibrationReview } from "@/components/calibration-review";
import { RubricReview } from "@/components/rubric-review";
import { TaskReview } from "@/components/task-review";
import { Button, Card, Chip, EmptyState } from "@/components/ui";
import { localReviewApi, type ReviewApi } from "@/lib/review-api";
import type {
  CalibrationApproval,
  ReviewSnapshot,
  RubricAction,
  ScoreOverride
} from "@/lib/review-types";

type ReviewTab = "tasks" | "rubric" | "calibration";

const tabs: Array<{ id: ReviewTab; label: string; icon: typeof ListChecks }> = [
  { id: "tasks", label: "Task review", icon: ListChecks },
  { id: "rubric", label: "Rubric scales", icon: Scale },
  { id: "calibration", label: "Calibration", icon: Gauge }
];

export function ReviewWorkbench({ api = localReviewApi }: { api?: ReviewApi }) {
  const [snapshot, setSnapshot] = useState<ReviewSnapshot | null>(null);
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [tab, setTab] = useState<ReviewTab>("tasks");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const next = await api.getSnapshot();
      setSnapshot(next);
      setSelectedTaskId((current) => current || next.task_set.tasks[0]?.task_id || "");
    } catch (reason) {
      setError(messageFor(reason));
    } finally {
      setBusy(false);
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (tab !== "tasks" || !snapshot) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName)
      ) {
        return;
      }
      const currentIndex = snapshot.task_set.tasks.findIndex((task) => task.task_id === selectedTaskId);
      if (event.key === "ArrowDown" && currentIndex < snapshot.task_set.tasks.length - 1) {
        event.preventDefault();
        setSelectedTaskId(snapshot.task_set.tasks[currentIndex + 1].task_id);
      }
      if (event.key === "ArrowUp" && currentIndex > 0) {
        event.preventDefault();
        setSelectedTaskId(snapshot.task_set.tasks[currentIndex - 1].task_id);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedTaskId, snapshot, tab]);

  const activeDimensions = useMemo(
    () => snapshot?.rubric_review.finalized_rubric?.dimensions ?? snapshot?.rubric_review.dimensions ?? [],
    [snapshot]
  );

  const mutateRubric = (action: RubricAction, payload: Record<string, unknown>) => {
    void perform(async () => {
      const next = await api.mutateRubric(action, payload);
      setSnapshot(next);
      setNotice(action === "finalize" ? "Rubric finalized locally." : "Rubric draft saved locally.");
    });
  };

  const overrideScore = (input: ScoreOverride) => {
    void perform(async () => {
      const response = await api.overrideScore(input);
      setSnapshot(response.snapshot);
      setNotice(response.notice);
    });
  };

  const approveCalibration = (reportId: string, input: CalibrationApproval) => {
    void perform(async () => {
      const response = await api.approveCalibration(reportId, input);
      setSnapshot(response.snapshot);
      setNotice(response.notice);
    });
  };

  async function perform(operation: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await operation();
    } catch (reason) {
      setError(messageFor(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-[1440px] px-4 py-5 sm:px-6 lg:px-8">
      <Header busy={busy} notice={notice} onRefresh={() => void refresh()} snapshot={snapshot} />
      <nav aria-label="Local review sections" className="mt-5 flex overflow-x-auto border-b border-line">
        {tabs.map((item) => {
          const Icon = item.icon;
          const selected = tab === item.id;
          return (
            <button
              aria-current={selected ? "page" : undefined}
              className={`inline-flex shrink-0 items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-colors ${
                selected
                  ? "border-ink text-ink"
                  : "border-transparent text-muted hover:border-line-strong hover:text-ink"
              }`}
              key={item.id}
              onClick={() => setTab(item.id)}
              type="button"
            >
              <Icon aria-hidden="true" className="size-4" />
              {item.label}
            </button>
          );
        })}
      </nav>
      {error ? (
        <div className="mt-4 rounded-[var(--radius-md)] border border-danger bg-danger-soft px-4 py-3 text-sm text-danger" role="alert">
          {error}
        </div>
      ) : null}
      <section className="mt-5" aria-live="polite">
        {!snapshot && busy ? <LoadingState /> : null}
        {!snapshot && !busy ? (
          <EmptyState
            body="Start local review with npm run review after wmo build has created the project artifacts."
            title="Local review could not load"
          />
        ) : null}
        {snapshot && tab === "tasks" ? (
          <TaskReview
            activeDimensions={activeDimensions}
            coverage={snapshot.coverage?.selections ?? []}
            humanScores={snapshot.human_score_history.scores}
            onScore={({ dimensionId, lineageId, rolloutId, score }) =>
              overrideScore({
                dimension_id: dimensionId,
                lineage_id: lineageId,
                rollout_id: rolloutId,
                score
              })
            }
            onSelectTask={setSelectedTaskId}
            rollouts={snapshot.rollouts}
            selectedTaskId={selectedTaskId}
            tasks={snapshot.task_set.tasks}
          />
        ) : null}
        {snapshot && tab === "rubric" ? (
          <RubricReview busy={busy} onMutation={mutateRubric} snapshot={snapshot} />
        ) : null}
        {snapshot && tab === "calibration" ? (
          <CalibrationReview
            busy={busy}
            calibrations={snapshot.calibrations}
            dimensions={activeDimensions}
            onApprove={approveCalibration}
            reports={snapshot.calibration_reports}
            scores={snapshot.human_score_history.scores}
          />
        ) : null}
      </section>
    </main>
  );
}

function Header({
  busy,
  notice,
  onRefresh,
  snapshot
}: {
  busy: boolean;
  notice: string;
  onRefresh: () => void;
  snapshot: ReviewSnapshot | null;
}) {
  return (
    <Card className="bg-surface p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-5">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <Chip label="Local only" tone="purple" />
            <span className="font-mono text-xs text-muted">{snapshot?.project_id ?? "loading project"}</span>
          </div>
          <h1 className="mb-0 mt-3 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Judgment review</h1>
          <p className="mb-0 mt-2 max-w-3xl text-sm leading-relaxed text-muted">
            Inspect selected tasks, rollout evidence, rubric scales, and judge disagreement without leaving
            this machine.
          </p>
        </div>
        <Button disabled={busy} onClick={onRefresh} type="button">
          <RefreshCw aria-hidden="true" className={`size-4 ${busy ? "animate-spin" : ""}`} />
          Refresh local data
        </Button>
      </div>
      <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-line pt-4 text-sm text-muted">
        <Database aria-hidden="true" className="size-4" />
        <span>{snapshot?.local_data_notice ?? "Reading only the local project store."}</span>
      </div>
      {notice ? (
        <div className="mt-3 flex items-center gap-2 rounded-[var(--radius-md)] bg-success-soft px-3 py-2 text-sm text-success">
          <ChevronRight aria-hidden="true" className="size-4" />
          {notice}
        </div>
      ) : null}
    </Card>
  );
}

function LoadingState() {
  return (
    <div className="grid min-h-[340px] place-items-center">
      <div className="text-center">
        <div className="mx-auto size-7 animate-spin rounded-full border-2 border-line-strong border-t-ink" />
        <p className="mb-0 mt-3 text-sm text-muted">Reading local review artifacts…</p>
      </div>
    </div>
  );
}

function messageFor(reason: unknown): string {
  return reason instanceof Error ? reason.message : "The local review adapter returned an unknown error.";
}
