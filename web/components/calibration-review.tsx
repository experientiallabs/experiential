import { AlertTriangle, BarChart3, CheckCircle2, History } from "lucide-react";

import { Card, Chip, EmptyState } from "@/components/ui";
import type { CalibrationReport, HumanScore, RubricDimension } from "@/lib/review-types";

export function CalibrationReview({
  dimensions,
  reports,
  scores
}: {
  dimensions: RubricDimension[];
  reports: CalibrationReport[];
  scores: HumanScore[];
}) {
  const latest = reports.at(-1);
  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="m-0 font-mono text-[11px] uppercase tracking-[0.12em] text-muted">Calibration review</p>
            <h2 className="mb-0 mt-2 text-xl font-semibold tracking-tight text-ink">Review disagreement before trusting a judge</h2>
            <p className="mb-0 mt-2 max-w-2xl text-sm leading-relaxed text-muted">
              Human score corrections remain append-only. When an eligible local report exists, the adapter
              rebuilds it from its frozen split and verified observations.
            </p>
          </div>
          <Chip label={`${scores.length} score events`} tone={scores.length === 0 ? "neutral" : "purple"} />
        </div>
      </Card>

      {latest ? (
        <>
          <Metrics report={latest} dimensions={dimensions} />
          <Disagreements report={latest} dimensions={dimensions} />
        </>
      ) : (
        <EmptyState
          body="Finalize a rubric, attach judged rollout evidence, and record local human scores. A calibration report appears only when W6 has a verified frozen split."
          title="No calibration report yet"
        />
      )}
      <ScoreHistory scores={scores} />
    </div>
  );
}

function Metrics({ dimensions, report }: { dimensions: RubricDimension[]; report: CalibrationReport }) {
  const tone =
    report.status === "ready_for_approval"
      ? "success"
      : report.status === "insufficient"
        ? "warning"
        : "purple";
  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <BarChart3 aria-hidden="true" className="size-4 text-muted" />
          <h3 className="m-0 text-sm font-semibold text-ink">Latest local calibration</h3>
        </div>
        <Chip label={report.status.replaceAll("_", " ")} tone={tone} />
      </div>
      <dl className="mt-4 grid gap-4 border-y border-line py-4 sm:grid-cols-2 lg:grid-cols-3">
        <Metric label="Eligible labels" value={String(report.eligible_label_count)} />
        <Metric label="Recommended labels" value={String(report.recommended_label_count)} />
        <Metric label="Report" value={report.report_id} />
      </dl>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[560px] border-separate border-spacing-0 text-left text-sm">
          <thead>
            <tr className="text-xs uppercase tracking-[0.08em] text-muted">
              <th className="border-b border-line pb-3 font-medium">Dimension</th>
              <th className="border-b border-line pb-3 font-medium">MAE</th>
              <th className="border-b border-line pb-3 font-medium">Rank agreement</th>
              <th className="border-b border-line pb-3 font-medium">Mean optimistic error</th>
            </tr>
          </thead>
          <tbody>
            {report.dimension_metrics.map((metric) => (
              <tr key={metric.dimension_id}>
                <td className="border-b border-line py-3 text-[#363636]">
                  {dimensions.find((item) => item.dimension_id === metric.dimension_id)?.name ?? metric.dimension_id}
                </td>
                <td className="border-b border-line py-3 font-mono text-muted">{metricValue(metric.mae)}</td>
                <td className="border-b border-line py-3 font-mono text-muted">{metricValue(metric.rank_agreement)}</td>
                <td className="border-b border-line py-3 font-mono text-muted">
                  {metricValue(metric.mean_optimistic_error)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Disagreements({ dimensions, report }: { dimensions: RubricDimension[]; report: CalibrationReport }) {
  return (
    <Card>
      <div className="flex items-center gap-2">
        <AlertTriangle aria-hidden="true" className="size-4 text-warning" />
        <h3 className="m-0 text-sm font-semibold text-ink">Worst disagreements</h3>
      </div>
      {report.worst_disagreements.length === 0 ? (
        <p className="mb-0 mt-3 text-sm text-muted">No disagreement rows were stored in this report.</p>
      ) : (
        <div className="mt-4 space-y-3">
          {report.worst_disagreements.map((item) => {
            const prediction = item.prediction;
            const name = dimensions.find((dimension) => dimension.dimension_id === prediction.dimension_id)?.name;
            return (
              <article className="rounded-[var(--radius-md)] bg-warning-soft p-4" key={`${prediction.rollout_id}-${prediction.dimension_id}`}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="m-0 text-sm font-semibold text-[#363636]">{name ?? prediction.dimension_id}</p>
                    <p className="mb-0 mt-1 font-mono text-xs text-muted">
                      rollout {prediction.rollout_id} · lineage {prediction.lineage_id}
                    </p>
                  </div>
                  <Chip label={item.direction} tone="warning" />
                </div>
                <div className="mt-3 grid gap-3 text-sm sm:grid-cols-4">
                  <Metric label="Judge raw" value={String(prediction.raw_score)} />
                  <Metric label="Human" value={String(prediction.human_score)} />
                  <Metric label="Calibrated" value={prediction.calibrated_score.toFixed(1)} />
                  <Metric label="Absolute error" value={prediction.absolute_error.toFixed(1)} />
                </div>
              </article>
            );
          })}
        </div>
      )}
    </Card>
  );
}

function ScoreHistory({ scores }: { scores: HumanScore[] }) {
  return (
    <Card>
      <div className="flex items-center gap-2">
        <History aria-hidden="true" className="size-4 text-muted" />
        <h3 className="m-0 text-sm font-semibold text-ink">Local score history</h3>
      </div>
      {scores.length === 0 ? (
        <p className="mb-0 mt-3 text-sm text-muted">No human score corrections have been recorded.</p>
      ) : (
        <ol className="mt-4 space-y-2">
          {scores.map((score) => (
            <li className="flex flex-wrap items-center justify-between gap-3 rounded-[var(--radius-md)] bg-surface-subtle px-3 py-2" key={score.label_id}>
              <span className="text-sm text-[#474747]">
                {score.rollout_id} · {score.dimension_id}
              </span>
              <span className="flex items-center gap-2 font-mono text-xs text-muted">
                {score.supersedes_label_id ? <CheckCircle2 aria-label="Correction" className="size-3 text-success" /> : null}
                score {score.score}
              </span>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="font-mono text-[11px] uppercase tracking-[0.09em] text-muted">{label}</dt>
      <dd className="mb-0 mt-1 break-words text-sm font-medium text-[#363636]">{value}</dd>
    </div>
  );
}

function metricValue(value: number | null): string {
  return value === null ? "Not available" : value.toFixed(2);
}
