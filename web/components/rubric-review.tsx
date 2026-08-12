import { Check, GripVertical, Pencil, Plus, Replace, ShieldCheck, X } from "lucide-react";
import { useState, type FormEvent } from "react";

import { Button, Card, Chip, Dialog, EmptyState } from "@/components/ui";
import type {
  ProposedDimension,
  ReviewSnapshot,
  RubricAction,
  RubricDimension,
  Score,
  ScoreAnchor
} from "@/lib/review-types";

type RubricReviewProps = {
  busy: boolean;
  onMutation: (action: RubricAction, payload: Record<string, unknown>) => void;
  snapshot: ReviewSnapshot;
};

type EditingScale = {
  initial?: RubricDimension;
  proposalDimensionId?: string;
};

export function RubricReview({ busy, onMutation, snapshot }: RubricReviewProps) {
  const { rubric_review: review } = snapshot;
  const activeDimensions = review.finalized_rubric?.dimensions ?? review.dimensions;
  const [editingScale, setEditingScale] = useState<EditingScale | null>(null);
  const [confirmingFinalization, setConfirmingFinalization] = useState(false);
  const [replacingAll, setReplacingAll] = useState(false);
  const [authoringReplacement, setAuthoringReplacement] = useState(false);
  const [replacementDimensions, setReplacementDimensions] = useState<RubricDimension[]>([]);
  const [swipeStart, setSwipeStart] = useState<number | null>(null);

  const mutate = (action: RubricAction, payload: Record<string, unknown>) => {
    if (!busy) {
      onMutation(action, payload);
    }
  };

  const acceptedIds = new Set(activeDimensions.map((item) => item.dimension_id));
  const rejectedIds = new Set(review.rejected_dimension_ids);
  const finalized = review.status === "finalized";

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="m-0 font-mono text-[11px] uppercase tracking-[0.12em] text-muted">Rubric scale review</p>
            <h2 className="mb-0 mt-2 text-xl font-semibold tracking-tight text-ink">Keep the judgment scale legible</h2>
            <p className="mb-0 mt-2 max-w-2xl text-sm leading-relaxed text-muted">
              Accept, reject, or edit model proposals before finalization. Every change is delegated to the
              local W6 review draft.
            </p>
          </div>
          <Chip label={finalized ? "human approved" : "draft"} tone={finalized ? "success" : "warning"} />
        </div>
        <div className="mt-5 flex flex-wrap gap-2 border-t border-line pt-4">
          <Button disabled={busy || finalized} onClick={() => setEditingScale({})} type="button" variant="primary">
            <Plus aria-hidden="true" className="size-4" />
            Add scale
          </Button>
          <Button
            disabled={busy || finalized}
            onClick={() => {
              setReplacementDimensions([]);
              setReplacingAll(true);
            }}
            type="button"
          >
            <Replace aria-hidden="true" className="size-4" />
            Design replacement set
          </Button>
          <Button
            disabled={busy || finalized || activeDimensions.length === 0}
            onClick={() => setConfirmingFinalization(true)}
            type="button"
            variant="primary"
          >
            <ShieldCheck aria-hidden="true" className="size-4" />
            Finalize rubric
          </Button>
        </div>
      </Card>

      <section aria-labelledby="proposals-heading">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="m-0 text-sm font-semibold text-ink" id="proposals-heading">
            Proposed scales
          </h2>
          <p className="m-0 text-xs text-muted">Focus a card, then use A, R, or E.</p>
        </div>
        {review.proposals.length === 0 ? (
          <EmptyState body="No proposal artifacts are available in this local project." title="No proposed scales" />
        ) : (
          <div className="grid gap-4 xl:grid-cols-2">
            {review.proposals.flatMap((proposal) =>
              proposal.dimensions.map((proposed) => {
                const dimensionId = proposed.dimension.dimension_id;
                const accepted = acceptedIds.has(dimensionId);
                const rejected = rejectedIds.has(dimensionId);
                return (
                  <ProposalCard
                    accepted={accepted}
                    busy={busy || finalized}
                    key={`${proposal.proposal_id}-${dimensionId}`}
                    onAccept={() => mutate("accept", { dimension_id: dimensionId })}
                    onEdit={() => setEditingScale({ initial: proposed.dimension, proposalDimensionId: dimensionId })}
                    onReject={() => mutate("reject", { dimension_id: dimensionId })}
                    onSwipeEnd={(delta) => {
                      if (delta > 48) {
                        mutate("accept", { dimension_id: dimensionId });
                      }
                      if (delta < -48) {
                        mutate("reject", { dimension_id: dimensionId });
                      }
                    }}
                    onSwipeStart={setSwipeStart}
                    proposed={proposed}
                    rejected={rejected}
                    swipeStart={swipeStart}
                  />
                );
              })
            )}
          </div>
        )}
      </section>

      <section aria-labelledby="active-scales-heading">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="m-0 text-sm font-semibold text-ink" id="active-scales-heading">
            {finalized ? "Finalized scales" : "Active scales"}
          </h2>
          <span className="font-mono text-xs text-muted">{activeDimensions.length}</span>
        </div>
        {activeDimensions.length === 0 ? (
          <EmptyState
            body="Accept a proposal or add a human-authored scale. Finalization remains unavailable until at least one scale is active."
            title="No active scales"
          />
        ) : (
          <div className="space-y-3">
            {activeDimensions.map((dimension, index) => (
              <ActiveScaleCard
                canEdit={!finalized && !busy}
                dimension={dimension}
                index={index}
                key={dimension.dimension_id}
                onEdit={() => setEditingScale({ initial: dimension, proposalDimensionId: dimension.dimension_id })}
                onMove={(direction) => {
                  const target = index + direction;
                  if (target < 0 || target >= activeDimensions.length) {
                    return;
                  }
                  const reordered = [...activeDimensions];
                  [reordered[index], reordered[target]] = [reordered[target], reordered[index]];
                  mutate("order", { dimension_ids: reordered.map((item) => item.dimension_id) });
                }}
              />
            ))}
          </div>
        )}
      </section>

      {editingScale ? (
        <ScaleEditorDialog
          initial={editingScale.initial}
          onClose={() => setEditingScale(null)}
          onSave={(dimension) => {
            if (editingScale.proposalDimensionId) {
              mutate("edit", {
                dimension_id: editingScale.proposalDimensionId,
                name: dimension.name,
                description: dimension.description,
                anchors: dimension.anchors
              });
            } else {
              mutate("add", { dimension });
            }
            setEditingScale(null);
          }}
        />
      ) : null}
      {confirmingFinalization ? (
        <Dialog onClose={() => setConfirmingFinalization(false)} title="Finalize this rubric?">
          <p className="mt-0 text-sm leading-relaxed text-muted">
            Finalization writes the immutable human-approved rubric. It cannot be reopened from this local
            review flow.
          </p>
          <div className="mt-5 flex justify-end gap-2">
            <Button onClick={() => setConfirmingFinalization(false)} type="button">
              Keep editing
            </Button>
            <Button
              onClick={() => {
                mutate("finalize", { confirmed: true });
                setConfirmingFinalization(false);
              }}
              type="button"
              variant="primary"
            >
              Confirm finalization
            </Button>
          </div>
        </Dialog>
      ) : null}
      {replacingAll && !authoringReplacement ? (
        <ReplaceAllDialog
          activeDimensions={activeDimensions}
          onAuthor={() => setAuthoringReplacement(true)}
          onClose={() => setReplacingAll(false)}
          onSubmit={() => {
            mutate("replace_all", { dimensions: replacementDimensions });
            setReplacingAll(false);
          }}
          proposals={review.proposals.flatMap((proposal) =>
            proposal.dimensions.map((item) => item.dimension)
          )}
          selected={replacementDimensions}
          setSelected={setReplacementDimensions}
        />
      ) : null}
      {authoringReplacement ? (
        <ScaleEditorDialog
          onClose={() => setAuthoringReplacement(false)}
          onSave={(dimension) => {
            setReplacementDimensions((current) => [
              ...current.filter((item) => item.dimension_id !== dimension.dimension_id),
              dimension
            ]);
            setAuthoringReplacement(false);
          }}
        />
      ) : null}
    </div>
  );
}

function ReplaceAllDialog({
  activeDimensions,
  onAuthor,
  onClose,
  onSubmit,
  proposals,
  selected,
  setSelected
}: {
  activeDimensions: RubricDimension[];
  onAuthor: () => void;
  onClose: () => void;
  onSubmit: () => void;
  proposals: RubricDimension[];
  selected: RubricDimension[];
  setSelected: (dimensions: RubricDimension[]) => void;
}) {
  const candidates = [...activeDimensions, ...proposals].filter(
    (dimension, index, all) =>
      all.findIndex((item) => item.dimension_id === dimension.dimension_id) === index
  );
  const selectedIds = new Set(selected.map((item) => item.dimension_id));
  return (
    <Dialog onClose={onClose} title="Design the complete replacement set">
      <p className="mt-0 text-sm leading-relaxed text-muted">
        Select existing or proposed scales, or author a new scale. Submission replaces every
        active scale with this exact ordered set.
      </p>
      <div className="mt-4 space-y-2">
        {candidates.map((dimension) => (
          <label
            className="flex items-start gap-3 rounded-[var(--radius-md)] border border-line p-3"
            key={dimension.dimension_id}
          >
            <input
              checked={selectedIds.has(dimension.dimension_id)}
              className="mt-1"
              onChange={(event) =>
                setSelected(
                  event.target.checked
                    ? [...selected, dimension]
                    : selected.filter((item) => item.dimension_id !== dimension.dimension_id)
                )
              }
              type="checkbox"
            />
            <span>
              <span className="block text-sm font-semibold text-ink">{dimension.name}</span>
              <span className="mt-1 block text-xs leading-relaxed text-muted">
                {dimension.description}
              </span>
            </span>
          </label>
        ))}
      </div>
      {selected.length > 0 ? (
        <ol className="mt-4 rounded-[var(--radius-md)] bg-surface-subtle p-3 text-sm text-muted">
          {selected.map((dimension, index) => (
            <li key={dimension.dimension_id}>
              {index + 1}. {dimension.name}
            </li>
          ))}
        </ol>
      ) : null}
      <div className="mt-5 flex flex-wrap justify-end gap-2 border-t border-line pt-4">
        <Button onClick={onClose} type="button">
          Cancel
        </Button>
        <Button onClick={onAuthor} type="button">
          <Plus aria-hidden="true" className="size-4" />
          Author replacement scale
        </Button>
        <Button disabled={selected.length === 0} onClick={onSubmit} type="button" variant="primary">
          Replace all scales
        </Button>
      </div>
    </Dialog>
  );
}

function ProposalCard({
  accepted,
  busy,
  onAccept,
  onEdit,
  onReject,
  onSwipeEnd,
  onSwipeStart,
  proposed,
  rejected,
  swipeStart
}: {
  accepted: boolean;
  busy: boolean;
  onAccept: () => void;
  onEdit: () => void;
  onReject: () => void;
  onSwipeEnd: (delta: number) => void;
  onSwipeStart: (position: number | null) => void;
  proposed: ProposedDimension;
  rejected: boolean;
  swipeStart: number | null;
}) {
  const status = accepted ? "accepted" : rejected ? "rejected" : "needs review";
  const tone = accepted ? "success" : rejected ? "danger" : "warning";
  return (
    <Card className="relative overflow-hidden">
      <article
        aria-label={`${proposed.dimension.name} proposal`}
        className="outline-none"
        onKeyDown={(event) => {
          if (event.currentTarget !== event.target || busy) {
            return;
          }
          const command = event.key.toLowerCase();
          if (command === "a") {
            event.preventDefault();
            onAccept();
          }
          if (command === "r") {
            event.preventDefault();
            onReject();
          }
          if (command === "e") {
            event.preventDefault();
            onEdit();
          }
        }}
        onPointerDown={(event) => onSwipeStart(event.clientX)}
        onPointerUp={(event) => {
          if (swipeStart !== null) {
            onSwipeEnd(event.clientX - swipeStart);
          }
          onSwipeStart(null);
        }}
        tabIndex={0}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="m-0 font-mono text-[11px] uppercase tracking-[0.12em] text-muted">Proposal</p>
            <h3 className="mb-0 mt-2 text-base font-semibold text-ink">{proposed.dimension.name}</h3>
          </div>
          <Chip label={status} tone={tone} />
        </div>
        <p className="mb-0 mt-3 text-sm leading-relaxed text-muted">{proposed.dimension.description}</p>
        <AnchorList anchors={proposed.dimension.anchors} />
        <p className="mb-0 mt-4 text-xs leading-relaxed text-muted">
          Evidence: {proposed.source_rollout_ids.join(", ")} · spans {proposed.evidence_span_ids.join(", ")}
        </p>
        <div className="mt-4 flex flex-wrap gap-2 border-t border-line pt-4">
          <Button disabled={busy || accepted || rejected} onClick={onAccept} type="button" variant="primary">
            <Check aria-hidden="true" className="size-4" />
            Accept
          </Button>
          <Button disabled={busy || accepted || rejected} onClick={onReject} type="button" variant="danger">
            <X aria-hidden="true" className="size-4" />
            Reject
          </Button>
          <Button disabled={busy || rejected} onClick={onEdit} type="button">
            <Pencil aria-hidden="true" className="size-4" />
            Edit
          </Button>
        </div>
      </article>
    </Card>
  );
}

function ActiveScaleCard({
  canEdit,
  dimension,
  index,
  onEdit,
  onMove
}: {
  canEdit: boolean;
  dimension: RubricDimension;
  index: number;
  onEdit: () => void;
  onMove: (direction: -1 | 1) => void;
}) {
  return (
    <Card className="p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex gap-3">
          <GripVertical aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-muted-2" />
          <div>
            <p className="m-0 font-mono text-[11px] uppercase tracking-[0.12em] text-muted">Scale {index + 1}</p>
            <h3 className="mb-0 mt-1 text-base font-semibold text-ink">{dimension.name}</h3>
            <p className="mb-0 mt-1 text-sm leading-relaxed text-muted">{dimension.description}</p>
          </div>
        </div>
        {canEdit ? (
          <div className="flex gap-2">
            <Button aria-label={`Move ${dimension.name} up`} className="min-h-8 px-2" onClick={() => onMove(-1)} type="button">
              ↑
            </Button>
            <Button aria-label={`Move ${dimension.name} down`} className="min-h-8 px-2" onClick={() => onMove(1)} type="button">
              ↓
            </Button>
            <Button onClick={onEdit} type="button">
              <Pencil aria-hidden="true" className="size-4" />
              Edit
            </Button>
          </div>
        ) : null}
      </div>
      <AnchorList anchors={dimension.anchors} />
    </Card>
  );
}

function AnchorList({ anchors }: { anchors: ScoreAnchor[] }) {
  return (
    <ol className="mt-4 grid gap-2 border-t border-line pt-4 sm:grid-cols-2 lg:grid-cols-3">
      {anchors.map((anchor) => (
        <li className="flex gap-2 text-sm leading-relaxed text-muted" key={anchor.score}>
          <span className="grid size-5 shrink-0 place-items-center rounded-full bg-surface-subtle font-mono text-[11px] text-[#474747]">
            {anchor.score}
          </span>
          {anchor.description}
        </li>
      ))}
    </ol>
  );
}

function ScaleEditorDialog({
  initial,
  onClose,
  onSave
}: {
  initial?: RubricDimension;
  onClose: () => void;
  onSave: (dimension: RubricDimension) => void;
}) {
  const [dimensionId, setDimensionId] = useState(initial?.dimension_id ?? "");
  const [name, setName] = useState(initial?.name ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [anchorDescriptions, setAnchorDescriptions] = useState<string[]>(
    initial?.anchors.map((anchor) => anchor.description) ?? Array.from({ length: 6 }, () => "")
  );

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedId = (dimensionId || name)
      .trim()
      .toLowerCase()
      .replaceAll(/[^a-z0-9]+/g, "-")
      .replaceAll(/^-|-$/g, "");
    onSave({
      dimension_id: normalizedId || "human-scale",
      name: name.trim(),
      description: description.trim(),
      anchors: anchorDescriptions.map((item, index) => ({
        score: index as Score,
        description: item.trim()
      }))
    });
  };

  return (
    <Dialog onClose={onClose} title={initial ? `Edit ${initial.name}` : "Add a rubric scale"}>
      <form className="space-y-4" onSubmit={submit}>
        <label className="block text-sm font-medium text-[#363636]">
          Scale name
          <input
            className="mt-1.5 block w-full rounded-[var(--radius-md)] border border-line-strong bg-surface px-3 py-2 text-sm"
            onChange={(event) => setName(event.target.value)}
            required
            value={name}
          />
        </label>
        <label className="block text-sm font-medium text-[#363636]">
          Stable identifier
          <input
            className="mt-1.5 block w-full rounded-[var(--radius-md)] border border-line-strong bg-surface px-3 py-2 font-mono text-sm"
            onChange={(event) => setDimensionId(event.target.value)}
            placeholder="task-success"
            value={dimensionId}
          />
        </label>
        <label className="block text-sm font-medium text-[#363636]">
          What does this scale measure?
          <textarea
            className="mt-1.5 block min-h-20 w-full rounded-[var(--radius-md)] border border-line-strong bg-surface px-3 py-2 text-sm"
            onChange={(event) => setDescription(event.target.value)}
            required
            value={description}
          />
        </label>
        <fieldset className="border-t border-line pt-4">
          <legend className="text-sm font-medium text-[#363636]">Zero-to-five anchors</legend>
          <div className="mt-3 grid gap-3">
            {anchorDescriptions.map((anchor, index) => (
              <label className="grid grid-cols-[24px_minmax(0,1fr)] items-center gap-2 text-sm text-muted" key={index}>
                <span className="font-mono text-xs">{index}</span>
                <input
                  className="w-full rounded-[var(--radius-md)] border border-line-strong bg-surface px-3 py-2 text-sm text-ink"
                  onChange={(event) => {
                    setAnchorDescriptions((previous) =>
                      previous.map((item, itemIndex) => (itemIndex === index ? event.target.value : item))
                    );
                  }}
                  required
                  value={anchor}
                />
              </label>
            ))}
          </div>
        </fieldset>
        <div className="flex justify-end gap-2 border-t border-line pt-4">
          <Button onClick={onClose} type="button">
            Cancel
          </Button>
          <Button type="submit" variant="primary">
            Save scale
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
