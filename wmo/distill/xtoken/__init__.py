"""Cross-tokenizer teacher scoring: INERT on this build, nothing imports it yet.

Read this first: no code outside this package calls anything in it. The
modules landed ahead of their only consumer, the cross-tokenizer distillation
mode in PR #258 ("Execute the cross-tokenizer teacher: GLM-5.2 scores Qwen
student rollouts", branch `feat/distill-xtoken`), and they are inert until
that lands. `wmo.distill.loop` refuses `teacher.backend = "openai_compat"` at
run start precisely because this package has no caller, so a config that
reaches for chunk alignment fails loudly instead of being served from Tinker
under the wrong alignment.

They are kept here, rather than deleted and re-added, because the files are
what PR #258 builds on. This module is the package's front door: it says
plainly what state the package is in and re-exports the surface that PR
consumes, so a reader who greps for a caller and finds none has an
explanation rather than a mystery.

What the pieces do, in the order that mode uses them:

- `teacher_render` renders the canonical message list with the TEACHER's chat
  template and reports which teacher token ranges cover byte-identical
  message content (assistant reasoning, visible text, tool-call argument
  values). Turn framing and tool-call syntax differ between templates and are
  deliberately left uncovered.
- `byte_offsets` gives exact per-token byte offsets on both sides with no
  re-encoding, which is what lets the two token streams be compared in one
  byte space. Re-encoding is not an option on the student side: sampling emits
  non-canonical BPE, so `encode(decode(ids)) != ids` for a large minority of
  real spans.
- `chunks` holds the alignment result (`ChunkPlan`) and turns it into the
  per-token advantages the existing `importance_sampling` wire format already
  carries, via `attach_chunk_advantages`.
- `prompt_logprobs` is the teacher's only network surface: one
  `/v1/completions` call per trajectory against a self-hosted vLLM server,
  sending the prompt as token ids and reading `prompt_logprobs` back. It is
  deliberately not re-exported below: it is a live HTTP client, and importing
  this package should not read as a licence to start calling one.

Invariants this package exists to protect, each of which was a real defect
found while building it:

- TITO is untouched. The student always trains on its exact sampled ids; every
  round trip here is scoring-side only.
- A chunk's influence is its reverse-KL gap, not its token count.
- Centering is over chunk TOTALS. Token-level centering inverts long chunks.
- A token no chunk covers keeps advantage exactly 0.0, which IS the mask on
  the wire, so centering must never touch it.
"""

from wmo.distill.xtoken.byte_offsets import span_byte_ends
from wmo.distill.xtoken.chunks import (
    ChunkAdvantageStats,
    ChunkPlan,
    ChunkSpan,
    attach_chunk_advantages,
)
from wmo.distill.xtoken.teacher_render import ContentIsland, TeacherRender, render_for_teacher

__all__ = [
    "ChunkAdvantageStats",
    "ChunkPlan",
    "ChunkSpan",
    "ContentIsland",
    "TeacherRender",
    "attach_chunk_advantages",
    "render_for_teacher",
    "span_byte_ends",
]
