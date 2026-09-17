"""Single-job CLI adapter using the same resident native veRL training implementation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.resident import ResidentTrainer
from exp.optimize.claas.training_contracts import TrainingJob, TrainingResult

logger = logging.getLogger(__name__)


def execute_training_job(job: TrainingJob) -> TrainingResult:
    """Open one burst, recover or commit its exact batch, and close native resources."""
    trainer = ResidentTrainer(
        job.spec,
        ResidentVerlSettings(
            checkpoint_root=Path(job.checkpoint_root),
            lineage_id=job.lineage_id,
        ),
    )
    try:
        trainer.initialize(job.resume_checkpoint)
        return trainer.train(job.batch)
    finally:
        trainer.close()


def main() -> None:
    """Execute one serialized job and write its typed completion receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    result = execute_training_job(TrainingJob.model_validate_json(args.job.read_text()))
    args.result.write_text(result.model_dump_json())
    logger.info("Completed upstream veRL optimizer step %s", result.checkpoint.step)


if __name__ == "__main__":
    main()
