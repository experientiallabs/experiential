"""A narrow public veRL server extension for private ingress and terminal provenance."""

import argparse
import secrets
from typing import Literal

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

from exp.optimize.claas.backends.verl.completion import observe_completion


class ExactVllmHttpServer(vLLMHttpServer):
    """Retain upstream engine ownership and original outputs while restricting HTTP ingress."""

    async def run_server(self, args: argparse.Namespace) -> None:
        """Bind internal HTTP to loopback and record native stop/length before veRL flattens it."""
        self._server_address = "127.0.0.1"
        # Set after upstream renders CLI arguments so this internal key is never logged there.
        args.api_key = [secrets.token_urlsafe(48)]
        await super().run_server(args)
        self._finish_reasons: dict[str, Literal["stop", "length"]] = {}
        self.engine.generate = observe_completion(self.engine.generate, self._finish_reasons)

    async def generate_exact(
        self, prompt_ids: list[int], sampling_params: dict[str, float | int | bool], request_id: str
    ) -> TokenOutput:
        """Preserve the public veRL token result and its observed vLLM completion reason."""
        try:
            output = await super().generate(prompt_ids, sampling_params, request_id)
            reason = self._finish_reasons.get(request_id)
            if output.stop_reason == "completed" and reason is None:
                raise ValueError("upstream generation omitted its native completion reason")
            if reason is not None:
                output.extra_fields["finish_reason"] = reason
            return output
        finally:
            self._finish_reasons.pop(request_id, None)
