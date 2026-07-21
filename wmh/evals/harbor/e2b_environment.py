"""Harbor E2B task environment using WMH's shared sandbox-create admission gate."""

from __future__ import annotations

import asyncio
from typing import override

from e2b import AsyncSandbox
from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import NetworkMode

from wmh.harness.e2b_sandbox import acquire_e2b_create_slot_async

_CREATE_ATTEMPTS = 2
_CREATE_RETRY_DELAY_S = 1.0


class WmhE2BEnvironment(E2BEnvironment):
    """Preserve Harbor's E2B behavior while pacing every provider create attempt."""

    @override
    async def _create_sandbox(self) -> None:
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
        }
        for attempt in range(_CREATE_ATTEMPTS):
            await acquire_e2b_create_slot_async()
            try:
                self._sandbox = await AsyncSandbox.create(
                    template=self._template_name,
                    metadata=metadata,
                    envs=self._startup_env(),
                    timeout=86_400,
                    allow_internet_access=(
                        self.network_policy.network_mode != NetworkMode.NO_NETWORK
                    ),
                    network=self._sandbox_create_network_options(),
                )
                return
            except Exception:  # noqa: BLE001 - preserve Harbor's retry-all create contract
                self._sandbox = None
                if attempt + 1 == _CREATE_ATTEMPTS:
                    raise
                await asyncio.sleep(_CREATE_RETRY_DELAY_S)
