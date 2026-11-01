"""Golden equivalence of the default mining path with the pre-geometry implementation.

Every digest below was captured from the mining code as it stands on ``main``, by extracting that
revision with ``git archive`` and mining the two fixture corpora through it. They are the bytes an
existing project replays, so ``MiningSpec()`` must keep reproducing them exactly: a selection
geometry, a clustering backend or a sampling strategy added beside the default must be reachable
only by naming it, never by changing what the default produces.

A change here is therefore never a test fix. If one of these digests moves, either the default path
changed (which needs a deliberate decision and a fresh capture from the new default) or a shared
helper drifted while being moved, which is the failure this file exists to catch.
"""

from __future__ import annotations

import json

from exp.common.core.artifacts import (
    ArtifactInput,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from exp.common.traces import Trace
from exp.simulation.mining.bindings import bindings_for_mining, task_set_content_id
from exp.simulation.mining.service import MiningSpec, TaskMiningResult, mine_tasks
from exp.simulation.mining.service_test import ContentStableEmbedder, IndexEmbedder, _trace

_DATASET = ArtifactInput(artifact_id="traces-golden", sha256="c" * 64)

_SPECIALTY_TASK_SET_ID = "task-set-c61da6ab3b7f95123c88"
_SPECIALTY_TASKS_SHA256 = "af7eafdf8ad327f1d26e6a234de6efb8506d650468de3067d779ee438550b61f"
_SPECIALTY_COVERAGE_SHA256 = "078e6313ceca6d7367996e40d226c1ddcbc4509541bfb5664b8caaf369725f8c"
_SPECIALTY_WEIGHTS_SHA256 = "430360f180343da290ee661389c9d7facbd9d61ab114086d44241bc4f09246ac"
_SPECIALTY_TASK_IDS = (
    "task-15899510273618546626",
    "task-ce39383b04e946800b2f",
    "task-1e288c4de7d8c1d78613",
    "task-8932634077744a94e2c9",
    "task-03ebca2fde6100fbf70b",
    "task-b3cc53768f1b5821dad7",
    "task-c047fecc858faaf0e814",
    "task-eacc12cd8b1892ce5abc",
    "task-350471864027cb8b39b4",
    "task-a1752399204173fdfb5f",
    "task-866dfc081267e6617f71",
    "task-759a33d60fbe268cc3d5",
    "task-2cc6f2fa4a55671033b0",
    "task-0574166482081242b091",
    "task-580812525ac5448258e5",
    "task-558c87c703f15c99e283",
    "task-6f9c85354389e1cb3b5f",
    "task-e5e73ebdcdccd05130a3",
    "task-18b112b949e6dffc92ce",
    "task-8863a5819d13caf271c1",
    "task-f66b0c2b11d56d32a627",
    "task-d1eaf95cf6cd8e9d1441",
    "task-c5a587999e8b9a8df065",
    "task-ff7d13807b91bca22d4a",
    "task-905dc2d85aa0f525b30f",
    "task-29308f28ddf2b1b354a9",
    "task-5c854511e64591d957c8",
    "task-465a0bc038210e308a13",
    "task-843dca7bfa6c6ba5a293",
    "task-73c4afb3ef06a831cb4b",
    "task-726ed0573acb3aa92018",
    "task-06ac5a70acae8826a16b",
    "task-f60113816312d47964b9",
    "task-ee4a78fe8c7d38c38d27",
    "task-b54765666e3bcd4ac261",
    "task-371b3892fbeb9ba8cdc7",
    "task-e7e16f2d6dad44c7ab7f",
    "task-e0d16758c41e9a678171",
    "task-b3186fa98dc4b20e515f",
    "task-9d56d805200c1774eefb",
    "task-d57f4f73da38338d2ed3",
    "task-2f96a983719abc313466",
    "task-1e1647c8ffe6d1efa55a",
    "task-19d77f13f58359182e6b",
    "task-831c4153be0944b37c64",
    "task-5c63a40b81ace2c9dd60",
    "task-753c9f598f9b309db7e8",
    "task-e5003d2d4ee8928b73a9",
    "task-4b6a5f05b104903ed6f8",
    "task-59adf033aef74f9d11a2",
    "task-a211de7772bbd4b1ecc3",
    "task-274469cd0d9ff724e15f",
    "task-ee5d045fb457578f1ffc",
    "task-5bd02012a110e3123307",
    "task-2964c5a329c909c172e0",
    "task-658140067fcf75d59236",
    "task-436e958c033420c056f3",
    "task-120b4b869406112569c1",
    "task-cb601c36fffecc0369cf",
    "task-df7e8588bab5a6d6a163",
    "task-ecde63fedb39a6943dc0",
    "task-3766e30ef91b3f3243cf",
    "task-09ffd7df340bd53113a5",
    "task-a8ba351532fe947f582e",
    "task-a847636f63521355f86b",
    "task-019c62249b5f6fffdc35",
    "task-ea56b170a12d0bde0cb2",
    "task-fc18c13a5be6f4a30786",
    "task-c4d709eededd3c5eb7bb",
    "task-7a3c98ba30307d8bb475",
)

_DEFAULT_TASK_SET_ID = "task-set-4f38ea2e9e05a032a3a0"
_DEFAULT_TASKS_SHA256 = "05113abe9998d1b096481cee948beaa78b6a5e7fbd83cf722676132cdff8b70a"
_DEFAULT_COVERAGE_SHA256 = "dcc27de1f91157a70769e0c243680564d3aed90ad3a97981a987f5edf9ee4f6c"
_DEFAULT_WEIGHTS_SHA256 = "ec28ace43b46b79843324be629433be566c4361fbe37726c08bb2bd0ad2e8733"
_DEFAULT_TASK_IDS = (
    "task-509a72a4ca17931cb300",
    "task-f6692f2378534430c64d",
    "task-68ed59c2108b055279cb",
    "task-dd69ea3739eb2491f8ad",
    "task-4ff9e2ae57aa8ceaab01",
    "task-fb81aae71057369c9667",
    "task-9e18152735e756c74d0a",
    "task-c78465db143661dad14e",
    "task-a84085aa796739754ec8",
    "task-3db43832648c3a6b2213",
    "task-ced185db719d953dea4f",
    "task-b5695ea915f1f2c89053",
    "task-0be16595847430d8a708",
    "task-71f0b7899ed154e5d717",
    "task-94ca7d5505b87e2d709a",
    "task-47ff1801baf4cec5e705",
    "task-d667e70646037d53ebae",
    "task-110cc28710e5954b51b9",
    "task-2cf2ad37720277c63998",
    "task-de3b827ec8bc51cd7579",
    "task-c4650415f3b75fc1a9f3",
    "task-efe50641739cf04217f5",
    "task-7bb6ddea05b28602dbbd",
    "task-1f19b5531b73a26854da",
    "task-3c30589f7b6de7c88129",
    "task-e72cbc21c0a03535d686",
    "task-1ae39e33c0f908bc7807",
    "task-0ea895406c69502c8a00",
    "task-591d23552502df1bbf85",
    "task-677c441c5aa4ddf07630",
    "task-72482e873fc80625fa18",
    "task-82b30c192785ef746411",
    "task-0583e65a999f76626908",
    "task-e5965cf06ddb5194f711",
    "task-8ef83ea772e85e511e49",
    "task-4e0545b47cde9f77c7ef",
    "task-6e6a585a33bf35fd9a1c",
    "task-c5c9fb24491be9f1d157",
    "task-c777af0a8cc597ed09aa",
    "task-eb8ac07673ff3b7f0cba",
    "task-d0250fe86903cb1bf5fe",
    "task-8d560a8710d6d8b9b221",
    "task-d02af0f9e38e8ce30d51",
    "task-329cf5bc78e8a4479d89",
    "task-b584965b6e8b5ec7be88",
    "task-6cdd9f20077356f838bf",
    "task-8aecf64309841a2b9f51",
    "task-129989622c15d80fbcf0",
    "task-4d4cf20ea96ce0669d07",
    "task-9aa63e8e46078b7e2762",
    "task-cdcba502b44585d6c087",
    "task-fc43fc8528d13524d6a4",
    "task-98719b92350d48aabfd2",
    "task-a905a4fb30865908c17f",
    "task-a7b2151e033958ef45c0",
    "task-ff39070bf5918158bdda",
    "task-f03fd233e37385cb8e47",
    "task-969bc5718b9a51b77ab8",
    "task-18cc275840f6d4d1ad30",
    "task-f56c3d8763f4774eaeae",
    "task-b3004f9bff57327cfa52",
    "task-12b3d3b842af99f911be",
    "task-50a59b7566a8e800a9d7",
    "task-e07da16263d519227fdb",
    "task-2b9da9e3b44cbd49b8cd",
    "task-6b950b1724d783c5343f",
    "task-db0a8b8f57acca0d9325",
    "task-30e34e47a337ce9898e5",
    "task-6b9b7085a14d78b99d52",
    "task-5166499317951f6d8101",
)


def _specialty_corpus() -> tuple[Trace, ...]:
    """Return the 80-lineage corpus that exercises failures, escalations and long episodes."""
    return tuple(
        _trace(
            index,
            task=f"Resolve specialty customer workflow {index * index}",
            conversation_id=f"lineage-{index}",
            outcome="failure" if index % 11 == 0 else "success",
            escalation=index % 13 == 0,
            span_count=8 if index % 7 == 0 else 1,
        )
        for index in range(80)
    )


def _weights_sha256(result: TaskMiningResult) -> str:
    """Digest the normalized workload weights keyed by task, independent of task order."""
    weights = {task.task_id: task.workload_weight for task in result.tasks}
    payload = json.dumps(weights, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode())


def _assert_matches(
    result: TaskMiningResult,
    *,
    task_set_id: str,
    task_ids: tuple[str, ...],
    tasks_sha256: str,
    coverage_sha256: str,
    weights_sha256: str,
) -> None:
    """Compare one mining result against every pinned byte of the pre-geometry output."""
    assert tuple(task.task_id for task in result.tasks) == task_ids
    assert sha256_bytes(canonical_jsonl_bytes(result.tasks)) == tasks_sha256
    assert sha256_bytes(canonical_json_bytes(result.coverage)) == coverage_sha256
    assert _weights_sha256(result) == weights_sha256
    assert (
        task_set_content_id(_DATASET, result.tasks, result.coverage, bindings_for_mining(result))
        == task_set_id
    )


def test_default_spec_reproduces_the_specialty_corpus_artifact_bytes() -> None:
    """The richest fixture corpus pins task order, weights and the whole coverage report."""
    result = mine_tasks(_specialty_corpus(), embedder=ContentStableEmbedder())

    _assert_matches(
        result,
        task_set_id=_SPECIALTY_TASK_SET_ID,
        task_ids=_SPECIALTY_TASK_IDS,
        tasks_sha256=_SPECIALTY_TASKS_SHA256,
        coverage_sha256=_SPECIALTY_COVERAGE_SHA256,
        weights_sha256=_SPECIALTY_WEIGHTS_SHA256,
    )


def test_default_spec_reproduces_the_100_trace_path_artifact_bytes() -> None:
    """The documented 100-trace operating point is the one most builds land on."""
    result = mine_tasks(tuple(_trace(index) for index in range(100)), embedder=IndexEmbedder())

    _assert_matches(
        result,
        task_set_id=_DEFAULT_TASK_SET_ID,
        task_ids=_DEFAULT_TASK_IDS,
        tasks_sha256=_DEFAULT_TASKS_SHA256,
        coverage_sha256=_DEFAULT_COVERAGE_SHA256,
        weights_sha256=_DEFAULT_WEIGHTS_SHA256,
    )


def test_an_explicit_default_spec_is_the_same_as_no_spec() -> None:
    """Naming today's geometry explicitly must not be a different code path from omitting it."""
    corpus = _specialty_corpus()

    implicit = mine_tasks(corpus, embedder=ContentStableEmbedder())
    explicit = mine_tasks(corpus, MiningSpec(), embedder=ContentStableEmbedder())

    assert tuple(task.task_id for task in explicit.tasks) == _SPECIALTY_TASK_IDS
    assert explicit.coverage == implicit.coverage
