"""Shared pytest configuration for the integration-owned test suite."""

from __future__ import annotations

import os
from types import SimpleNamespace


def _install_hosted_cpu_native_contract_stub() -> None:
    """Make DMI API v1 importable for contract-only hosted CPU tests.

    DMI API v1 derives its public hook constants from the native extension's
    ``HOOK_DEFS`` table at import time. GitHub-hosted CPU runners cannot build
    that CUDA extension, but the integration's portable tests need the public
    Python types and constants, not native transport execution. The opt-in
    stub mirrors the immutable API-v1 table and supplies inert configuration
    types. Manual GPU and local native-backed runs do not set the environment
    variable and therefore continue to load the real extension.
    """

    if os.environ.get("DMI_VLLM_TEST_NATIVE_STUB") != "1":
        return

    from monitoring import _native_engine

    class _UnavailableNative:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError(
                "native DMI execution is unavailable in the hosted CPU gate"
            )

    class _UnavailableStageConfig(_UnavailableNative):
        @classmethod
        def clickhouse_insert(cls, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                "native DMI execution is unavailable in the hosted CPU gate"
            )

    hook_defs = (
        (0, "hook_resid_pre", "resid_pre", True, 2, False, 0, 0),
        (1, "hook_ln1", "ln1", True, 2, False, 0, 0),
        (2, "hook_attn_out", "attn_out", True, 0, False, 0, 0),
        (3, "hook_resid_mid", "resid_mid", True, 2, False, 0, 0),
        (4, "attn.hook_attn_scores", "attn_scores", True, 0, True, 4, 0),
        (5, "attn.hook_pattern", "pattern", True, 0, True, 4, 0),
        (6, "attn.hook_q", "q", True, 0, True, 1, 0),
        (7, "attn.hook_k", "k", True, 0, True, 2, 0),
        (8, "attn.hook_v", "v", True, 0, True, 2, 0),
        (9, "attn.hook_z", "z", True, 0, True, 3, 0),
        (11, "hook_ln2", "ln2", True, 2, False, 0, 0),
        (12, "hook_mlp_in", "mlp_in", True, 1, False, 0, 0),
        (13, "hook_mlp_out", "mlp_out", True, 1, False, 0, 0),
        (20, "hook_mlp_post", "mlp_post", True, 1, True, 5, 0),
        (14, "hook_resid_final", "resid_final", False, 2, False, 0, 2),
        (15, "hook_embed", "embed", False, 2, False, 0, 1),
        (16, "hook_pos_embed", "pos_embed", False, 2, False, 0, 1),
        (17, "hook_final_ln", "final_ln", False, 2, False, 0, 2),
        (18, "token_ids", "token_ids", False, 2, False, 6, 1),
        (19, "final_logits", "final_logits", False, 2, False, 7, 2),
        (
            21,
            "mlp.hook_router_logits",
            "router_logits",
            True,
            2,
            False,
            8,
            0,
        ),
        (22, "mlp.hook_topk_ids", "topk_ids", True, 2, False, 9, 0),
        (
            23,
            "mlp.hook_topk_weights",
            "topk_weights",
            True,
            2,
            False,
            10,
            0,
        ),
    )
    native_stub = SimpleNamespace(
        HOOK_DEFS=hook_defs,
        ClickHouseClientConfig=_UnavailableNative,
        DMXHostEngine=_UnavailableNative,
        EnqueuePolicy=_UnavailableNative,
        OnClosedPolicy=_UnavailableNative,
        OnFullPolicy=_UnavailableNative,
        QueueConfig=_UnavailableNative,
        RingConfig=_UnavailableNative,
        StageConfig=_UnavailableStageConfig,
        ThreadFailure=RuntimeError,
    )
    _native_engine._load_extension = lambda: native_stub


_install_hosted_cpu_native_contract_stub()


def pytest_configure(config):
    for marker, description in (
        ("clickhouse", "requires a reachable ClickHouse test instance"),
        ("cpu", "runs without a CUDA-capable GPU"),
        ("e2e", "runs an end-to-end integration workflow"),
        ("gpu", "requires a CUDA-capable GPU"),
        ("slow", "runs a comparatively slow validation"),
        ("vllm", "exercises the installed supported vLLM release"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")
