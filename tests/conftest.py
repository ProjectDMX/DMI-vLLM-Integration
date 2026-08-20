"""Shared pytest configuration for the integration-owned test suite."""


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
