"""ENF-02 guard (Part 5): every endpoint answers with p95 < 500 ms on synthetic run
folders, in process. Lenient on purpose (CI machines vary); the real-data measurement is
`python -m traffic_twin.api.bench` (docs/api.md, latency block)."""

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("duckdb")

from traffic_twin.api import bench  # noqa: E402
from traffic_twin.api.fixtures import make_fake_eta  # noqa: E402
from traffic_twin.api.settings import load_settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LIMIT_MS = 500.0


@pytest.fixture
def settings(api_results: Path, tmp_path: Path):
    for run in api_results.glob("*/run.json"):
        make_fake_eta(run.parent)
    make_fake_eta(tmp_path / "models")
    return replace(
        load_settings(repo_root=REPO_ROOT, environ={}),
        results_dirs=(api_results,),
        evaluation_dir=api_results / "evaluation",
        eta_models_dir=tmp_path / "models",
        default_run=(api_results / "pitie_rules_seed4000",),
    )


def test_every_endpoint_p95_under_500_ms(settings):
    stats, info = bench.bench_inprocess(settings, n=20)
    assert info["runs_indexed"] == 3 and info["eta_model_version"] == "fake-ref-0"
    assert set(stats) == set(bench.endpoints([], None))
    for name, s in stats.items():
        assert s["p95_ms"] < LIMIT_MS, (name, s)
        assert not any(k.startswith("5") for k in s["answers"]), (name, s["answers"])
    assert stats["error (404)"]["answers"] == {"404 unavailable": 20}


def test_latency_block_and_docs_rewrite(settings):
    result = bench.run_bench(settings, n=3, http=False)
    assert result["enf02"]["pass"] and result["http"] is None
    block = bench.latency_block(result)
    assert "ENF-02" in block and "| crossing |" in block and "PASS" in block
    page = f"# API\n\nintro\n\n{bench.BEGIN}\nSTALE-BLOCK\n{bench.END}\n\nafter\n"
    new = bench.rewrite_block(page, block)
    assert new.startswith("# API\n\nintro\n\n" + bench.BEGIN) and new.endswith(
        bench.END + "\n\nafter\n"
    )
    assert "STALE-BLOCK" not in new and block.strip() in new
    with pytest.raises(ValueError):
        bench.rewrite_block("no markers here", block)
