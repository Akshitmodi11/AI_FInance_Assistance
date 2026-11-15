"""Insights tab: the committed snapshot has what the tab reads; the tab renders headlessly."""

from streamlit.testing.v1 import AppTest

from src.web_app.tabs import insights


def test_snapshot_has_every_section_the_tab_reads():
    m = insights.load_metrics()
    assert m is not None, "run `python -m scripts.collect_metrics`"
    assert {
        "config",
        "kb",
        "retrieval",
        "routing",
        "safety",
        "latency",
        "tests",
        "experiments",
    } <= m.keys()
    modes = {r["mode"] for r in m["retrieval"]["rows"]}
    assert m["config"]["retrieval_mode"] in modes
    for r in m["retrieval"]["rows"]:
        assert 0 <= r["recall"] <= 1 and 0 <= r["mrr"] <= 1 and "by_type" in r
    assert sum(m["kb"]["by_category"].values()) == m["kb"]["chunks"]


def test_missing_snapshot_shows_hint(tmp_path):
    assert insights.load_metrics(tmp_path / "nope.json") is None


def _render():
    from types import SimpleNamespace

    from src.web_app.tabs import insights

    # Stub the app services: real Deps.from_settings() loads .env into os.environ (for LangSmith),
    # which would leak API keys into later tests.
    insights.get_deps = lambda: SimpleNamespace(market=SimpleNamespace(stats={"cache_hits": 2}))
    insights.render()


def test_tab_renders_without_errors(monkeypatch):
    monkeypatch.setattr(insights, "get_deps", insights.get_deps)  # restored after the test
    at = AppTest.from_function(_render, default_timeout=60).run()
    assert not at.exception
    labels = [x.label for x in at.metric]
    assert labels[:2] == ["Recall@5", "MRR"] and "Safety prompts" in labels
    assert any("Experiment log" in x.value for x in at.markdown)
