"""Keep official tests intact while excluding their groups from teaching."""
from pathlib import Path


def test_gauntlet_reserves_test_groups_and_excludes_conflicting_training_labels(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.quality.n8n_gauntlet.prepare import group, partition, rows

    training = rows([(f"support request {i}", "a") for i in range(100)] + [
        ("OFFICIAL test!", "a"), ("conflicting example", "a"),
        ("conflicting example", "b"), ("support request 1", "a"),
    ], "train")
    test = rows([("official test", "a"), ("OFFICIAL test!", "b")], "test")
    result = partition(training, test)
    assert result["splits"]["test"] == test  # Even awkward official rows remain.
    assert result["removed"]["training"] == {
        "reserved_overlap": 1, "conflicting_label": 2, "duplicate": 1,
    }
    sets = [{r["group"] for r in split} for split in result["splits"].values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    assert [len(result["splits"][s]) for s in ("fit", "calibration", "development")] == [60, 20, 20]
    assert group(" Straße! ") == group("STRASSE")


def test_gauntlet_official_validation_takes_priority_over_training(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.quality.n8n_gauntlet.prepare import partition, rows

    training = rows([(f"request {i}", "a") for i in range(100)], "train")
    validation = rows([("request 0", "a"), ("final test", "a")], "val")
    test = rows([("final test!", "a")], "test")
    result = partition(training, test, validation)
    assert result["splits"]["development"] == validation[:1]
    assert result["removed"]["training"]["reserved_overlap"] == 1
    assert result["removed"]["development"]["reserved_overlap"] == 1
    assert len(result["splits"]["calibration"]) == 20
    assert len(result["splits"]["fit"]) == 79


def test_gauntlet_metrics_count_unsupported_acceptance_separately(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.quality.n8n_gauntlet.evaluate import quality, gates

    outcomes = [dict(truth="a", suggestion="a", category="a", needsReview=False) for _ in range(79)]
    outcomes += [dict(truth="b", suggestion="a", category="a", needsReview=False)]
    outcomes += [dict(truth="b", suggestion="b", category=None, needsReview=True) for _ in range(20)]
    outcomes += [dict(truth="oos", suggestion="a", category="a", needsReview=False)]
    outcomes += [dict(truth="oos", suggestion="oos", category=None, needsReview=True) for _ in range(99)]
    metrics = quality(outcomes)
    assert metrics["supported_coverage"]["fraction"] == .8
    assert metrics["accepted_accuracy"]["fraction"] == 79 / 80
    assert metrics["raw_supported_accuracy"]["fraction"] == .99
    assert metrics["oos_false_acceptance"]["fraction"] == .01
    passed = gates(metrics, 5.0, 0)
    assert passed["coverage_at_least_80"]
    assert not passed["accepted_accuracy_at_least_99"]
    assert passed["oos_false_acceptance_at_most_1"]
    assert not passed["adapter_p95_below_5_ms"]
    empty = quality([dict(truth="a", suggestion="a", category=None, needsReview=True)])
    assert empty["accepted_accuracy"]["fraction"] is None
    assert not all(gates(empty, 1, 0).values())


def test_gauntlet_wilson_intervals_include_sampling_uncertainty(monkeypatch):
    import pytest
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.quality.n8n_gauntlet.evaluate import proportion

    assert proportion(0, 0)["wilson95"] is None
    assert proportion(0, 100)["wilson95"] == pytest.approx([0, .0369934982])
    assert proportion(100, 100)["wilson95"] == pytest.approx([.9630065018, 1])
    assert proportion(99, 100)["wilson95"] == pytest.approx([.9455138038, .9982325679])
