"""Keep official tests intact while excluding their groups from teaching."""
from benchmarks.quality.n8n_gauntlet.prepare import group, partition, rows


def test_gauntlet_reserves_test_groups_and_excludes_conflicting_training_labels():
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


def test_gauntlet_official_validation_takes_priority_over_training():
    training = rows([(f"request {i}", "a") for i in range(100)], "train")
    validation = rows([("request 0", "a"), ("final test", "a")], "val")
    test = rows([("final test!", "a")], "test")
    result = partition(training, test, validation)
    assert result["splits"]["development"] == validation[:1]
    assert result["removed"]["training"]["reserved_overlap"] == 1
    assert result["removed"]["development"]["reserved_overlap"] == 1
    assert len(result["splits"]["calibration"]) == 20
    assert len(result["splits"]["fit"]) == 79
