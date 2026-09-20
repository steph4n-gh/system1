"""Check the saved numeric skill and the boundary between advice and execution."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.gaming.teach_paperclips_wire import main, suggest, telemetry


def test_teach_save_reload_and_retain_boundary_errors(tmp_path):
    report = main(["--output-dir", str(tmp_path)])
    assert (tmp_path / "wire.s1m").is_file()
    assert report["counts"] == {"teach": 24, "calibration": 48, "evaluate": 16}
    assert len(report["results"]) == 16
    # Exercise useful decisions at both ends; do not turn development accuracy
    # into a passing autonomy threshold or hide errors near the preference edge.
    assert report["results"][0]["action"] == "buy_wire"
    assert report["results"][-1]["action"] == "hold"
    assert report["correct"] == sum(row["prediction"] == row["label"] for row in report["results"])


@pytest.mark.parametrize("review,funds,expected", [(False, 5, "hold"),
                                                   (True, 100, "review"),
                                                   (False, 20, "buy_wire")])
def test_unaffordable_and_ambiguous_purchases_are_not_executable(review, funds, expected):
    engine = SimpleNamespace(decide=Mock(return_value=SimpleNamespace(
        values={"action": "buy_wire"}, is_ambiguous=review,
        conformal_sets={"action": ["buy_wire", "hold"] if review else ["buy_wire"]})))
    assert suggest(engine, wire=10, funds=funds, wire_cost=20)["action"] == expected


@pytest.mark.parametrize("wire", [-1, float("nan"), float("inf")])
def test_invalid_live_stock_is_not_an_actionable_state(wire):
    with pytest.raises(ValueError, match="finite nonnegative"):
        telemetry(wire)
