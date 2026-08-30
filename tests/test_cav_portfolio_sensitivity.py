import json
import unittest
from pathlib import Path

from experiments.cav_portfolio_evidence_sensitivity import select_path


ROOT = Path(__file__).resolve().parents[1]


class CAVPortfolioSensitivityReplayTest(unittest.TestCase):
    def test_reported_setting_reproduces_all_strict_selected_paths(self):
        artifact = json.loads((
            ROOT / "experiments" / "results" /
            "cav_portfolio_independent_calibration.json"
        ).read_text())
        checked = 0
        for block in artifact["domains"].values():
            for record in block["sensitivity_partitions"]:
                self.assertEqual(
                    select_path(record, 0.05, 5),
                    record["portfolio_stats"]["selected_path"],
                )
                checked += 1
        self.assertEqual(checked, 120)


if __name__ == "__main__":
    unittest.main()
