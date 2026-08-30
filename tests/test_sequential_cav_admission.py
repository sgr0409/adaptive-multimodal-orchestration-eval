import unittest

from framework.sequential_cav_admission import SequentialCAVAdmission


class SequentialCAVAdmissionTests(unittest.TestCase):
    def test_reference_is_exact_until_crossing_and_selection_freezes(self):
        guard = SequentialCAVAdmission(["a", "b"], alternative_grid=[0.75])
        for _ in range(10):
            stats = guard.update({"a": 1, "b": 0})
        self.assertEqual(stats.selected_path, "a")
        admitted = stats.admission_time
        for _ in range(10):
            stats = guard.update({"a": -1, "b": 1})
        self.assertEqual(stats.selected_path, "a")
        self.assertEqual(stats.admission_time, admitted)

    def test_neutral_updates_do_not_change_e_values(self):
        guard = SequentialCAVAdmission(["a", "b"])
        before = guard.stats().log_e_values
        after = guard.update({"a": 0, "b": 0}).log_e_values
        self.assertEqual(before, after)

    def test_harms_prevent_admission(self):
        guard = SequentialCAVAdmission(["a"])
        for i in range(100):
            guard.update({"a": 1 if i % 2 == 0 else -1})
        self.assertEqual(guard.stats().selected_path, "equal_weight")

    def test_invalid_updates_are_rejected(self):
        guard = SequentialCAVAdmission(["a", "b"])
        with self.assertRaises(ValueError):
            guard.update({"a": 1})
        with self.assertRaises(ValueError):
            guard.update({"a": 2, "b": 0})


if __name__ == "__main__":
    unittest.main()
