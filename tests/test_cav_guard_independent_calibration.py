import unittest

from experiments.cav_guard_independent_calibration import (
    stratified_three_way_split,
)


class IndependentCalibrationSplitTest(unittest.TestCase):
    def test_split_is_disjoint_complete_stratified_and_deterministic(self):
        labels = [label for label in ("a", "b", "c") for _ in range(100)]
        first = stratified_three_way_split(labels, seed=17)
        second = stratified_three_way_split(labels, seed=17)
        train, calibration, test = first

        self.assertEqual((len(train), len(calibration), len(test)), (120, 90, 90))
        self.assertEqual(set(train) & set(calibration), set())
        self.assertEqual(set(train) & set(test), set())
        self.assertEqual(set(calibration) & set(test), set())
        self.assertEqual(set(train) | set(calibration) | set(test), set(range(300)))
        for indices, expected_per_class in ((train, 40), (calibration, 30), (test, 30)):
            counts = {label: 0 for label in ("a", "b", "c")}
            for index in indices:
                counts[labels[index]] += 1
            self.assertEqual(set(counts.values()), {expected_per_class})
        for left, right in zip(first, second):
            self.assertEqual(left.tolist(), right.tolist())


if __name__ == "__main__":
    unittest.main()
