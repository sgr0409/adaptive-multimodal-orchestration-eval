import unittest

from framework.cav_portfolio_guard import CAVPortfolioGuardFusion


def result(label):
    return {"label": label, "probs": {"a": 0.5, "b": 0.5}}


class CAVPortfolioGuardFusionTest(unittest.TestCase):
    def test_selects_supported_candidate_and_predicts_it(self):
        labels = ["b"] * 12
        reference = [result("a") for _ in labels]
        strong = [result("b") for _ in labels]
        weak = [result("a") for _ in labels]
        guard = CAVPortfolioGuardFusion().fit(
            reference,
            {"strong": strong, "weak": weak},
            labels,
        )
        self.assertEqual(guard.stats_.selected_path, "strong")
        self.assertTrue(
            guard.stats_.candidate_evidence["strong"]["admissible"]
        )
        evidence = guard.stats_.candidate_evidence["strong"]
        self.assertGreater(evidence["lower_decisive_probability"], 0.0)
        self.assertGreater(evidence["lower_benefit_probability"], 0.5)
        self.assertGreater(evidence["lower_accuracy_gain"], 0.0)
        self.assertAlmostEqual(
            evidence["certified_total_variation_radius"],
            evidence["lower_accuracy_gain"] / 2.0,
        )
        predicted = guard.predict(result("a"), {
            "strong": result("b"), "weak": result("a")
        })
        self.assertEqual(predicted["label"], "b")
        self.assertEqual(predicted["selected_path"], "strong")

    def test_falls_back_when_candidate_is_harmful(self):
        labels = ["a"] * 12
        reference = [result("a") for _ in labels]
        harmful = [result("b") for _ in labels]
        guard = CAVPortfolioGuardFusion().fit(
            reference, {"harmful": harmful}, labels
        )
        self.assertEqual(guard.stats_.selected_path, "equal_weight")
        predicted = guard.predict(result("a"), {"harmful": result("b")})
        self.assertEqual(predicted["label"], "a")

    def test_requires_minimum_decisive_evidence(self):
        labels = ["b"] * 4 + ["a"] * 8
        reference = [result("a") for _ in labels]
        sparse = [result("b") if index < 4 else result("a")
                  for index in range(len(labels))]
        guard = CAVPortfolioGuardFusion().fit(
            reference, {"sparse": sparse}, labels
        )
        evidence = guard.stats_.candidate_evidence["sparse"]
        self.assertEqual(evidence["decisive_switches"], 4)
        self.assertFalse(evidence["admissible"])
        self.assertEqual(guard.stats_.selected_path, "equal_weight")

    def test_gain_bound_includes_decisive_coverage_uncertainty(self):
        labels = ["b"] * 12 + ["a"] * 18
        reference = [result("a") for _ in labels]
        sparse_but_supported = [
            result("b") if index < 12 else result("a")
            for index in range(len(labels))
        ]
        guard = CAVPortfolioGuardFusion().fit(
            reference, {"candidate": sparse_but_supported}, labels
        )
        evidence = guard.stats_.candidate_evidence["candidate"]
        empirical_decisive_rate = evidence["decisive_switches"] / len(labels)
        self.assertLess(
            evidence["lower_decisive_probability"], empirical_decisive_rate
        )
        self.assertGreater(evidence["lower_accuracy_gain"], 0.0)
        self.assertAlmostEqual(
            guard.stats_.selected_total_variation_radius,
            evidence["lower_accuracy_gain"] / 2.0,
        )

    def test_rejects_invalid_inputs_and_missing_selected_candidate(self):
        with self.assertRaises(ValueError):
            CAVPortfolioGuardFusion(familywise_error_rate=0.0)
        guard = CAVPortfolioGuardFusion()
        with self.assertRaises(ValueError):
            guard.fit([result("a")], {}, ["a"])
        labels = ["b"] * 12
        guard.fit(
            [result("a") for _ in labels],
            {"strong": [result("b") for _ in labels]},
            labels,
        )
        with self.assertRaises(ValueError):
            guard.predict(result("a"), {})


if __name__ == "__main__":
    unittest.main()
