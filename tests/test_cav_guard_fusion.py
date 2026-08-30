import unittest

import numpy as np

from framework.cav_guard_fusion import (
    CAVGuardFusion,
    HierarchicalCAVGuardFusion,
    cav_switch_features,
)
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion


def modalities(reference_label, candidate_label):
    """Construct two-modality predictions whose equal- and confidence-
    weighted paths take the requested labels."""
    if reference_label == candidate_label:
        return [
            {"modality": "a", "label": reference_label, "confidence": 0.9,
             "probs": {"a": 0.9, "b": 0.1} if reference_label == "a"
                      else {"a": 0.1, "b": 0.9}},
            {"modality": "b", "label": reference_label, "confidence": 0.8,
             "probs": {"a": 0.8, "b": 0.2} if reference_label == "a"
                      else {"a": 0.2, "b": 0.8}},
        ]
    # Two moderately confident a predictions outvote one b prediction under
    # equal weighting; power-three confidence weighting selects the highly
    # confident b prediction.
    return [
        {"modality": "a", "label": "a", "confidence": 0.80,
         "probs": {"a": 0.80, "b": 0.20}},
        {"modality": "b", "label": "a", "confidence": 0.80,
         "probs": {"a": 0.80, "b": 0.20}},
        {"modality": "c", "label": "b", "confidence": 0.95,
         "probs": {"a": 0.05, "b": 0.95}},
    ]


def switch_profile(candidate_confidence):
    """EW predicts a and power-three CW predicts b for confidence >= .92."""
    return [
        {"modality": "a", "label": "a", "confidence": 0.80,
         "probs": {"a": 0.80, "b": 0.20}},
        {"modality": "b", "label": "a", "confidence": 0.80,
         "probs": {"a": 0.80, "b": 0.20}},
        {"modality": "c", "label": "b", "confidence": candidate_confidence,
         "probs": {"a": 1.0 - candidate_confidence,
                   "b": candidate_confidence}},
    ]


class CAVGuardFusionTest(unittest.TestCase):
    def test_paths_differ_in_fixture(self):
        mods = modalities("a", "b")
        self.assertEqual(equal_weight_fusion(mods)["label"], "a")
        self.assertEqual(confidence_weighted_fusion(mods)["label"], "b")

    def test_enables_candidate_with_credible_positive_value(self):
        examples = [modalities("a", "b") for _ in range(12)]
        guard = CAVGuardFusion(credibility_threshold=0.95).fit(
            examples, ["b"] * 12
        )
        self.assertTrue(guard.stats_.candidate_enabled)
        self.assertEqual(guard.stats_.benefits, 12)
        self.assertEqual(guard.predict(examples[0])["label"], "b")

    def test_preserves_reference_when_candidate_is_harmful(self):
        examples = [modalities("a", "b") for _ in range(12)]
        guard = CAVGuardFusion(credibility_threshold=0.95).fit(
            examples, ["a"] * 12
        )
        self.assertFalse(guard.stats_.candidate_enabled)
        self.assertEqual(guard.stats_.harms, 12)
        self.assertEqual(guard.predict(examples[0])["label"], "a")

    def test_defaults_to_reference_when_evidence_is_insufficient(self):
        examples = [modalities("a", "b") for _ in range(4)]
        guard = CAVGuardFusion(credibility_threshold=0.95).fit(
            examples, ["b", "b", "a", "a"]
        )
        self.assertFalse(guard.stats_.candidate_enabled)
        self.assertAlmostEqual(guard.stats_.posterior_probability_positive, 0.5)

    def test_rejects_unfitted_prediction_and_bad_input(self):
        guard = CAVGuardFusion()
        with self.assertRaises(RuntimeError):
            guard.predict(modalities("a", "b"))
        with self.assertRaises(ValueError):
            guard.fit([modalities("a", "b")], [])


class HierarchicalCAVGuardFusionTest(unittest.TestCase):
    def test_switch_features_are_order_invariant_and_label_free(self):
        example = switch_profile(0.99)
        forward = cav_switch_features(example)
        reverse = cav_switch_features(list(reversed(example)))
        np.testing.assert_allclose(forward, reverse)
        self.assertEqual(forward.shape, (19,))

    def test_learns_instance_selective_candidate_region(self):
        benefits = [switch_profile(0.99) for _ in range(20)]
        harms = [switch_profile(0.92) for _ in range(6)]
        guard = HierarchicalCAVGuardFusion(seed=7).fit(
            benefits + harms,
            ["b"] * len(benefits) + ["a"] * len(harms),
        )
        self.assertTrue(guard.global_guard_.stats_.candidate_enabled)
        self.assertTrue(guard.stats_.local_model_fitted)
        self.assertGreater(
            guard.stats_.clopper_pearson_lower_benefit_probability, 0.5
        )
        self.assertEqual(guard.predict(switch_profile(0.99))["label"], "b")
        self.assertEqual(guard.predict(switch_profile(0.92))["label"], "a")

    def test_global_permission_blocks_harmful_candidate(self):
        examples = [switch_profile(0.99) for _ in range(16)]
        guard = HierarchicalCAVGuardFusion().fit(examples, ["a"] * 16)
        self.assertFalse(guard.global_guard_.stats_.candidate_enabled)
        self.assertFalse(guard.stats_.local_model_fitted)
        self.assertIsNone(guard.stats_.local_score_threshold)
        self.assertEqual(guard.predict(examples[0])["label"], "a")

    def test_independent_calibration_mode_records_exact_bound(self):
        train_benefits = [switch_profile(0.99) for _ in range(20)]
        train_harms = [switch_profile(0.92) for _ in range(6)]
        cal_benefits = [switch_profile(0.99) for _ in range(30)]
        cal_harms = [switch_profile(0.92) for _ in range(2)]
        guard = HierarchicalCAVGuardFusion(
            seed=7,
            local_score_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
            require_local_certification=True,
        ).fit(
            train_benefits + train_harms,
            ["b"] * len(train_benefits) + ["a"] * len(train_harms),
            cal_benefits + cal_harms,
            ["b"] * len(cal_benefits) + ["a"] * len(cal_harms),
        )
        self.assertEqual(guard.stats_.calibration_mode, "independent_exact")
        self.assertTrue(guard.stats_.local_model_fitted)
        self.assertEqual(guard.stats_.calibration_thresholds_tested, 5)
        self.assertGreater(
            guard.stats_.clopper_pearson_lower_benefit_probability, 0.5
        )

    def test_requires_fit_and_valid_calibration_arguments(self):
        guard = HierarchicalCAVGuardFusion()
        with self.assertRaises(RuntimeError):
            guard.predict(switch_profile(0.99))
        with self.assertRaises(ValueError):
            guard.fit([switch_profile(0.99)], ["b"], [switch_profile(0.99)])

    def test_strict_profile_preserves_reference_without_local_certificate(self):
        train_benefits = [switch_profile(0.99) for _ in range(20)]
        train_harms = [switch_profile(0.92) for _ in range(6)]
        calibration_harms = [switch_profile(0.99) for _ in range(12)]
        guard = HierarchicalCAVGuardFusion(
            seed=7,
            local_score_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
            require_local_certification=True,
        ).fit(
            train_benefits + train_harms,
            ["b"] * len(train_benefits) + ["a"] * len(train_harms),
            calibration_harms,
            ["a"] * len(calibration_harms),
        )
        self.assertTrue(guard.global_guard_.stats_.candidate_enabled)
        self.assertFalse(guard.stats_.local_model_fitted)
        self.assertIn("reference_fallback", guard.stats_.calibration_mode)
        self.assertEqual(guard.predict(switch_profile(0.99))["label"], "a")

    def test_rejects_invalid_fixed_local_threshold_grid(self):
        with self.assertRaises(ValueError):
            HierarchicalCAVGuardFusion(local_score_thresholds=())
        with self.assertRaises(ValueError):
            HierarchicalCAVGuardFusion(local_score_thresholds=(0.5, 1.1))


if __name__ == "__main__":
    unittest.main()
