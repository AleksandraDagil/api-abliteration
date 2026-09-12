import unittest

from safety_gap.api_evaluation.metrics import compute_summary


class MetricsTests(unittest.TestCase):
    def test_summary_separates_target_and_judge_costs(self):
        rows = [
            {
                "task_type": "capability",
                "target_success": True,
                "target_latency_seconds": 1.0,
                "target_estimated_cost_usd": 0.10,
                "target_prompt_tokens": 100,
                "target_cached_input_tokens": 20,
                "parsed_answer_letter": "A",
                "capability_correct": True,
            },
            {
                "task_type": "capability",
                "target_success": True,
                "target_latency_seconds": 3.0,
                "target_estimated_cost_usd": 0.10,
                "target_prompt_tokens": 100,
                "target_cached_input_tokens": 0,
                "parsed_answer_letter": "B",
                "capability_correct": False,
            },
            {
                "task_type": "propensity",
                "target_success": True,
                "target_latency_seconds": 2.0,
                "target_estimated_cost_usd": 0.20,
                "target_prompt_tokens": 100,
                "target_cached_input_tokens": 30,
                "judge_success": True,
                "judge_latency_seconds": 4.0,
                "judge_estimated_cost_usd": 0.05,
                "strongreject_refusal": 0,
                "strongreject_score": 0.75,
                "strongreject_useful_at_threshold": True,
            },
            {
                "task_type": "propensity",
                "target_success": True,
                "target_latency_seconds": 4.0,
                "target_estimated_cost_usd": 0.20,
                "target_prompt_tokens": 100,
                "target_cached_input_tokens": 10,
                "judge_success": True,
                "judge_latency_seconds": 6.0,
                "judge_estimated_cost_usd": 0.05,
                "strongreject_refusal": 1,
                "strongreject_score": 0.0,
                "strongreject_useful_at_threshold": False,
            },
        ]
        summary = compute_summary(rows)

        self.assertEqual(summary["capability_accuracy"], 0.5)
        self.assertEqual(summary["binary_compliance_rate"], 0.5)
        self.assertEqual(summary["mean_strongreject_score"], 0.375)
        self.assertEqual(summary["edc_binary_proxy"], 0.25)
        self.assertEqual(summary["edc_continuous_proxy"], 0.1875)
        self.assertAlmostEqual(summary["total_target_estimated_cost_usd"], 0.60)
        self.assertAlmostEqual(summary["total_judge_estimated_cost_usd"], 0.10)
        self.assertAlmostEqual(
            summary["target_cost_per_successful_api_response_usd"], 0.15
        )
        self.assertAlmostEqual(
            summary["judge_cost_per_successful_api_response_usd"], 0.05
        )
        self.assertAlmostEqual(
            summary["target_cost_per_correct_capability_answer_usd"], 0.20
        )
        self.assertAlmostEqual(summary["target_cost_per_non_refusal_usd"], 0.40)
        self.assertAlmostEqual(summary["target_cost_per_useful_answer_usd"], 0.40)
        self.assertAlmostEqual(summary["target_cached_input_token_rate"], 0.15)

    def test_cost_falls_back_to_recalculated_value_per_row(self):
        rows = [
            {
                "task_type": "capability",
                "target_success": True,
                "target_estimated_cost_usd": None,
                "target_recalculated_cost_usd": 0.25,
                "parsed_answer_letter": "A",
                "capability_correct": True,
            },
            {
                "task_type": "capability",
                "target_success": True,
                "target_estimated_cost_usd": 0.10,
                "target_recalculated_cost_usd": 9.99,
                "parsed_answer_letter": "A",
                "capability_correct": True,
            },
        ]

        summary = compute_summary(rows)

        self.assertAlmostEqual(summary["total_target_estimated_cost_usd"], 0.35)


if __name__ == "__main__":
    unittest.main()
