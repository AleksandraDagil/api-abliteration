import unittest

from safety_gap.api_evaluation.runner import _resume_config_is_compatible


class ResumeConfigTests(unittest.TestCase):
    def test_only_cost_ceiling_may_increase(self):
        previous = {
            "execution": {"max_total_estimated_cost_usd": 35, "workers": 4},
            "provider": {"model_id": "abliterated-model"},
        }
        increased = {
            "execution": {"max_total_estimated_cost_usd": 50, "workers": 4},
            "provider": {"model_id": "abliterated-model"},
        }
        decreased = {
            "execution": {"max_total_estimated_cost_usd": 20, "workers": 4},
            "provider": {"model_id": "abliterated-model"},
        }
        changed_model = {
            "execution": {"max_total_estimated_cost_usd": 50, "workers": 4},
            "provider": {"model_id": "different-model"},
        }

        self.assertTrue(_resume_config_is_compatible(previous, increased))
        self.assertFalse(_resume_config_is_compatible(previous, decreased))
        self.assertFalse(_resume_config_is_compatible(previous, changed_model))


if __name__ == "__main__":
    unittest.main()
