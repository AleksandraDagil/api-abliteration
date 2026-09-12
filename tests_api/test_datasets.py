import tempfile
import unittest
from pathlib import Path

from safety_gap.api_evaluation.config import PropensityDatasetConfig
from safety_gap.api_evaluation.datasets import load_propensity_samples


class DatasetTests(unittest.TestCase):
    def test_limited_pilot_selects_first_eligible_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "propensity.csv"
            path.write_text(
                "questions,pr_unsafe\n"
                "excluded,0.5\n"
                "first,0.9\n"
                "second,0.8\n"
                "third,0.7\n",
                encoding="utf-8",
            )
            config = PropensityDatasetConfig(
                name="propensity_bio",
                path=path,
                question_column="questions",
                unsafe_probability_column="pr_unsafe",
                unsafe_probability_threshold=0.5,
                sample_limit=2,
            )

            samples = load_propensity_samples(config, seed=123)

        self.assertEqual([sample.question for sample in samples], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
