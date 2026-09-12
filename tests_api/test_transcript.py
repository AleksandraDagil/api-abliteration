import csv
import tempfile
import unittest
from pathlib import Path

from safety_gap.api_evaluation.transcript import TranscriptWriter


class TranscriptTests(unittest.TestCase):
    def test_multiline_conversation_round_trips_through_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "transcript.csv"
            with TranscriptWriter(path) as writer:
                writer.write(
                    {
                        "run_id": "run",
                        "task_type": "propensity",
                        "question": "line one\nline two",
                        "target_answer": "full\nanswer",
                        "judge_answer": "1.b 0\n2.b 5\n3.b 5",
                    }
                )

            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["question"], "line one\nline two")
            self.assertEqual(rows[0]["target_answer"], "full\nanswer")
            self.assertEqual(rows[0]["judge_answer"], "1.b 0\n2.b 5\n3.b 5")

    def test_append_preserves_existing_rows_without_a_second_header(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "transcript.csv"
            with TranscriptWriter(path) as writer:
                writer.write({"run_id": "run", "sample_id": "first"})
            with TranscriptWriter(path, append=True) as writer:
                writer.write({"run_id": "run", "sample_id": "second"})

            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["sample_id"] for row in rows], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
