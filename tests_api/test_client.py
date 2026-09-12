import unittest

from safety_gap.api_evaluation.client import AbliterationClient
from tests_api.helpers import generation_config, provider_config


class FakeResponse:
    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "model": "abliterated-model-large",
            "choices": [
                {
                    "message": {
                        "content": "B",
                        "reasoning": "private reasoning",
                        "annotations": [{"url": "https://example.test"}],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 15},
            },
            "estimated_cost_usd": 0.0004,
            "estimated_credits_used": 2,
            "remaining_credits": 98,
        }


class RetryableError(Exception):
    status_code = 429

    class Response:
        headers = {"Retry-After": "2.5"}

    response = Response()


class ClientTests(unittest.TestCase):
    def test_builds_capability_request_and_captures_custom_metrics(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return FakeResponse()

        times = iter([10.0, 12.5])
        client = AbliterationClient(
            provider_config(),
            api_key="test",
            create_completion=create,
            clock=lambda: next(times),
        )
        result = client.complete(
            [{"role": "user", "content": "test"}], generation_config()
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "B")
        self.assertEqual(result.reasoning_content, "private reasoning")
        self.assertEqual(result.cached_input_tokens, 40)
        self.assertEqual(result.reasoning_tokens, 15)
        self.assertEqual(result.estimated_cost_usd, 0.0004)
        self.assertAlmostEqual(result.recalculated_cost_usd, 0.00045)
        self.assertEqual(result.latency_seconds, 2.5)

        request = calls[0]
        self.assertEqual(request["model"], "abliterated-model-large")
        self.assertEqual(request["temperature"], 1.0)
        self.assertEqual(request["top_p"], 0.95)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(request["reasoning_effort"], "max")
        self.assertTrue(request["extra_body"]["include_reasoning"])
        self.assertEqual(
            request["extra_body"]["web_search_options"]["search_context_size"],
            "high",
        )

    def test_retries_429_and_records_retry_after(self):
        calls = 0
        delays = []

        def create(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RetryableError("rate limited")
            return FakeResponse()

        times = iter([0.0, 3.0])
        client = AbliterationClient(
            provider_config(),
            api_key="test",
            create_completion=create,
            sleep=delays.append,
            clock=lambda: next(times),
        )
        result = client.complete(
            [{"role": "user", "content": "test"}], generation_config()
        )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.retry_delays_seconds, [2.5])
        self.assertEqual(delays, [2.5])


if __name__ == "__main__":
    unittest.main()
