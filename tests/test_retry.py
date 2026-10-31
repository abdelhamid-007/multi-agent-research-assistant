import unittest
from unittest.mock import patch

from app.retry import invoke_llm, is_transient, retry_call


class RateLimit(Exception):
    status_code = 429


class BadKey(Exception):
    status_code = 401


class Flaky:
    def __init__(self, failures, exc=RateLimit):
        self.failures, self.exc, self.calls = failures, exc, 0

    def __call__(self, *a):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("Error code: 429 - Please try again in 1.5s")
        return "ok"


class RetryTests(unittest.TestCase):
    def test_classification(self):
        self.assertTrue(is_transient(RateLimit()))
        self.assertTrue(is_transient(RuntimeError("Error code: 413 - tokens per minute")))
        self.assertTrue(is_transient(TimeoutError("request timed out")))
        self.assertFalse(is_transient(BadKey()))
        self.assertFalse(is_transient(KeyError("x")))

    def test_recovers_after_transient_failures(self):
        slept = []
        fn = Flaky(2)
        self.assertEqual(retry_call(fn, sleep=slept.append), "ok")
        self.assertEqual(fn.calls, 3)
        self.assertEqual(len(slept), 2)
        self.assertGreaterEqual(slept[0], 2.0)   # backoff base
        self.assertGreater(slept[1], slept[0] - 1)  # grows

    def test_gives_up_and_raises_original_error(self):
        fn = Flaky(99)
        with self.assertRaises(RateLimit):
            retry_call(fn, attempts=3, sleep=lambda s: None)
        self.assertEqual(fn.calls, 3)

    def test_non_transient_not_retried(self):
        fn = Flaky(99, exc=BadKey)
        with self.assertRaises(BadKey):
            retry_call(fn, sleep=lambda s: None)
        self.assertEqual(fn.calls, 1)

    def test_invoke_llm_uses_retry(self):
        class LLM:
            calls = 0

            def invoke(self, prompt):
                LLM.calls += 1
                if LLM.calls == 1:
                    raise RateLimit("429")
                return prompt.upper()

        with patch("app.retry._sleep"):
            self.assertEqual(invoke_llm(LLM(), "hi"), "HI")
        self.assertEqual(LLM.calls, 2)


if __name__ == "__main__":
    unittest.main()


class QuotaTests(unittest.TestCase):
    def test_long_suggested_wait_fails_fast(self):
        calls = []

        def daily_quota():
            calls.append(1)
            raise RateLimit("Error code: 429 - tokens per day (TPD): Please try again in 20m42.8s.")

        slept = []
        with self.assertRaises(RateLimit):
            retry_call(daily_quota, sleep=slept.append)
        self.assertEqual((len(calls), slept), (1, []))

    def test_short_suggested_wait_still_retries(self):
        fn = Flaky(1)  # "try again in 1.5s"
        self.assertEqual(retry_call(fn, sleep=lambda s: None), "ok")
        self.assertEqual(fn.calls, 2)

    def test_daily_quota_message(self):
        from app.runner import friendly_error
        msg = friendly_error(RuntimeError(
            "Error code: 429 - on tokens per day (TPD): Limit 200000. Please try again in 20m42.8s."))
        self.assertIn("daily token quota", msg)
        self.assertIn("20m42.8s", msg)
