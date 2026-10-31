"""Parallel research via LangGraph's Send API."""
import json
import os
import re
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from app import config
from app.agents.research_agent import research_agent, research_worker
from app.errors import AgentError
from app.research.aggregation import merge_results, normalize_url
from app.research.planner import default_tasks, plan_research_tasks
from app.runner import AgentTracker, apply_update, build_initial_state, graph_config, stream_research
from app.schemas import Finding, ResearchTask, Source, WorkerResult
from app.state_validation import validate_state
from tests.fakes import APPROVED, FakeLLM, Fakes

DELAY = 0.25
PLAN4 = json.dumps({"tasks": [
    {"id": 1, "aspect": "Adoption", "query": "ai agents adoption"},
    {"id": 2, "aspect": "Technology", "query": "ai agents technology"},
    {"id": 3, "aspect": "Risks", "query": "ai agents risks"},
    {"id": 4, "aspect": "Opportunities", "query": "ai agents opportunities"},
]})
SHARED_URL = "https://www.shared.example/overview/"


def slug(query):
    return re.sub(r"[^a-z]+", "-", query.lower()).strip("-")


class SlowSearch:
    """Thread-safe fake search: slow, records concurrency, can fail on demand."""

    def __init__(self, delay=DELAY, fail_queries=(), transient_once=()):
        self.delay, self.fail_queries = delay, set(fail_queries)
        self.transient_once = set(transient_once)
        self.lock = threading.Lock()
        self.in_flight = self.max_in_flight = 0
        self.queries = []

    def __call__(self, query):
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.queries.append(query)
            flaky = query in self.transient_once
            self.transient_once.discard(query)
        try:
            time.sleep(self.delay)
            if query in self.fail_queries:
                raise RuntimeError("search backend down")
            if flaky:
                raise RuntimeError("Error code: 429 - rate_limit_exceeded")
            return {"results": [
                {"url": SHARED_URL, "title": "Shared overview", "content": "Overview.", "score": 0.9},
                {"url": f"https://site.example/{slug(query)}", "title": f"Page for {query}",
                 "content": f"Details about {query}.", "published_date": "2026-02-01"},
            ]}
        finally:
            with self.lock:
                self.in_flight -= 1


class AspectLLM:
    """Thread-safe research LLM: answers from the 'Research focus' in the prompt."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        aspect = re.search(r"Research focus: (.+)", prompt).group(1).strip()
        return SimpleNamespace(content=json.dumps({
            "summary": f"{aspect} summary.",
            "findings": [
                {"claim": "AI agents are widely adopted", "evidence": "Overview.",
                 "source_ids": [1], "confidence": "medium"},
                {"claim": f"Distinct finding about {aspect}", "evidence": f"{aspect} detail.",
                 "source_ids": [2], "confidence": "high"},
            ],
            "conflicts": [],
        }))


def run_graph(search, concurrency=4, plan=PLAN4, reviewer=None, research_llm=None, **state_kw):
    from app.workflows.graph import research_graph

    fakes = Fakes(reviewer or [APPROVED], search=search, planner_llm=FakeLLM([plan]),
                  research_llm=research_llm or AspectLLM())
    with fakes:
        started = time.perf_counter()
        state = build_initial_state("How are AI agents evolving?", **state_kw)
        result = research_graph.invoke(state, config=graph_config(concurrency))
        elapsed = time.perf_counter() - started
    return result, fakes, elapsed


class ConcurrencyTests(unittest.TestCase):
    def test_four_tasks_execute_concurrently(self):
        search = SlowSearch()
        result, fakes, elapsed = run_graph(search, concurrency=4)

        self.assertEqual(len(result["research_tasks"]), 4)
        self.assertEqual(len(result["worker_results"]), 4)
        self.assertEqual(search.max_in_flight, 4)  # all four searches overlapped
        self.assertEqual(sorted(search.queries), sorted(t.query for t in result["research_tasks"]))
        self.assertEqual(len(fakes.research.prompts), 4)  # one extraction per worker

    def test_concurrency_limit_is_respected(self):
        for limit in (1, 2):
            search = SlowSearch(delay=0.1)
            run_graph(search, concurrency=limit)
            self.assertEqual(search.max_in_flight, limit, f"limit={limit}")

    def test_parallel_is_faster_than_sequential(self):
        _, _, parallel = run_graph(SlowSearch(), concurrency=4)
        _, _, sequential = run_graph(SlowSearch(), concurrency=1)
        print(f"\n[timing] 4 tasks x {DELAY}s search: parallel={parallel:.2f}s "
              f"sequential={sequential:.2f}s speedup={sequential / parallel:.1f}x")
        self.assertGreater(sequential, 4 * DELAY)          # truly serial
        self.assertLess(parallel, sequential * 0.6)        # clearly faster


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, _, _ = run_graph(SlowSearch(delay=0.05))
        cls.data = cls.result["research_data"]

    def test_all_successful_results_are_aggregated(self):
        self.assertEqual([r.status for r in self.data.task_reports], ["ok"] * 4)
        claims = [f.claim for f in self.data.findings]
        for aspect in ("Adoption", "Technology", "Risks", "Opportunities"):
            self.assertIn(f"Distinct finding about {aspect}", claims)
        self.assertEqual(len(self.data.sources), 5)  # 1 shared + 4 unique

    def test_duplicate_urls_and_findings_removed(self):
        self.assertEqual(self.data.duplicate_sources_removed, 3)
        self.assertEqual(self.data.duplicate_findings_removed, 3)
        self.assertEqual(sum(f.claim == "AI agents are widely adopted" for f in self.data.findings), 1)
        urls = [normalize_url(s.url) for s in self.data.sources]
        self.assertEqual(len(urls), len(set(urls)))

    def test_citations_are_remapped_to_the_right_sources(self):
        by_id = self.data.source_map()
        for finding in self.data.findings:
            if finding.claim.startswith("Distinct finding about "):
                aspect = finding.claim.rsplit(" ", 1)[-1].lower()
                query_slug = {"adoption": "ai-agents-adoption", "technology": "ai-agents-technology",
                              "risks": "ai-agents-risks", "opportunities": "ai-agents-opportunities"}[aspect]
                self.assertIn(query_slug, by_id[finding.source_ids[0]].url)

    def test_shared_source_keeps_provenance_and_verification(self):
        shared = self.data.sources[0]
        self.assertEqual(shared.id, 1)  # found by every task -> ranked first
        self.assertEqual(shared.found_by, [1, 2, 3, 4])
        self.assertTrue(all(s.verified for s in self.data.sources))
        self.assertEqual(self.data.sources[1].published_date, "2026-02-01")

    def test_downstream_agents_receive_the_aggregate(self):
        self.assertIn("Adoption summary.", self.result["research"])
        self.assertEqual(self.result["analysis_data"].opportunities, ["Governance tooling"])
        self.assertTrue(self.result["approved"])
        self.assertEqual(validate_state(self.result), [])
        self.assertIn(SHARED_URL.rstrip("/"), self.result["draft"].replace("www.", "www.").rstrip("/") + SHARED_URL.rstrip("/"))


class WorkerFailureTests(unittest.TestCase):
    def test_one_failing_worker_keeps_the_other_results(self):
        search = SlowSearch(delay=0.05, fail_queries={"ai agents risks"})
        with self.assertLogs(level="INFO") as logs:  # worker + aggregation loggers
            result, _, _ = run_graph(search)

        reports = {r.aspect: r for r in result["research_data"].task_reports}
        self.assertEqual(reports["Risks"].status, "failed")
        self.assertIn("search backend down", reports["Risks"].error)
        self.assertEqual([a for a, r in reports.items() if r.status == "ok"],
                         ["Adoption", "Technology", "Opportunities"])
        claims = [f.claim for f in result["research_data"].findings]
        self.assertIn("Distinct finding about Technology", claims)
        self.assertNotIn("Distinct finding about Risks", claims)
        # downstream agents are told about the gap, and the run still completes
        self.assertIn("Coverage gaps", result["research"])
        self.assertIn("Risks", result["research"])
        self.assertTrue(result["approved"])
        self.assertEqual(validate_state(result), [])

        text = "\n".join(logs.output)
        self.assertIn("FAILED", text)
        self.assertRegex(text, r"Research worker \d \[Adoption\] finished in \d+\.\d+s: 2 source\(s\), 2 finding\(s\)")
        self.assertRegex(text, r"Research aggregate: 3/4 task\(s\) ok")

    def test_worker_returns_failed_result_instead_of_raising(self):
        task = ResearchTask(id=7, aspect="Risks", query="ai agents risks")
        with Fakes([APPROVED], search=SlowSearch(delay=0, fail_queries={task.query}), research_llm=AspectLLM()):
            update = research_worker({"task": task, "question": "q"})
        (result,) = update["worker_results"]
        self.assertEqual((result.status, result.task.id), ("failed", 7))
        self.assertIn("search backend down", result.error)
        self.assertEqual(update.keys(), {"worker_results"})  # no unreduced keys written in parallel

    def test_all_workers_failing_raises_a_clear_error(self):
        search = SlowSearch(delay=0, fail_queries={t["query"] for t in json.loads(PLAN4)["tasks"]})
        with self.assertRaisesRegex(AgentError, r"all 4 research task\(s\) failed"):
            run_graph(search)

    def test_transient_search_error_is_retried_inside_the_worker(self):
        search = SlowSearch(delay=0, transient_once={"ai agents risks"})
        result, _, _ = run_graph(search)
        self.assertEqual([r.status for r in result["research_data"].task_reports], ["ok"] * 4)
        self.assertEqual(search.queries.count("ai agents risks"), 2)

    def test_worker_with_invalid_llm_output_fails_alone(self):
        class OneBad(AspectLLM):
            def invoke(self, prompt):
                if "Research focus: Risks" in prompt:
                    self.prompts.append(prompt)
                    return SimpleNamespace(content="no json at all")
                return super().invoke(prompt)

        result, _, _ = run_graph(SlowSearch(delay=0), research_llm=OneBad())
        statuses = {r.aspect: r.status for r in result["research_data"].task_reports}
        self.assertEqual(statuses["Risks"], "failed")
        self.assertEqual(sum(s == "ok" for s in statuses.values()), 3)


def ok_result(task_id, aspect, sources, findings, **kw):
    srcs = [Source(id=i, title=t, url=u, **extra) for i, (t, u, extra) in enumerate(sources, 1)]
    return WorkerResult(
        task=ResearchTask(id=task_id, aspect=aspect, query=f"q{task_id}"), status="ok",
        sources=srcs, findings=[Finding(**f) for f in findings],
        summary=kw.pop("summary", f"{aspect} summary"), **kw)


class MergeResultsTests(unittest.TestCase):
    def test_url_normalization_dedupes_variants(self):
        variants = ["https://www.a.com/x/", "http://a.com/x", "https://a.com/x#frag",
                    "https://a.com/x?utm_source=news", "HTTPS://A.COM/x"]
        self.assertEqual(len({normalize_url(u) for u in variants}), 1)
        self.assertNotEqual(normalize_url("https://a.com/x?id=1"), normalize_url("https://a.com/x?id=2"))
        self.assertNotEqual(normalize_url("https://a.com/x"), normalize_url("https://a.com/y"))

    def test_merge_fills_missing_metadata_from_duplicates(self):
        r1 = ok_result(1, "A", [("T", "https://a.com/x", {})],
                       [dict(claim="Claim one", source_ids=[1])])
        r2 = ok_result(2, "B", [("T2", "https://www.a.com/x/?utm_x=1",
                                  {"publisher": "ACME", "published_date": "2026-01-01", "score": 0.5})],
                       [dict(claim="Claim two", source_ids=[1])])
        data = merge_results("q", [r1, r2])
        (src,) = data.sources
        self.assertEqual((src.publisher, src.published_date, src.found_by), ("ACME", "2026-01-01", [1, 2]))
        self.assertEqual(data.duplicate_sources_removed, 1)
        self.assertEqual([f.source_ids for f in data.findings], [[1], [1]])

    def test_near_duplicate_findings_merge_and_keep_best_data(self):
        r1 = ok_result(1, "A", [("T", "https://a.com/x", {})], [dict(
            claim="Enterprises are adopting multi-agent orchestration", evidence="short",
            source_ids=[1], confidence="low")])
        r2 = ok_result(2, "B", [("U", "https://b.com/y", {})], [dict(
            claim="Enterprises are adopting multi agent orchestration.", evidence="a much longer evidence text",
            source_ids=[1], confidence="high")])
        data = merge_results("q", [r1, r2])
        (finding,) = data.findings
        self.assertEqual(finding.confidence, "high")
        self.assertEqual(finding.evidence, "a much longer evidence text")
        self.assertEqual(finding.source_ids, [1, 2])
        self.assertEqual(data.duplicate_findings_removed, 1)

    def test_claims_with_different_numbers_are_not_merged(self):
        r1 = ok_result(1, "A", [("T", "https://a.com/x", {})], [
            dict(claim="Revenue from agents grew 10% in 2025", source_ids=[1]),
            dict(claim="Revenue from agents grew 12% in 2025", source_ids=[1]),
            dict(claim="Revenue from agents grew 10% in 2025.", source_ids=[1])])
        data = merge_results("q", [r1])
        self.assertEqual(len(data.findings), 2)
        self.assertEqual(data.duplicate_findings_removed, 1)

    def test_different_claims_are_not_merged(self):
        r1 = ok_result(1, "A", [("T", "https://a.com/x", {})], [
            dict(claim="Adoption is rising quickly", source_ids=[1]),
            dict(claim="Governance remains immature", source_ids=[1])])
        self.assertEqual(len(merge_results("q", [r1]).findings), 2)

    def test_source_cap_drops_sources_and_uncitable_findings(self):
        sources = [(f"T{i}", f"https://s{i}.com/", {}) for i in range(1, 5)]
        claims = ["Adoption is rising quickly", "Governance remains immature",
                  "Costs are falling for inference", "Interoperability standards are emerging"]
        findings = [dict(claim=c, source_ids=[i]) for i, c in enumerate(claims, 1)]
        data = merge_results("q", [ok_result(1, "A", sources, findings)], max_sources=2)
        self.assertEqual(len(data.sources), 2)
        self.assertEqual(len(data.findings), 2)  # claims 3 and 4 lost their only source
        self.assertTrue(all(i in data.source_map() for f in data.findings for i in f.source_ids))

    def test_result_does_not_depend_on_completion_order(self):
        a = ok_result(1, "A", [("T", "https://a.com/x", {})], [dict(claim="Claim from A", source_ids=[1])])
        b = ok_result(2, "B", [("U", "https://b.com/y", {})], [dict(claim="Claim from B", source_ids=[1])])
        self.assertEqual(merge_results("q", [a, b]).model_dump(), merge_results("q", [b, a]).model_dump())

    def test_failed_results_are_reported_not_hidden(self):
        good = ok_result(1, "A", [("T", "https://a.com/x", {})], [dict(claim="Claim from A", source_ids=[1])])
        bad = WorkerResult(task=ResearchTask(id=2, aspect="B", query="qb"), status="failed", error="boom")
        data = merge_results("q", [good, bad])
        self.assertEqual([(r.aspect, r.status) for r in data.task_reports], [("A", "ok"), ("B", "failed")])
        self.assertIn("Coverage gaps (research task failed, not covered): B", data.to_text())

    def test_all_failed_raises(self):
        bad = WorkerResult(task=ResearchTask(id=1, aspect="B", query="qb"), status="failed", error="boom")
        with self.assertRaisesRegex(AgentError, "all 1 research task"):
            merge_results("q", [bad])
        with self.assertRaises(AgentError):
            merge_results("q", [])


class WorkerResultSchemaTests(unittest.TestCase):
    def test_rules(self):
        task = ResearchTask(id=1, aspect="A", query="query")
        with self.assertRaisesRegex(ValidationError, "needs an error"):
            WorkerResult(task=task, status="failed")
        with self.assertRaisesRegex(ValidationError, "needs sources"):
            WorkerResult(task=task, status="ok")
        src = Source(id=1, title="T", url="https://a.com/")
        with self.assertRaisesRegex(ValidationError, "unknown source ids"):
            WorkerResult(task=task, status="ok", sources=[src],
                         findings=[Finding(claim="A claim", source_ids=[5])])
        with self.assertRaisesRegex(ValidationError, "cannot carry"):
            WorkerResult(task=task, status="failed", error="x", sources=[src])


class PlannerTests(unittest.TestCase):
    def plan(self, reply, max_tasks=4):
        llm = FakeLLM([reply])
        with patch("app.research.planner.llm", llm), patch("app.retry._sleep"):
            return plan_research_tasks("AI agents", max_tasks), llm

    def test_uses_llm_plan_and_reassigns_ids(self):
        tasks, _ = self.plan(PLAN4)
        self.assertEqual([t.aspect for t in tasks], ["Adoption", "Technology", "Risks", "Opportunities"])
        self.assertEqual([t.id for t in tasks], [1, 2, 3, 4])

    def test_truncates_to_max_tasks(self):
        tasks, _ = self.plan(PLAN4, max_tasks=2)
        self.assertEqual(len(tasks), 2)

    def test_falls_back_to_default_aspects_on_bad_plan(self):
        for bad in ("not json", json.dumps({"tasks": []}),
                    json.dumps({"tasks": [{"id": 1, "aspect": "A", "query": "same"},
                                          {"id": 2, "aspect": "B", "query": "same"}]})):
            tasks, _ = self.plan(bad, max_tasks=3)
            self.assertEqual(len(tasks), 3, bad)
            self.assertTrue(all("AI agents" in t.query for t in tasks))

    def test_single_task_needs_no_llm(self):
        tasks, llm = self.plan(PLAN4, max_tasks=1)
        self.assertEqual([t.query for t in tasks], ["AI agents"])
        self.assertEqual(llm.prompts, [])

    def test_default_tasks(self):
        self.assertEqual(len(default_tasks("topic", 99)), 8)
        self.assertEqual(len(default_tasks("topic", 0)), 1)

    def test_supervisor_plans_tasks_into_state(self):
        result, fakes, _ = run_graph(SlowSearch(delay=0), max_research_tasks=2)
        self.assertEqual(len(result["research_tasks"]), 2)
        self.assertEqual(len(result["worker_results"]), 2)
        self.assertEqual(len(fakes.planner.prompts), 1)
        self.assertIn("2 DISTINCT aspects", fakes.planner.prompts[0])


class ConfigTests(unittest.TestCase):
    def env(self, **values):
        return patch.dict(os.environ, values, clear=False)

    def test_defaults_and_overrides(self):
        with self.env(RESEARCH_MAX_TASKS="6", RESEARCH_MAX_CONCURRENCY="2"):
            self.assertEqual((config.max_research_tasks(), config.research_concurrency()), (6, 2))
            self.assertEqual(graph_config()["max_concurrency"], 2)
            self.assertEqual(build_initial_state("q")["max_research_tasks"], 6)
        self.assertEqual(graph_config(3)["max_concurrency"], 3)

    def test_invalid_values_are_clamped_or_ignored(self):
        with self.env(RESEARCH_MAX_TASKS="99", RESEARCH_MAX_CONCURRENCY="abc"):
            self.assertEqual(config.max_research_tasks(), 8)
            self.assertEqual(config.research_concurrency(), config.DEFAULT_RESEARCH_CONCURRENCY)
        with self.env(RESEARCH_MAX_TASKS="0"):
            self.assertEqual(config.max_research_tasks(), 1)


class RunnerIntegrationTests(unittest.TestCase):
    def test_apply_update_appends_worker_results_only(self):
        state = {"worker_results": ["a"], "draft": "old"}
        apply_update(state, {"worker_results": ["b"], "draft": "new"})
        self.assertEqual(state, {"worker_results": ["a", "b"], "draft": "new"})

    def test_stream_emits_one_event_per_worker_and_tracks_progress(self):
        tracker = AgentTracker()
        tracker.start()
        events, details = [], []
        with Fakes([APPROVED], search=SlowSearch(delay=0.02), planner_llm=FakeLLM([PLAN4]),
                   research_llm=AspectLLM()):
            for node, update, state in stream_research("How are AI agents evolving?"):
                tracker.on_event(node, update, state)
                events.append(node)
                if node == "research_worker":
                    details.append(tracker.status["research"])
        self.assertEqual(events.count("research_worker"), 4)
        self.assertEqual(len(state["worker_results"]), 4)
        self.assertTrue(all(status == "running" for status, _ in details))
        self.assertEqual(details[-1][1], "Research tasks finished: 4/4")
        self.assertEqual(tracker.status["research"][0], "done")
        self.assertIn("From 4 task(s).", tracker.status["research"][1])

    def test_research_agent_without_workers_still_works(self):
        with Fakes([APPROVED]):
            update = research_agent(build_initial_state("How are AI agents evolving?"))
        self.assertEqual(len(update["research_data"].task_reports), 1)


class StateValidationTests(unittest.TestCase):
    def test_worker_state_rules(self):
        task = ResearchTask(id=1, aspect="A", query="query")
        src = Source(id=1, title="T", url="https://a.com/")
        good = WorkerResult(task=task, status="ok", sources=[src],
                            findings=[Finding(claim="A claim", source_ids=[1])])
        base = {"research_tasks": [task]}
        self.assertEqual(validate_state({**base, "worker_results": [good]}), [])
        dup = validate_state({**base, "worker_results": [good, good]})
        self.assertTrue(any("two results for the same task" in p for p in dup))
        stray = good.model_copy(update={"task": ResearchTask(id=9, aspect="X", query="other")})
        self.assertTrue(any("never planned" in p for p in
                            validate_state({**base, "worker_results": [stray]})))
        self.assertTrue(any("not a valid WorkerResult" in p for p in
                            validate_state({"worker_results": [{"task": 1}]})))


if __name__ == "__main__":
    unittest.main()


class UnknownNodeTests(unittest.TestCase):
    def test_tracker_ignores_nodes_it_does_not_know(self):
        tracker = AgentTracker()
        tracker.start()
        before = dict(tracker.status)
        tracker.on_event("some_future_node", {}, {})
        self.assertEqual(tracker.status, before)
