"""Offline checks for the build tooling: the cost guard, label validation, state
checks, one complete generate -> label -> freeze -> fold run against a fake API, and
the guards around it (a site with nothing to generate from, a call lost to the budget,
an unfrozen fold). Also the paired test's refusal of mismatched prediction files and
the data scrub. No network, no GPU.

    python scripts/test_generate.py
"""
import copy
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import fold  # noqa: E402
import generate  # noqa: E402
import llm  # noqa: E402
import check_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = "data-next"   # generate.example.json's working dataset directory
MODELS = llm.model_table(["provider-a/model-a", "provider-b/model-b"], [(1, 1), (2, 12)])


class ClientChecks(unittest.TestCase):
    def test_interrupted_request_remains_reserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LLM_OPENROUTER_API_KEYS": "test-only"}):
            client = llm.Client(Path(directory), MODELS, cap_usd=11)
            llm.append(client.events, dict(event="start", call_id="a", reservation=10.99))
            with patch("urllib.request.urlopen") as network:
                with self.assertRaisesRegex(RuntimeError, "Budget cap"):
                    client.call("blocked", "provider-b/model-b", "test", "test", 6000, .2)
                network.assert_not_called()
            llm.append(client.events, dict(event="end", call_id="a", cost=.2))
            self.assertAlmostEqual(client.spent(), .2)

    def test_custom_provider_uses_its_endpoint_and_key_name(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MY_PROVIDER_KEY": "test-only"}):
            protocol = {"build": "custom", "budget_usd": 1,
                        "providers": {"custom": {"endpoint": "https://api.example.com/v1/chat/completions",
                                                 "api_key_env": "MY_PROVIDER_KEY"}},
                        "models": [{"id": "model-id", "provider": "custom", "price_per_million": [1, 2]}]}
            with patch.object(generate, "ROOT", Path(directory)):
                generate.build_dir("custom").mkdir(parents=True)
                client = generate.client_for(protocol)
            self.assertEqual(client.models["model-id"]["endpoint"], "https://api.example.com/v1/chat/completions")
            self.assertFalse(client.models["model-id"]["openrouter"])
            reply = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                     "usage": {"cost": 0.0001}}
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as network:
                self.assertEqual(client.call("one", "model-id", "system", "user", 20, .2, reasoning="medium"), "{}")
            request = network.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.example.com/v1/chat/completions")
            payload = json.loads(request.data)
            self.assertNotIn("provider", payload)
            self.assertNotIn("reasoning", payload)


class ValidationChecks(unittest.TestCase):
    def test_invalid_probability_vectors_are_not_repaired(self):
        rows = [{"questions": {"yes": {"type": "noul"}}}]
        for p in ({"true": .9, "false": .9}, {"true": float("nan"), "false": .5}, {"true": -.1, "false": 1.1},
                  {"true": True, "false": 0}, {"true": 1}):
            with self.assertRaises(ValueError):
                llm.parse_labels(json.dumps({"labels": [{"yes": p}]}), rows)
        accepted = llm.parse_labels('{"labels":[{"yes":{"true":0.7,"false":0.3}}]}', rows)
        self.assertEqual(accepted[0]["yes"]["true"], .7)

    def test_missing_content_is_a_format_failure(self):
        with self.assertRaisesRegex(ValueError, "Missing response content"):
            llm.parse_json(None)

    def test_host_rewrite_preserves_mismatch(self):
        value = {"request": "a@one.com", "tool": "a@two.com"}
        safe = llm.reserve_hosts(value)
        self.assertNotEqual(safe["request"], safe["tool"])
        self.assertEqual(llm.reserve_hosts(safe), safe)

    def test_choice_order_is_preserved(self):
        q = {"type": "choice", "criteria": {"z": "last", "a": "first"}}
        self.assertEqual(llm.keys(q), ["z", "a"])

    def test_list_form_choice_names_its_options(self):
        import evaluate
        import gliner2_transfer

        q = {"type": "choice", "instructions": "Which one?", "criteria": ["z", "a"]}
        self.assertEqual(llm.keys(q), ["z", "a"])
        self.assertEqual(evaluate.option_keys(q), ["z", "a"])
        self.assertEqual(check_data.question_problems("q", q), [])
        row = {"questions": {"q": q}, "gold": {"q": {"probabilities": {"z": .25, "a": .75}}}}
        task = gliner2_transfer.build_tasks(row)[0]
        self.assertEqual((task["labels"], task["label_descriptions"]), (["z", "a"], {}))
        self.assertEqual(task["label_probs"], {"z": .25, "a": .75})

    def test_malformed_questions_are_refused(self):
        for q in ({"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
                  {"type": "choice", "instructions": "x", "criteria": ["a"]},
                  {"type": "choice", "instructions": "x", "criteria": "a, b"},
                  {"type": "score", "instructions": "x", "criteria": {"0": "low", "1": "high"}},
                  {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}},
                  {"type": "noul", "instructions": {"text": "x"}},
                  {"type": "rank", "instructions": "x"}):
            self.assertTrue(check_data.question_problems("q", q), q)
        sites = json.loads((ROOT / "data/sites.json").read_text(encoding="utf-8"))
        self.assertEqual(check_data.site_problems(sites), [])

    def test_host_rewrite_matches_the_publishing_scan(self):
        for text in ("call api.service.co.uk", "mail ops@acme.de", "see Shop.Acme.IO!", "...acme.com."):
            safe = llm.reserve_hosts(text)
            self.assertEqual(check_data.scan_text(safe), [], safe)
        self.assertEqual(llm.reserve_hosts("call api.service.co.uk"), "call api-service-co-uk.example")
        self.assertEqual(llm.reserve_hosts("https://docs.example.com/a"), "https://docs.example.com/a")

    def test_neighbourhood_accepts_structured_facts(self):
        reference = {"signal": {}, "neighbourhood": [], "user_state": "", "quiet_hours": False}
        state = {**reference, "neighbourhood": [{"source": "calendar", "summary": "In a meeting"}]}
        self.assertTrue(llm.valid_state(state, reference))
        self.assertFalse(llm.valid_state({**state, "quiet_hours": "false"}, reference))


class BuildChecks(unittest.TestCase):
    """One small build, end to end, with the API replaced by a deterministic fake."""

    def test_generate_label_freeze_fold(self):
        rng = random.Random(0)
        words = "amber birch cobalt dune ember fjord garnet harbor iris juniper kestrel lagoon".split()
        references = {}
        for r in data.load_split(ROOT / "data", "train"):
            references.setdefault(r["site"], r["state"])

        def perturb(v):
            if isinstance(v, str):
                return " ".join(rng.sample(words, 6)) + f" {rng.random()}"
            if isinstance(v, dict):
                return {k: perturb(x) for k, x in v.items()}
            return [perturb(x) for x in v] if isinstance(v, list) else v

        def fake_call(self, tag, model, system, user, max_tokens, temperature, reasoning=None):
            request = json.loads(user.split("\n")[0])
            if "count" in request:
                return json.dumps({"states": [perturb(copy.deepcopy(references[request["site"]]))
                                              for _ in range(request["count"])]})
            labels = []
            for _ in request["states"]:
                answer = {}
                for qid, q in request["questions"].items():
                    w = [rng.random() for _ in llm.keys(q)]
                    answer[qid] = {k: x / sum(w) for k, x in zip(llm.keys(q), w)}
                labels.append(answer)
            return json.dumps({"labels": labels})

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LLM_OPENROUTER_API_KEYS": "test-only"}), \
                patch.object(llm.Client, "call", fake_call), patch.object(generate, "ROOT", Path(directory)):
            root = Path(directory)
            shutil.copytree(ROOT / "data", root / WORK, ignore=shutil.ignore_patterns("provenance"))
            config = json.loads((ROOT / "scripts/generate.example.json").read_text(encoding="utf-8"))
            config.update(build="unittest", split="validation", sites=["tool.risk", "voice.endpoint"], cases_per_batch=2)
            config["models"] = [{"id": f"provider-{i}/model", "price_per_million": [1, 1]}
                                for i in range(4)]
            config["annotators_per_case"] = 2
            config["allow_same_family_labels"] = False
            (root / "build.json").write_text(json.dumps(config), encoding="utf-8")
            before = len(data.load_split(root / WORK, "validation"))
            generate.initialise(root / "build.json")
            protocol = generate.load_protocol("unittest")
            llm.OUT = generate.build_dir("unittest")
            client = generate.client_for(protocol)
            generate.generate(client, protocol)
            generate.label(client, protocol)
            generate.freeze(protocol)
            generate.fold_build(protocol)

            rows = data.load_split(root / WORK, "validation")
            new = [r for r in rows if data.build_of(r["case_id"]) == "unittest"]
            self.assertEqual(len(rows), before + 16)               # 2 sites x 4 generators x 2 cases
            self.assertEqual(len(new), 16)
            self.assertEqual(set(new[0]), {"case_id", "site", "workflow", "state", "questions", "gold"})
            gens = {r["case_id"]: r["generator"] for r in json.loads((llm.OUT / "candidates.json").read_text(encoding="utf-8"))}
            for a in json.loads((llm.OUT / "annotations.json").read_text(encoding="utf-8")):
                self.assertNotEqual(generate.family(a["annotator"]), generate.family(gens[a["case_id"]]))
            with self.assertRaises(SystemExit):
                generate.fold_build(protocol)                      # a build folds once

            # fold.py run directly: a frozen build passes for its own split only, and
            # cases with no freeze record are refused.
            cases = llm.OUT / "cases.jsonl"
            fold.check_frozen(cases, "validation")
            with self.assertRaisesRegex(SystemExit, "frozen for"):
                fold.check_frozen(cases, "train")
            loose = root / "loose.jsonl"
            loose.write_text(cases.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "no freeze.json"):
                fold.check_frozen(loose, "validation")

    def test_init_refuses_a_site_with_no_reference_case(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(generate, "ROOT", Path(directory)):
            root = Path(directory)
            shutil.copytree(ROOT / "data", root / WORK, ignore=shutil.ignore_patterns("provenance"))
            sites = json.loads((root / WORK / "sites.json").read_text(encoding="utf-8"))
            prompts = json.loads((root / WORK / "prompts.json").read_text(encoding="utf-8"))
            new = dict(sites[0], site="new.site")
            prompts["per_site"]["new.site"] = prompts["per_site"][sites[0]["site"]]
            (root / WORK / "sites.json").write_text(json.dumps(sites + [new]), encoding="utf-8")
            (root / WORK / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
            config = json.loads((ROOT / "scripts/generate.example.json").read_text(encoding="utf-8"))
            config.update(build="unittest", sites=["new.site"])
            (root / "build.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "no existing case"):
                generate.initialise(root / "build.json")       # before any call is paid for

    def test_a_skipped_batch_is_counted(self):
        def refused(self, *args, **kwargs):
            raise llm.BudgetLimit("Budget cap: no request sent")

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LLM_OPENROUTER_API_KEYS": "test-only"}), \
                patch.object(llm.Client, "call", refused), patch.object(generate, "ROOT", Path(directory)):
            root = Path(directory)
            shutil.copytree(ROOT / "data", root / WORK, ignore=shutil.ignore_patterns("provenance"))
            config = json.loads((ROOT / "scripts/generate.example.json").read_text(encoding="utf-8"))
            config.update(build="unittest", sites=["tool.risk"], cases_per_batch=2)
            (root / "build.json").write_text(json.dumps(config), encoding="utf-8")
            generate.initialise(root / "build.json")
            protocol = generate.load_protocol("unittest")
            llm.OUT = generate.build_dir("unittest")
            self.assertEqual(generate.generate(generate.client_for(protocol), protocol), 1)
            with self.assertRaisesRegex(SystemExit, "generate did not finish"):
                generate.freeze(protocol)

    def test_init_refuses_the_release_directory_and_used_prefixes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(generate, "ROOT", Path(directory)):
            root = Path(directory)
            shutil.copytree(ROOT / "data", root / "data", ignore=shutil.ignore_patterns("provenance"))
            shutil.copytree(ROOT / "data", root / WORK, ignore=shutil.ignore_patterns("provenance"))
            config = json.loads((ROOT / "scripts/generate.example.json").read_text(encoding="utf-8"))
            for change, message in (({"data_dir": "data"}, "must not be data/"),
                                    ({"data_dir": "missing"}, "missing"),
                                    ({"build": "exp"}, "already exist"),
                                    ({"build": "fill1"}, "already exist")):
                (root / "build.json").write_text(json.dumps({**config, **change}), encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, message):
                    generate.initialise(root / "build.json")
            self.assertFalse((root / "data" / "provenance").exists())    # refused before any record

    def test_fold_refuses_the_release_and_scrub_findings(self):
        with self.assertRaisesRegex(SystemExit, "v1 release"):
            fold.fold(ROOT / "unused.jsonl", "train", ROOT / "data", dry_run=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "data", root / WORK, ignore=shutil.ignore_patterns("provenance"))
            row = data.load_split(ROOT / "data", "train")[0]
            row = dict(row, case_id="scan-tool-0-0-0", state={"leak": "token sk-abcdefghijklmnopqrstuvwx"})
            source = root / "cases.jsonl"
            source.write_text(fold.dump(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "scrub"):
                fold.fold(source, "train", root / WORK)

    def test_reference_state_ignores_a_misspelled_key(self):
        references = generate.reference_states(ROOT / "data")
        self.assertNotIn("transence", references["voice.endpoint"])

    def test_single_model_can_label_in_a_blind_second_call(self):
        row = data.load_split(ROOT / "data", "train")[0]
        with tempfile.TemporaryDirectory() as directory, patch.object(generate, "ROOT", Path(directory)):
            out = generate.build_dir("single")
            out.mkdir(parents=True)
            candidate = {k: row[k] for k in ("site", "state", "questions")}
            candidate.update(case_id="single-case-0", generator="z-ai/glm-5.3", cluster_id="single-batch")
            (out / "candidates.json").write_text(json.dumps([candidate]), encoding="utf-8")
            labels = {qid: gold["probabilities"] for qid, gold in row["gold"].items()}
            (out / "annotations.json").write_text(json.dumps([
                {"case_id": candidate["case_id"], "annotator": "z-ai/glm-5.3", "probabilities": labels}
            ]), encoding="utf-8")
            protocol = {"build": "single", "split": "train", "models": [{"id": "z-ai/glm-5.3"}],
                        "annotators_per_case": 1, "allow_same_family_labels": True}
            (out / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            llm.write_json(out / "generate.complete.json", {"sha256": {
                "candidates.json": generate.file_sha256(out / "candidates.json")}})
            llm.write_json(out / "label.complete.json", {"sha256": {
                name: generate.file_sha256(out / name) for name in ("candidates.json", "annotations.json")}})
            generate.freeze(protocol)
            self.assertEqual(len((out / "cases.jsonl").read_text().splitlines()), 1)


class ScoringChecks(unittest.TestCase):
    def test_training_batch_plan_uses_tail_and_can_replay_v1(self):
        plan = data.training_batch_plan(35, 4, 8)
        self.assertEqual(sum(stop - start for start, stop, _, _ in plan), 35)
        self.assertEqual(sum(step for _, _, _, step in plan), 2)
        self.assertAlmostEqual(sum(scale for _, _, scale, _ in plan[-1:]), 1.0)
        legacy = data.training_batch_plan(35, 4, 8, legacy_drop_tail=True)
        self.assertEqual(sum(step for _, _, _, step in legacy), 1)
        self.assertEqual(len(data.training_batch_plan(3, 4, 8, legacy_drop_tail=True)), 0)
        self.assertEqual(len(data.training_batch_plan(3, 4, 8)), 1)

    def test_paired_test_refuses_files_scored_on_different_data(self):
        import numpy as np
        import evaluate

        def decision(p, t):
            return {"p": np.array(p), "t": np.array(t)}
        a = {("c1", "q"): decision([.9, .1], [1., 0.]), ("c2", "q"): decision([.2, .8], [1., 0.])}
        b = {("c1", "q"): decision([.4, .6], [1., 0.]), ("c2", "q"): decision([.7, .3], [1., 0.])}
        result = evaluate.mcnemar(a, b)
        self.assertEqual((result["a_only_correct"], result["b_only_correct"]), (1, 1))
        b[("c2", "q")]["t"] = np.array([0., 1.])                # same key, another label
        with self.assertRaisesRegex(SystemExit, "different labels"):
            evaluate.mcnemar(a, b)
        b.pop(("c2", "q"))
        with self.assertRaisesRegex(SystemExit, "coverage differs"):
            evaluate.mcnemar(a, b)
        b[("c2", "q")] = decision([.7, .3], [1., 0.])
        a = evaluate.PredictionRows(a, header={"split": "test", "dataset_sha256": "one"})
        b = evaluate.PredictionRows(b, header={"split": "test", "dataset_sha256": "two"})
        with self.assertRaisesRegex(SystemExit, "dataset_sha256 differs"):
            evaluate.mcnemar(a, b)

    def test_inference_error_does_not_shrink_evaluation(self):
        import evaluate

        class FailingAgent:
            def system_one(self, state, questions):
                raise ValueError("option budget")

        row = data.load_split(ROOT / "data", "test")[0]
        with self.assertRaisesRegex(ValueError, "inference failed"):
            evaluate.score(FailingAgent(), [row])

    def test_jev_restart_reserves_interrupted_calls(self):
        import jev_decisions

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LLM_OPENROUTER_API_KEYS": "test-only"}):
            out = Path(directory)
            (out / "jev.calls.jsonl").write_text(
                json.dumps({"event": "start", "tag": "one", "reservation": 1.25}) + "\n")
            self.assertEqual(jev_decisions.Client(out, "jev", 2.0).spent, 1.25)


class DataScanChecks(unittest.TestCase):
    def test_scan(self):
        clean = "mail ana@studio.example or see https://github.com/x/y and 192.0.2.7; arXiv cs.AI; 10.5 km"
        self.assertEqual(check_data.scan_text(clean), [])
        kinds = {kind for kind, _ in check_data.scan_text(
            "write to ana@realcorp.com, fetch http://tracker.realcorp.io/p from 93.184.216.34, "
            "key sk-abcdefghijklmnopqrstuvwx, card 4012 8888 8888 1881")}
        self.assertEqual(kinds, {"email", "host", "ip", "key", "card"})


if __name__ == "__main__":
    unittest.main()
