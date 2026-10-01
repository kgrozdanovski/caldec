"""Validate the public JSONL splits and scan their contents before publishing.

    python3 scripts/check_data.py
    python3 scripts/check_data.py --data data-next     # a working copy for a new build

Checks question specifications, row schema, duplicate ids, decoded JSON, label
vectors and the scrub rules. Exact state overlaps across splits are printed for review.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
FIELDS = {"case_id", "site", "workflow", "state", "questions", "gold"}
# Documented v1 overlap groups. New cross-split duplicates fail validation; the v1
# split files and their reported scores remain immutable.
KNOWN_CROSS_SPLIT_OVERLAPS = {
    frozenset({"4ff5abb68fb762e9", "f3c227fb28ba960f"}),
    frozenset({"65f00f40f5ecf144", "18f024444062de0b"}),
    frozenset({"756bd6eeb279a9f6", "a0310add1cea5dec", "c90909f4f7ffbda7",
               "c9c15259b3787de5", "03a01942d69c016d", "a9b9505a70958cdb"}),
}

# Names that cannot be registered: RFC 2606 / RFC 6761 reserved names, and suffixes that
# are not public top-level domains.
RESERVED = re.compile(r"(^|\.)(example|test|invalid|localhost|local|internal|lan|home|corp)$"
                      r"|(^|\.)example\.(com|org|net)$", re.I)
# Public sites a realistic page or tool result cites by name. Anything else on a public
# top-level domain is a finding.
PUBLIC = {
    "github.com", "raw.githubusercontent.com", "arxiv.org", "pypi.org", "stackoverflow.com",
    "news.ycombinator.com", "weather.gov", "api.weather.gov", "api.open-meteo.com",
    "geocoding-api.open-meteo.com", "gov.uk", "irs.gov", "fdic.gov", "service-public.fr",
    "www.frankfurt.de", "www.raileurope.com", "www.ecb.europa.eu", "zoom.us",
}
PUBLIC_IPS = {"1.1.1.1", "8.8.8.8", "0.0.0.0"}                  # public resolvers, the unspecified address
TLDS = ("com|org|net|io|ai|dev|co|app|edu|gov|uk|de|fr|eu|us|ru|cn|jp|in|br|au|ca|nl|es|it|ch|se|"
        "info|biz|me|xyz|cloud|tech|site|online|shop")
HOST = re.compile(rf"(?<![\w@.-])((?:[a-z0-9-]+\.)+(?:{TLDS}))(?![\w-])", re.I)
EMAIL = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
IP = re.compile(r"(?<![\d.])((?:\d{1,3}\.){3}\d{1,3})(?![\d.])")
PRIVATE_IP = re.compile(r"^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|169\.254\.|"
                        r"192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)")
KEYS = re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|sk-or-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|"
                  r"gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b")
CARD = re.compile(r"(?<![\w.])(?:\d[ -]?){13,19}(?![\w.])")      # not a run inside a hex or base64 string


def luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def scan_text(text: str) -> list:
    """Findings in one string, as (kind, value)."""
    found = []
    for domain in EMAIL.findall(text):
        if not RESERVED.search(domain) and re.search(rf"\.({TLDS})$", domain, re.I):
            found.append(("email", domain.lower()))
    for host in HOST.findall(text):
        host = host.lower()
        if RESERVED.search(host) or host in PUBLIC or any(host.endswith("." + p) for p in PUBLIC):
            continue
        if re.fullmatch(r"[a-z]{2}\.[a-z]{2}", host):             # "cs.ai": an arXiv category, not a host
            continue
        found.append(("host", host))
    for ip in IP.findall(text):
        if all(int(part) <= 255 for part in ip.split(".")) and not PRIVATE_IP.match(ip) and ip not in PUBLIC_IPS:
            found.append(("ip", ip))
    found += [("key", m[:12] + "…") for m in KEYS.findall(text)]
    for match in CARD.findall(text):
        digits = re.sub(r"\D", "", match)
        if len(set(digits)) > 2 and luhn(digits):
            found.append(("card", digits[:6] + "…"))
    return found


def scan_row(state, gold=None) -> list:
    """Findings in a row's state and targets. Strings are decoded first, so escaped
    content cannot hide a match."""
    text = json.dumps(state, ensure_ascii=False)
    if gold is not None:
        text += "\n" + json.dumps(gold, ensure_ascii=False)
    return scan_text(text)


def question_problems(qid: str, question) -> list:
    """A question specification the training, serving and scoring paths all read the
    same way: noul criteria are absent or a false/true dict, choice criteria name at
    least two distinct options (a dict of descriptions or a list of names), and score
    criteria are a list of at least two levels."""
    if not isinstance(question, dict):
        return [f"{qid}: question must be an object"]
    kind, criteria = question.get("type"), question.get("criteria")
    if kind not in {"noul", "choice", "score"}:
        return [f"{qid}: unknown question type {kind!r}"]
    problems = []
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        problems.append(f"{qid}: instructions must be a nonempty string")
    if kind == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= {"false", "true"}):
            problems.append(f"{qid}: noul criteria must be absent or a dict with false/true keys")
    elif kind == "choice":
        if isinstance(criteria, list):
            if any(not isinstance(c, str) or not c for c in criteria) or len(set(criteria)) != len(criteria):
                problems.append(f"{qid}: list-form choice criteria must be distinct nonempty strings")
        elif not isinstance(criteria, dict):
            problems.append(f"{qid}: choice criteria must be a dict or a list of option names")
        if isinstance(criteria, (dict, list)) and len(criteria) < 2:
            problems.append(f"{qid}: choice criteria need at least two options")
    elif not isinstance(criteria, list) or len(criteria) < 2:
        problems.append(f"{qid}: score criteria must be a list of at least two levels")
    return problems


def site_problems(site_entries) -> list:
    problems = []
    for entry in site_entries:
        questions = entry.get("questions")
        if not isinstance(questions, dict) or not questions:
            problems.append(f"sites.json {entry.get('site')}: questions must be a nonempty object")
            continue
        problems += [f"sites.json {entry['site']}: {p}" for qid, q in questions.items()
                     for p in question_problems(qid, q)]
    return problems


def data_problems(data: Path = ROOT / "data") -> list:
    problems, counts, findings = [], {}, {}
    ids, states = {}, defaultdict(list)
    site_entries = json.loads((data / "sites.json").read_text(encoding="utf-8"))
    if any("cases" in entry for entry in site_entries):
        problems.append("sites.json contains stale per-site case counts")
    problems += site_problems(site_entries)
    sites = {entry["site"]: entry for entry in site_entries}
    for split in SPLITS:
        path = data / f"{split}.jsonl"
        if not path.exists():
            problems.append(f"{path} does not exist")
            continue
        counts[split] = 0
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            counts[split] += 1
            try:
                row = json.loads(line)
                if set(row) != FIELDS or any(not isinstance(v, str) for v in row.values()):
                    raise ValueError("expected six string fields")
                state, questions, gold = (json.loads(row[key]) for key in ("state", "questions", "gold"))
                if not isinstance(state, dict) or not isinstance(questions, dict) or not isinstance(gold, dict):
                    raise ValueError("state, questions and gold must decode to objects")
                if not questions or set(questions) != set(gold):
                    raise ValueError("questions and gold keys differ or are empty")
                site = sites.get(row["site"])
                if not site or row["workflow"] != site["workflow"] or questions != site["questions"]:
                    raise ValueError("site, workflow or questions differ from sites.json")
                for qid, question in questions.items():
                    kind = question.get("type")
                    if kind not in {"noul", "choice", "score"}:
                        raise ValueError(f"{qid}: unknown question type")
                    options = (["false", "true"] if kind == "noul" else
                               list(question["criteria"]) if kind == "choice" else
                               [str(i) for i in range(len(question["criteria"]))])
                    probabilities = gold[qid]["probabilities"]
                    if set(probabilities) != set(options) or any(
                            isinstance(v, bool) or not isinstance(v, (int, float))
                            or not math.isfinite(v) or v < 0 or v > 1
                            for v in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 0.02:
                        raise ValueError(f"{qid}: invalid probability distribution")
            except (ValueError, KeyError, TypeError) as exc:
                problems.append(f"{path}:{number}: {exc}")
                continue
            case_id = row["case_id"]
            if case_id in ids:
                problems.append(f"duplicate case_id {case_id}: {ids[case_id]} and {split}:{number}")
            ids[case_id] = f"{split}:{number}"
            states[json.dumps(state, sort_keys=True, ensure_ascii=False)].append((split, case_id, row["site"]))
            for finding in scan_row(state, gold):
                findings.setdefault(finding, []).append(case_id)
    for (kind, value), cases in sorted(findings.items()):
        problems.append(f"data scrub: {kind} {value} in {len(cases)} case(s), e.g. {cases[0]}")
    print("splits:", ", ".join(f"{split} {count:,}" for split, count in counts.items()))
    overlaps = [group for group in states.values() if len({entry[0] for entry in group}) > 1]
    if overlaps:
        print(f"review: {len(overlaps)} exact state(s) occur in more than one split")
        for group in overlaps:
            print("  " + ", ".join(f"{split}/{case_id} ({site})" for split, case_id, site in group))
            if frozenset(case_id for _, case_id, _ in group) not in KNOWN_CROSS_SPLIT_OVERLAPS:
                problems.append("new cross-split state overlap: " + ", ".join(
                    f"{split}/{case_id}" for split, case_id, _ in group))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data",
                        help="dataset directory with sites.json and the three split files (default: data/)")
    problems = data_problems(parser.parse_args().data)
    for p in problems:
        print(p)
    print("data valid" if not problems else f"data invalid: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
