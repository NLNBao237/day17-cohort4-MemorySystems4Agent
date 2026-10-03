from __future__ import annotations

import json
import shutil
import sys
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import estimate_tokens


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _hits(answer: str, expected: list[str]) -> int:
    # Case-sensitive on purpose: a lowercase match on "AI" would hit words like "hai".
    normalized = unicodedata.normalize("NFC", answer or "")
    return sum(1 for item in expected if unicodedata.normalize("NFC", item) in normalized)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 0.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality score in [0, 1].

    - 70%: coverage of the expected facts
    - 15%: the answer is concise (the user asked for short answers)
    - 15%: the answer commits to something instead of saying it has no information
    """

    if not (answer or "").strip():
        return 0.0
    coverage = _hits(answer, expected) / len(expected) if expected else 0.0
    concise = 1.0 if estimate_tokens(answer) <= 80 else 0.5
    committed = 0.0 if "chưa có thông tin" in answer.lower() else 1.0
    return round(0.7 * coverage + 0.15 * concise + 0.15 * committed, 3)


def judge_quality(judge, question: str, answer: str, expected: list[str]) -> float:
    """Response quality: LLM judge in live mode, heuristic otherwise."""

    if judge is None:
        return heuristic_quality(answer, expected)
    prompt = (
        "Chấm chất lượng câu trả lời của một AI agent trên thang 0 đến 1.\n"
        f"Câu hỏi: {question}\n"
        f"Thông tin đúng cần có: {', '.join(expected)}\n"
        f"Câu trả lời: {answer}\n"
        "Tiêu chí: đúng fact, không nhắc fact cũ đã bị đính chính, ngắn gọn.\n"
        "Chỉ trả về một số thập phân."
    )
    try:
        content = judge.invoke(prompt).content
        text = content if isinstance(content, str) else str(content)
        return max(0.0, min(1.0, float(text.strip().split()[0].replace(",", "."))))
    except Exception:
        return heuristic_quality(answer, expected)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
    judge=None,
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Every conversation runs in its own thread. Recall questions are asked in
    fresh threads, so only memory that survives a thread change can answer them.
    """

    thread_ids: list[str] = []
    user_ids: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = conversation["user_id"]
        if user_id not in user_ids:
            user_ids.append(user_id)
    size_before = sum(agent.memory_file_size(user_id) for user_id in user_ids)

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = f"{conversation['id']}-main"
        thread_ids.append(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, turn)

        for index, item in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread = f"{conversation['id']}-recall-{index}"
            thread_ids.append(recall_thread)
            answer = agent.reply(user_id, recall_thread, item["question"])["response"]
            recall_scores.append(recall_points(answer, item["expected_contains"]))
            quality_scores.append(judge_quality(judge, item["question"], answer, item["expected_contains"]))

    size_after = sum(agent.memory_file_size(user_id) for user_id in user_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids),
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=size_after - size_before,
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


HEADERS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render the rows as a markdown table."""

    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=HEADERS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join("---" for _ in HEADERS) + "|"]
        lines += ["| " + " | ".join(str(cell) for cell in line) + " |" for line in table]
        return "\n".join(lines)


def run_suite(name: str, conversations: list[dict[str, Any]], config, force_offline: bool, judge=None) -> list[BenchmarkRow]:
    """Run Baseline and Advanced on the same input, each from a clean state."""

    # Isolated, disposable state so every run starts from an empty memory.
    suite_state = config.state_dir / "benchmark" / name
    if suite_state.exists():
        shutil.rmtree(suite_state)
    suite_config = replace(config, state_dir=suite_state)

    baseline = BaselineAgent(suite_config, force_offline=force_offline)
    advanced = AdvancedAgent(suite_config, force_offline=force_offline)
    return [
        run_agent_benchmark("Baseline", baseline, conversations, suite_config, judge),
        run_agent_benchmark("Advanced", advanced, conversations, suite_config, judge),
    ]


def main() -> None:
    """Run both benchmark suites and print one comparison table per suite.

    Offline (deterministic) by default. Pass `--live` to call the configured LLM.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(Path(__file__).resolve().parent.parent)
    live = "--live" in sys.argv[1:]

    judge = None
    if live and config.judge_model.has_credentials():
        from model_provider import build_chat_model

        judge = build_chat_model(config.judge_model)

    suites = [
        ("Standard Benchmark", "standard", "conversations.json"),
        ("Long-Context Stress Benchmark", "stress", "advanced_long_context.json"),
    ]

    print(f"Mode: {'live (' + config.model.provider + ':' + config.model.model_name + ')' if live else 'offline (deterministic)'}")
    print(f"Compact threshold: {config.compact_threshold_tokens} tokens, keep {config.compact_keep_messages} messages")

    for title, name, filename in suites:
        conversations = load_conversations(config.data_dir / filename)
        rows = run_suite(name, conversations, config, force_offline=not live, judge=judge)
        turns = sum(len(conversation["turns"]) for conversation in conversations)
        print(f"\n## {title} ({len(conversations)} conversations, {turns} turns)\n")
        print(format_rows(rows))

        baseline, advanced = rows
        if baseline.prompt_tokens_processed:
            delta = (advanced.prompt_tokens_processed - baseline.prompt_tokens_processed) / baseline.prompt_tokens_processed
            print(f"\nAdvanced vs Baseline, prompt tokens processed: {delta:+.1%}")


if __name__ == "__main__":
    main()
