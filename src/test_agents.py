from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, load_conversations, recall_points, run_suite
from config import load_config
from memory_store import (
    CompactMemoryManager,
    FactRecord,
    UserProfileStore,
    estimate_tokens,
    extract_profile_candidates,
    extract_profile_updates,
    fact_score,
)

LONG_TURN = (
    "Đây là một đoạn tin tức rất dài để ép hội thoại vượt ngưỡng compact. "
    "Nội dung nói về roadmap, dependency, rủi ro vận hành và cách cân bằng giữa scale với efficiency. " * 3
)


def make_config(tmp_path: Path, threshold: int = 200, keep: int = 2):
    """Isolated config: state lives in tmp_path, compaction triggers quickly."""

    config = load_config()
    return replace(
        config,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        profile_confidence_threshold=0.6,
    )


# --- User.md -----------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Reading a missing profile gives an empty default and does not create a file.
    assert "# User Profile: dungct" in store.read_text("dungct")
    assert store.file_size("dungct") == 0

    path = store.write_text("dungct", "# User Profile: dungct\n\n## Facts\n- location: Đà Nẵng\n")
    assert path.exists() and path.name == "User.md"
    assert store.file_size("dungct") > 0
    assert store.facts("dungct") == {"location": "Đà Nẵng"}

    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert "Huế" in store.read_text("dungct")
    assert "Đà Nẵng" not in store.read_text("dungct")
    assert store.edit_text("dungct", "không tồn tại", "x") is False


def test_user_id_is_sanitized_into_a_safe_path(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.path_for("../../etc/passwd")
    assert (tmp_path / "profiles").resolve() in path.resolve().parents
    assert store.path_for("a") != store.path_for("b")


def test_upsert_replaces_old_value_on_correction(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.upsert_fact("u", "location", "Đà Nẵng") is True
    assert store.upsert_fact("u", "location", "Huế") is True
    assert store.upsert_fact("u", "location", "Huế") is False

    assert store.facts("u")["location"] == "Huế"
    facts_section = store.read_text("u").split("## History")[0]
    assert "Đà Nẵng" not in facts_section
    assert store.fact_records("u")["location"].seen == 3


# --- extraction guardrails ---------------------------------------------------


def test_extract_basic_facts() -> None:
    updates = extract_profile_updates("Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    assert updates == {"location": "Đà Nẵng", "profession": "backend engineer"}
    assert extract_profile_updates("Chào bạn, mình tên là DũngCT.") == {"name": "DũngCT"}
    assert extract_profile_updates("Mình nuôi một bé corgi tên Bơ.") == {"pet": "corgi tên Bơ"}


def test_questions_are_not_stored_as_facts() -> None:
    assert extract_profile_updates("Hiện tại mình làm nghề gì và mình còn ở Huế không?") == {}
    assert extract_profile_updates("Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.") == {}
    assert extract_profile_updates("Bạn có biết DũngCT không?") == {}


def test_correction_keeps_only_the_new_fact() -> None:
    message = "À, mình đính chính một chút: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa."
    assert extract_profile_updates(message) == {"location": "Huế"}

    message = "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer."
    assert extract_profile_updates(message) == {"profession": "MLOps engineer"}


def test_confidence_threshold_rejects_noise() -> None:
    joke = "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager cho đỡ phải ngồi canh pipeline."
    candidates = extract_profile_candidates(joke)
    assert [c.value for c in candidates] == ["product manager"]
    assert candidates[0].confidence < 0.6
    assert extract_profile_updates(joke) == {}
    # Without the gate the joke would be stored as the profession.
    assert extract_profile_updates(joke, min_confidence=0.0) == {"profession": "product manager"}

    trip = "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại."
    assert extract_profile_updates(trip) == {}

    past = "Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng."
    assert extract_profile_updates(past) == {"location": "Đà Nẵng"}


def test_memory_decay_prefers_recent_and_repeated_facts(tmp_path: Path) -> None:
    old = FactRecord("x", confidence=0.9, seen=1, last_turn=0)
    fresh = FactRecord("y", confidence=0.9, seen=1, last_turn=50)
    repeated = FactRecord("z", confidence=0.9, seen=5, last_turn=0)
    assert fact_score(fresh, now=50) > fact_score(old, now=50)
    assert fact_score(repeated, now=50) > fact_score(old, now=50)

    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("u", "favorite_food", "mì Quảng")
    for _ in range(30):
        store.upsert_fact("u", "location", "Huế")
    view = store.prompt_view("u", max_facts=1)
    assert "Huế" in view and "mì Quảng" not in view


# --- compact memory ----------------------------------------------------------


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=200, keep_messages=2)
    manager.append("t1", "user", "Câu ngắn.")
    assert manager.compaction_count("t1") == 0

    for index in range(6):
        manager.append("t1", "user", f"Lượt {index}. {LONG_TURN}")
        manager.append("t1", "assistant", "Đã ghi nhận.")

    context = manager.context("t1")
    assert manager.compaction_count("t1") >= 1
    assert len(context["messages"]) <= 3
    assert context["summary"]
    # The summary is bounded: it does not grow with the thread.
    assert len(str(context["summary"]).splitlines()) <= manager.summary_max_items
    # Threads are isolated from each other.
    assert manager.compaction_count("other") == 0


def test_short_thread_does_not_compact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path, threshold=800, keep=4), force_offline=True)
    for turn in ["Chào bạn, mình tên là An.", "Mình ở Huế.", "Mình tên gì?"]:
        agent.reply("an", "short", turn)
    assert agent.compaction_count("short") == 0


# --- agents ------------------------------------------------------------------


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for agent in (baseline, advanced):
        agent.reply("dungct", "s1", "Chào bạn, mình tên là DũngCT.")
        agent.reply("dungct", "s1", "Đồ uống yêu thích là cà phê sữa đá.")

    question = "Mình tên gì và đồ uống yêu thích là gì?"

    # Same thread: both agents remember (the baseline is naive, not broken).
    assert "DũngCT" in baseline.reply("dungct", "s1", question)["response"]

    # New thread: only the agent with User.md still knows.
    baseline_answer = baseline.reply("dungct", "s2", question)["response"]
    advanced_answer = advanced.reply("dungct", "s2", question)["response"]
    assert "DũngCT" not in baseline_answer and "cà phê sữa đá" not in baseline_answer
    assert "DũngCT" in advanced_answer and "cà phê sữa đá" in advanced_answer

    # A brand new agent instance (process restart) still recalls from disk.
    restarted = AdvancedAgent(config, force_offline=True)
    assert "DũngCT" in restarted.reply("dungct", "s3", question)["response"]

    # Memory is per user.
    assert "DũngCT" not in advanced.reply("someone_else", "s4", question)["response"]
    assert advanced.memory_file_size("dungct") > 0
    assert baseline.memory_file_size("dungct") == 0


def test_advanced_recalls_latest_fact_after_correction(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "a", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply("u", "b", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    agent.reply("u", "b", "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    agent.reply("u", "b", "Có lúc mình đùa là hay là chuyển sang product manager.")

    answer = agent.reply("u", "c", "Hiện tại mình làm nghề gì và đang ở đâu?")["response"]
    assert "MLOps engineer" in answer and "Huế" in answer
    assert "backend engineer" not in answer
    assert "Đà Nẵng" not in answer
    assert "product manager" not in answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=300, keep=2)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    last = {}
    for index in range(12):
        turn = f"Lượt {index}. {LONG_TURN}"
        last["baseline"] = baseline.reply("u", "long", turn)
        last["advanced"] = advanced.reply("u", "long", turn)

    assert advanced.compaction_count("long") >= 2
    assert baseline.compaction_count("long") == 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    # Per turn, the advanced context stays bounded while the baseline keeps growing.
    assert last["advanced"]["prompt_tokens"] < last["baseline"]["prompt_tokens"] / 2
    # The tokens exchanged in the conversation itself are the same order of magnitude:
    # compaction optimizes the context carried along, not what is said.
    assert abs(advanced.token_usage("long") - baseline.token_usage("long")) < 0.1 * baseline.token_usage("long")


def test_short_thread_costs_more_prompt_for_advanced(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=800, keep=4)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for turn in ["Chào bạn, mình tên là DũngCT.", "Mình ở Huế.", "Mình thích Python, AI ứng dụng."]:
        baseline.reply("u", "short", turn)
        advanced.reply("u", "short", turn)
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")


# --- benchmark ---------------------------------------------------------------


def test_scoring_helpers() -> None:
    assert estimate_tokens("") == 0 and estimate_tokens("   ") == 0
    assert estimate_tokens("abcd" * 10) == 10
    assert recall_points("Tên: DũngCT, uống cà phê sữa đá", ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points("Tên: DũngCT", ["DũngCT", "cà phê sữa đá"]) == 0.5
    assert recall_points("Mình chưa có thông tin.", ["DũngCT"]) == 0.0
    assert heuristic_quality("Tên: DũngCT", ["DũngCT"]) > heuristic_quality("Mình chưa có thông tin.", ["DũngCT"])


def test_benchmark_tells_the_expected_story(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=800, keep=4)

    baseline, advanced = run_suite(
        "standard", load_conversations(config.data_dir / "conversations.json"), config, force_offline=True
    )
    assert baseline.recall_score == 0.0
    assert advanced.recall_score >= 0.9
    assert advanced.memory_growth_bytes > 0 and baseline.memory_growth_bytes == 0
    # Short conversations: the profile is extra context, so advanced costs more.
    assert advanced.prompt_tokens_processed > baseline.prompt_tokens_processed

    baseline, advanced = run_suite(
        "stress", load_conversations(config.data_dir / "advanced_long_context.json"), config, force_offline=True
    )
    assert advanced.compactions >= 2
    assert advanced.recall_score >= 0.9 and baseline.recall_score == 0.0
    # Long conversation: compaction makes advanced clearly cheaper to run.
    assert advanced.prompt_tokens_processed < 0.7 * baseline.prompt_tokens_processed
