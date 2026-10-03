from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    FIELD_LABELS,
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    estimate_tokens,
    extract_profile_candidates,
)
from model_provider import build_chat_model

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI trả lời bằng tiếng Việt và có bộ nhớ dài hạn về người dùng (User.md). "
    "Luôn ưu tiên fact mới nhất khi người dùng đính chính."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Three memory layers:
    1. within-session memory: recent messages of the thread, kept in full
    2. persistent `User.md`: stable facts that survive across threads
    3. compact memory: older messages of a long thread folded into a summary
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between live mode and the deterministic offline mode."""

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _persist_profile_updates(self, user_id: str, message: str) -> dict[str, str]:
        """Layer 2: extract stable facts and write the confident ones into `User.md`."""

        return self.profile_store.apply_updates(
            user_id,
            extract_profile_candidates(message),
            min_confidence=self.config.profile_confidence_threshold,
        )

    def _record_usage(self, thread_id: str, agent_tokens: int, prompt_tokens: int) -> None:
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path.

        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory (may trigger a compaction).
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        updates = self._persist_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        response = self._offline_response(user_id, thread_id, message, updates)
        self.compact_memory.append(thread_id, "assistant", response)

        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        self._record_usage(thread_id, agent_tokens, prompt_tokens)

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": updates,
            "compactions": self.compaction_count(thread_id),
            "mode": "offline",
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        from live_agents import turn_usage

        # The rule-based extractor still runs as a guardrail next to the LLM tools.
        updates = self._persist_profile_updates(user_id, message)
        # Mirror of the thread, used only to report compaction counts.
        self.compact_memory.append(thread_id, "user", message)

        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id))),
        )
        response, agent_tokens, prompt_tokens = turn_usage(result, message)
        self.compact_memory.append(thread_id, "assistant", response)
        self._record_usage(thread_id, agent_tokens, prompt_tokens)

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": updates,
            "compactions": self.compaction_count(thread_id),
            "mode": "live",
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: system prompt + `User.md` facts + summary + recent messages."""

        context = self.compact_memory.context(thread_id)
        total = estimate_tokens(SYSTEM_PROMPT)
        total += estimate_tokens(self.profile_store.prompt_view(user_id))
        total += estimate_tokens(str(context["summary"]))
        for item in context["messages"]:  # type: ignore[union-attr]
            total += estimate_tokens(item["content"])
        return total

    def _offline_response(
        self,
        user_id: str,
        thread_id: str,
        message: str,
        updates: dict[str, str] | None = None,
    ) -> str:
        """Deterministic answer built from persisted memory.

        Recall questions are answered from `User.md`, so they work in any
        thread. Other turns get a short acknowledgement of what was stored.
        """

        answer = answer_from_facts(message, self.profile_store.facts(user_id))
        if answer is not None:
            return answer

        stored = [f"{FIELD_LABELS.get(key, key).lower()} = {value}" for key, value in (updates or {}).items() if value]
        if stored:
            return "Đã ghi nhớ vào User.md: " + "; ".join(stored) + "."
        return "Đã ghi nhận."

    def _maybe_build_langchain_agent(self):
        """Wire the live agent: thread checkpointer, `User.md` tools, dynamic prompt, summarization.

        Returns None (offline mode) when credentials or dependencies are missing.
        """

        if not self.config.model.has_credentials():
            return None
        try:
            from live_agents import build_advanced_agent

            return build_advanced_agent(
                build_chat_model(self.config.model),
                self.profile_store,
                context_schema=AgentContext,
                threshold_tokens=self.config.compact_threshold_tokens,
                keep_messages=self.config.compact_keep_messages,
            )
        except Exception as error:  # missing SDK, bad config, ...
            print(f"[advanced] live agent unavailable, using offline mode: {error}")
            return None
