from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    answer_from_facts,
    apply_candidates,
    estimate_tokens,
    extract_profile_candidates,
)
from model_provider import build_chat_model

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI trả lời bằng tiếng Việt. "
    "Bạn chỉ biết những gì người dùng đã nói trong chính cuộc hội thoại này."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A.

    - Within-session memory only: the full message list of one thread
    - No persistent `User.md`
    - Forgets every fact as soon as the thread id changes
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting for one turn."""

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behavior.

        The baseline re-sends the whole thread on every turn, so its prompt
        load grows with the length of the thread. It can only answer from
        facts stated inside this same thread.
        """

        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = estimate_tokens(SYSTEM_PROMPT) + sum(
            estimate_tokens(item["content"]) for item in session.messages
        )

        # Facts visible to the baseline = whatever was said in this thread only.
        thread_facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                apply_candidates(
                    thread_facts,
                    extract_profile_candidates(item["content"]),
                    self.config.profile_confidence_threshold,
                )

        response = answer_from_facts(message, thread_facts) or "Đã ghi nhận trong phiên này."
        session.messages.append({"role": "assistant", "content": response})

        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
            "mode": "offline",
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        from live_agents import turn_usage

        session = self._session(thread_id)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response, agent_tokens, prompt_tokens = turn_usage(result, message)

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Wire `create_agent` + `InMemorySaver` when a live run is possible.

        Returns None (offline mode) when credentials or dependencies are missing.
        """

        if not self.config.model.has_credentials():
            return None
        try:
            from live_agents import build_baseline_agent

            return build_baseline_agent(build_chat_model(self.config.model))
        except Exception as error:  # missing SDK, bad config, ...
            print(f"[baseline] live agent unavailable, using offline mode: {error}")
            return None
