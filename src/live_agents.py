"""Live (LangChain / LangGraph) wiring for both agents.

Kept in its own module so the offline path never imports LangChain. Imported
lazily by `BaselineAgent` / `AdvancedAgent` only when a live run is possible.

Note: no `from __future__ import annotations` here, tool signatures must be
real types for LangChain to build their schemas.
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver

from memory_store import FACT_KEYS, UserProfileStore, estimate_tokens

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI trả lời bằng tiếng Việt. "
    "Bạn chỉ biết những gì người dùng đã nói trong chính cuộc hội thoại này. "
    "Nếu được hỏi về thông tin chưa từng xuất hiện trong cuộc hội thoại, hãy nói rõ là bạn không biết."
)

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI trả lời bằng tiếng Việt và có bộ nhớ dài hạn về người dùng (User.md). "
    "Luôn ưu tiên fact mới nhất khi người dùng đính chính. "
    "Chỉ ghi nhớ fact ổn định do chính người dùng khẳng định; không ghi câu hỏi, câu đùa hay thông tin tạm thời. "
    "Tôn trọng style trả lời mà người dùng đã yêu cầu."
)


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def turn_usage(result: dict[str, Any], user_message: str) -> tuple[str, int, int]:
    """Return (response text, agent tokens, prompt tokens) for the turn just run."""

    messages = result["messages"]
    turn: list[Any] = []
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            break
        turn.append(message)
    ai_messages = [message for message in reversed(turn) if getattr(message, "type", "") == "ai"]

    response = message_text(ai_messages[-1]) if ai_messages else ""
    input_tokens = 0
    output_tokens = 0
    for message in ai_messages:
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))

    if not output_tokens:
        output_tokens = estimate_tokens(response)
    if not input_tokens:
        input_tokens = sum(estimate_tokens(message_text(message)) for message in messages[:-1])
    return response, estimate_tokens(user_message) + output_tokens, input_tokens


def build_baseline_agent(model):
    """Agent A: short-term thread memory only (checkpointer), no tools, no profile."""

    return create_agent(model, tools=[], system_prompt=BASELINE_SYSTEM_PROMPT, checkpointer=InMemorySaver())


def build_advanced_agent(model, store: UserProfileStore, context_schema: type, threshold_tokens: int, keep_messages: int):
    """Agent B: thread memory + `User.md` tools + profile-injecting prompt + summarization."""

    @tool
    def read_user_profile(runtime: ToolRuntime) -> str:
        """Đọc toàn bộ User.md (hồ sơ bền vững) của người dùng hiện tại."""

        return store.read_text(runtime.context.user_id)

    @tool
    def remember_user_fact(key: str, value: str, runtime: ToolRuntime) -> str:
        """Lưu hoặc cập nhật một fact ổn định về người dùng vào User.md.

        `key` nên là một trong: name, location, profession, response_style,
        favorite_drink, favorite_food, pet, interests. Giá trị mới thay giá trị cũ.
        """

        changed = store.upsert_fact(runtime.context.user_id, key, value, confidence=0.8)
        return "updated" if changed else "unchanged"

    @tool
    def edit_user_profile(search_text: str, replacement: str, runtime: ToolRuntime) -> str:
        """Sửa trực tiếp một đoạn văn bản trong User.md (thay lần xuất hiện đầu tiên)."""

        edited = store.edit_text(runtime.context.user_id, search_text, replacement)
        return "edited" if edited else "search_text not found"

    @dynamic_prompt
    def profile_prompt(request: ModelRequest) -> str:
        profile = store.prompt_view(request.runtime.context.user_id)
        known = profile or "(chưa có fact nào)"
        return (
            f"{ADVANCED_SYSTEM_PROMPT}\n\n"
            f"Các field hợp lệ: {', '.join(FACT_KEYS)}.\n"
            f"## User.md của người dùng hiện tại\n{known}"
        )

    summarization = SummarizationMiddleware(
        model=model,
        trigger=("tokens", threshold_tokens),
        keep=("messages", keep_messages),
    )

    return create_agent(
        model,
        tools=[read_user_profile, remember_user_fact, edit_user_profile],
        middleware=[profile_prompt, summarization],
        context_schema=context_schema,
        checkpointer=InMemorySaver(),
    )
