from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~4 characters per token.

    Not tokenizer-accurate, but stable and deterministic, which is what the
    offline benchmark needs.
    """

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


# ---------------------------------------------------------------------------
# Profile schema (structured entity fields)
# ---------------------------------------------------------------------------

FACT_KEYS = (
    "name",
    "location",
    "profession",
    "response_style",
    "favorite_drink",
    "favorite_food",
    "pet",
    "interests",
)

FIELD_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm",
}

MAX_INTERESTS = 5
MAX_HISTORY = 8
DECAY_PER_TURN = 0.98

# Order in which response-style slots are rendered.
_STYLE_SLOTS = ("length", "format", "example", "emphasis", "structure", "clarity")


def _style_slot(descriptor: str) -> str:
    text = descriptor.lower()
    if "bullet" in text:
        return "format"
    if "ví dụ" in text:
        return "example"
    if "trade-off" in text:
        return "emphasis"
    if "cấu trúc" in text:
        return "structure"
    if "rõ ý" in text:
        return "clarity"
    return "length"


def _split_items(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"\s*[;,]\s*", value or "") if item.strip()]


def merge_fact_value(key: str, old: str | None, new: str) -> str:
    """Conflict handling: decide what a field holds after a new mention.

    - scalar fields (name, location, profession, ...): the newest value replaces the old one
    - response_style: slot-based merge, a new preference only overrides its own slot
    - interests: LRU list, re-mentioned items move to the end, oldest are dropped past the cap
    """

    new = (new or "").strip()
    if not old:
        old = ""

    if key == "response_style":
        slots: dict[str, str] = {}
        for descriptor in _split_items(old):
            slots[_style_slot(descriptor)] = descriptor
        for descriptor in _split_items(new):
            slot = _style_slot(descriptor)
            current = slots.get(slot, "")
            # "bullet" is less specific than "3 bullet": never downgrade.
            if slot == "format" and re.search(r"\d", current) and not re.search(r"\d", descriptor):
                continue
            slots[slot] = descriptor
        return ", ".join(slots[slot] for slot in _STYLE_SLOTS if slot in slots)

    if key == "interests":
        items = _split_items(old)
        for item in _split_items(new):
            items = [existing for existing in items if existing.casefold() != item.casefold()]
            items.append(item)
        return ", ".join(items[-MAX_INTERESTS:])

    return new


# ---------------------------------------------------------------------------
# Fact extraction with confidence
# ---------------------------------------------------------------------------


@dataclass
class FactCandidate:
    """One fact detected in a user message, before the confidence gate."""

    key: str
    value: str
    confidence: float
    retract: bool = False
    reason: str = ""


_NEGATION = re.compile(r"không còn|không phải|chưa từng|đừng|chỉ là", re.IGNORECASE)
_PAST = re.compile(r"lúc đầu|trước đó|trước đây|hồi trước", re.IGNORECASE)
_NOISE = re.compile(r"\bđùa\b|giả sử|\bhay là\b", re.IGNORECASE)
_PRESENT = re.compile(r"\b(?:đang|hiện|vẫn|giờ)\b", re.IGNORECASE)
_SUBJECT = re.compile(r"\b(?:mình|tôi|hiện|giờ|đang|vẫn)\b", re.IGNORECASE)

_ROLE = (
    r"((?:[A-Za-z][A-Za-z\-]*\s+)?"
    r"(?:engineer|developer|manager|scientist|analyst|designer|researcher|architect))\b"
)
_PROFESSION = re.compile(r"(?:\blàm|chuyển sang|\blà|nghề(?:\s+nghiệp)?)\s+" + _ROLE, re.IGNORECASE)

_NAME_STRONG = re.compile(r"(?:\b(?:mình|tôi)\s+tên\s+(?:là\s+)?|\btên\s+(?:mình|tôi)\s+là\s+)", re.IGNORECASE)
_NAME_WEAK = re.compile(r"^\s*tên\s+(?:là\s+)?", re.IGNORECASE)

_LOCATION_MOVE = re.compile(r"nơi ở\b.*?\btừ\s+.+?\s+sang\s+", re.IGNORECASE)
_LOCATION_IS = re.compile(r"nơi ở(?:\s+hiện tại)?(?:\s+của\s+mình)?\s+(?:vẫn\s+)?là\s+", re.IGNORECASE)
_LOCATION_AT = re.compile(r"\bở\s+", re.IGNORECASE)

_PET = re.compile(
    r"\bnuôi\s+(?:một\s+)?(?:bé\s+|con\s+|chú\s+|em\s+)?([^\W\d_]+)(?:\s+tên\s+([^\W\d_]+))?",
    re.IGNORECASE,
)
_DRINK = re.compile(r"đồ uống yêu thích(?:\s+của\s+mình)?\s+(?:vẫn\s+)?là\s+([^,.;!?]+)", re.IGNORECASE)
_FOOD = re.compile(r"món ăn yêu thích(?:\s+của\s+mình)?\s+(?:vẫn\s+)?là\s+([^,.;!?]+)", re.IGNORECASE)
_INTERESTS = re.compile(
    r"(?<!không )(?:\bthích|quan tâm(?:\s+(?:nhiều|nhất))?\s+(?:đến|tới))\s+([^.!?]+)",
    re.IGNORECASE,
)
_STYLE_CONTEXT = re.compile(r"trả lời|giải thích|style", re.IGNORECASE)


def _take_capitalized(text: str, max_words: int = 4) -> str:
    """Take the leading run of capitalized words, e.g. `Đà Nẵng vài tháng` -> `Đà Nẵng`."""

    match = re.match(r"[^\W_]\w*(?: [^\W_]\w*)*", text)
    if not match:
        return ""
    words: list[str] = []
    for word in match.group(0).split(" "):
        if not word[0].isupper() or len(words) >= max_words:
            break
        words.append(word)
    return " ".join(words)


def _split_sentences(message: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", message) if part.strip()]


def _split_clauses(sentence: str) -> list[str]:
    parts = re.split(r"\s*[,;:]\s*|\s+chứ\s+|\s+nhưng\s+", sentence)
    return [part.strip() for part in parts if part and part.strip()]


def _style_descriptors(sentence: str) -> list[str]:
    lowered = sentence.lower()
    found: list[str] = []
    if re.search(r"\bngắn\b|\bgọn\b", lowered):
        found.append("ngắn gọn")
    numbered = re.search(r"(\d+)\s*bullet", lowered)
    if numbered:
        found.append(f"{numbered.group(1)} bullet")
    elif "bullet" in lowered:
        found.append("bullet")
    if "ví dụ thực chiến" in lowered:
        found.append("có ví dụ thực chiến")
    elif "ví dụ thực tế" in lowered:
        found.append("có ví dụ thực tế")
    if "trade-off" in lowered:
        found.append("nhấn trade-off")
    if "cấu trúc" in lowered:
        found.append("có cấu trúc")
    if "rõ ý" in lowered:
        found.append("rõ ý")
    return found


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return not lowered or lowered.startswith("gì") or lowered in {"ai", "đâu", "nào"}


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Structured entity extraction: raw user text -> fact candidates with confidence.

    Guardrails against storing wrong facts:
    - question sentences are skipped entirely (asking is not telling)
    - negated mentions ("không còn ở X") become retractions instead of facts
    - past-tense mentions ("lúc đầu mình nói ...") and jokes / hypotheticals get a
      low confidence, so the confidence threshold drops them
    """

    candidates: list[FactCandidate] = []

    for raw_sentence in _split_sentences(_nfc(message)):
        if raw_sentence.endswith("?"):
            continue
        sentence = raw_sentence.rstrip(".! ")
        noisy = bool(_NOISE.search(sentence))

        def scored(base: float, reason: str) -> tuple[float, str]:
            if noisy:
                return 0.2, "joke-or-hypothetical"
            return base, reason

        # --- sentence-level fields (their values may contain commas) ---
        for key, pattern in (("favorite_drink", _DRINK), ("favorite_food", _FOOD)):
            match = pattern.search(sentence)
            if match and not _is_placeholder(match.group(1)):
                confidence, reason = scored(0.9, "explicit-favorite")
                candidates.append(FactCandidate(key, match.group(1).strip(), confidence, reason=reason))

        match = _INTERESTS.search(sentence)
        if match:
            items = re.split(r"\s*,\s*(?:và\s+)?|\s+và\s+", match.group(1))
            technical = [
                item.strip()
                for item in items
                if item.strip() and item.strip()[0].isupper() and len(item.split()) <= 3
            ]
            if technical:
                confidence, reason = scored(0.75, "stated-interest")
                candidates.append(FactCandidate("interests", ", ".join(technical), confidence, reason=reason))

        if _STYLE_CONTEXT.search(sentence):
            descriptors = _style_descriptors(sentence)
            if descriptors:
                confidence, reason = scored(0.85, "style-preference")
                candidates.append(FactCandidate("response_style", ", ".join(descriptors), confidence, reason=reason))

        # --- clause-level fields (negation / tense are scoped to one clause) ---
        for clause in _split_clauses(sentence):
            name_match = _NAME_STRONG.search(clause)
            name_confidence = 0.95
            if not name_match:
                name_match = _NAME_WEAK.search(clause)
                name_confidence = 0.8
            if name_match:
                name = _take_capitalized(clause[name_match.end():])
                if name:
                    confidence, reason = scored(name_confidence, "self-introduction")
                    candidates.append(FactCandidate("name", name, confidence, reason=reason))

            located = False
            for pattern in (_LOCATION_MOVE, _LOCATION_IS):
                move = pattern.search(clause)
                if move:
                    place = _take_capitalized(clause[move.end():], max_words=3)
                    if place:
                        confidence, reason = scored(0.95, "explicit-residence")
                        candidates.append(FactCandidate("location", place, confidence, reason=reason))
                        located = True
                        break
            if not located:
                for at in _LOCATION_AT.finditer(clause):
                    prefix = clause[: at.start()]
                    place = _take_capitalized(clause[at.end():], max_words=3)
                    if not place or prefix.rstrip().lower().endswith("nơi"):
                        continue
                    if _NEGATION.search(prefix):
                        candidates.append(FactCandidate("location", place, 0.9, retract=True, reason="negated"))
                    elif _PAST.search(prefix):
                        candidates.append(FactCandidate("location", place, 0.3, reason="past-mention"))
                    elif _SUBJECT.search(prefix):
                        base = 0.9 if _PRESENT.search(prefix) else 0.85
                        confidence, reason = scored(base, "stated-residence")
                        candidates.append(FactCandidate("location", place, confidence, reason=reason))

            for job in _PROFESSION.finditer(clause):
                prefix = clause[: job.start()]
                role = job.group(1).strip()
                if _NEGATION.search(prefix):
                    candidates.append(FactCandidate("profession", role, 0.9, retract=True, reason="negated"))
                elif _PAST.search(prefix):
                    candidates.append(FactCandidate("profession", role, 0.3, reason="past-mention"))
                else:
                    confidence, reason = scored(0.9, "stated-profession")
                    candidates.append(FactCandidate("profession", role, confidence, reason=reason))

            pet = _PET.search(clause)
            if pet and not _is_placeholder(pet.group(1)):
                value = pet.group(1)
                if pet.group(2) and pet.group(2)[0].isupper():
                    value = f"{value} tên {pet.group(2)}"
                confidence, reason = scored(0.85, "stated-pet")
                candidates.append(FactCandidate("pet", value, confidence, reason=reason))

    return candidates


def apply_candidates(
    facts: dict[str, str],
    candidates: list[FactCandidate],
    min_confidence: float = 0.6,
) -> dict[str, str]:
    """Apply candidates to a plain fact dict in place. Returns the changed fields."""

    changed: dict[str, str] = {}
    for candidate in candidates:
        if candidate.confidence < min_confidence:
            continue
        current = facts.get(candidate.key)
        if candidate.retract:
            if current is not None and current.casefold() == candidate.value.casefold():
                del facts[candidate.key]
                changed[candidate.key] = ""
            continue
        merged = merge_fact_value(candidate.key, current, candidate.value)
        if merged != current:
            facts[candidate.key] = merged
            changed[candidate.key] = merged
    return changed


def extract_profile_updates(message: str, min_confidence: float = 0.6) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Returns only the facts that are confidently present in the message.
    """

    facts: dict[str, str] = {}
    apply_candidates(facts, extract_profile_candidates(message), min_confidence)
    return facts


# ---------------------------------------------------------------------------
# Recall questions (shared by both agents in offline mode)
# ---------------------------------------------------------------------------

_QUERY = re.compile(
    r"(?:^|[.!?:]\s+)(?:nhắc lại|tóm tắt|hãy nhắc|nhớ lại)|nhắc lại giúp|(?:thử|có thể)\s+(?:nhớ lại|nhắc lại|mô tả|tóm tắt)",
    re.IGNORECASE,
)

_FIELD_KEYWORDS = (
    ("name", r"\btên\b"),
    ("location", r"ở đâu|nơi ở|còn ở|đang ở"),
    ("profession", r"\bnghề\b|làm gì|công việc"),
    ("favorite_drink", r"đồ uống"),
    ("favorite_food", r"món ăn"),
    ("pet", r"\bnuôi\b|thú cưng"),
    ("response_style", r"style|kiểu trả lời|cách trả lời|phong cách"),
    ("interests", r"quan tâm|sở thích"),
)


def is_recall_query(message: str) -> bool:
    text = _nfc(message)
    return "?" in text or bool(_QUERY.search(text))


def requested_fields(message: str) -> list[str]:
    lowered = _nfc(message).lower()
    fields = [key for key, pattern in _FIELD_KEYWORDS if re.search(pattern, lowered)]
    if re.search(r"là ai\b", lowered):
        for key in ("name", "profession", "interests"):
            if key not in fields:
                fields.append(key)
    return fields


def answer_from_facts(message: str, facts: dict[str, str]) -> str | None:
    """Deterministic answer to a recall question, using only the given facts.

    Returns None when the message is not a recall question. The caller decides
    which facts are visible: the baseline passes facts from the current thread,
    the advanced agent passes facts persisted in `User.md`.
    """

    if not is_recall_query(message):
        return None

    fields = requested_fields(message)
    if not fields:
        fields = [key for key in ("name", "profession", "location") if key in facts]
        if not fields:
            return "Mình chưa có thông tin nào được ghi nhớ để trả lời câu này."

    known = [f"- {FIELD_LABELS[key]}: {facts[key]}" for key in fields if facts.get(key)]
    missing = [FIELD_LABELS[key].lower() for key in fields if not facts.get(key)]

    lines = list(known)
    if missing:
        lines.append("- Mình chưa có thông tin về: " + ", ".join(missing) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------


@dataclass
class FactRecord:
    value: str
    confidence: float = 1.0
    seen: int = 1
    last_turn: int = 0


def fact_score(record: FactRecord, now: int) -> float:
    """Memory decay: older, rarely repeated facts lose priority over time."""

    age = max(0, now - record.last_turn)
    reinforcement = 1.0 + 0.1 * min(record.seen, 5)
    return record.confidence * (DECAY_PER_TURN**age) * reinforcement


@dataclass
class _Profile:
    turn: int = 0
    facts: dict[str, FactRecord] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)


_FACT_LINE = re.compile(r"^- ([A-Za-z_]+):\s*(.*)$")
_TURN_LINE = re.compile(r"^<!-- turn: (\d+) -->$")


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("|", "/")).strip()


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`: one markdown file per user.

    File layout:

        # User Profile: <user_id>
        <!-- turn: 12 -->

        ## Facts
        - location: Huế | conf=0.90 | seen=3 | last=12

        ## History
        - turn 12: location: Đà Nẵng -> Huế
    """

    root_dir: Path

    # --- raw file operations -------------------------------------------------

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", (user_id or "").strip()).strip("._") or "anonymous"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self._render(user_id, _Profile())

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # --- structured helpers --------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        return {key: record.value for key, record in self._load(user_id).facts.items()}

    def fact_records(self, user_id: str) -> dict[str, FactRecord]:
        return self._load(user_id).facts

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float = 1.0) -> bool:
        """Insert or update one fact. Returns whether the stored value changed."""

        changed = self.apply_updates(user_id, [FactCandidate(key, value, confidence)], min_confidence=0.0)
        return key in changed

    def remove_fact(self, user_id: str, key: str) -> bool:
        profile = self._load(user_id)
        if key not in profile.facts:
            return False
        del profile.facts[key]
        self.write_text(user_id, self._render(user_id, profile))
        return True

    def apply_updates(
        self,
        user_id: str,
        candidates: list[FactCandidate],
        min_confidence: float = 0.6,
    ) -> dict[str, str]:
        """Persist the candidates of one user turn.

        - confidence threshold: low-confidence candidates never reach the file
        - conflict handling: a correction overwrites the old value (never both),
          and the change is recorded in a short, capped history
        - decay bookkeeping: every mention refreshes `seen` and `last`
        """

        profile = self._load(user_id)
        profile.turn += 1
        changed: dict[str, str] = {}
        touched = False

        for candidate in candidates:
            if candidate.confidence < min_confidence:
                continue
            key = candidate.key.strip().lower()
            value = _clean_value(candidate.value)
            if not key or not value:
                continue
            record = profile.facts.get(key)

            if candidate.retract:
                if record is not None and record.value.casefold() == value.casefold():
                    del profile.facts[key]
                    profile.history.append(f"turn {profile.turn}: {key}: {record.value} -> (retracted)")
                    changed[key] = ""
                    touched = True
                continue

            merged = merge_fact_value(key, record.value if record else None, value)
            touched = True
            if record is None:
                profile.facts[key] = FactRecord(merged, candidate.confidence, 1, profile.turn)
                changed[key] = merged
                continue
            if merged != record.value:
                # A pure reordering of a list field is not a correction worth logging.
                if sorted(_split_items(merged)) != sorted(_split_items(record.value)):
                    profile.history.append(f"turn {profile.turn}: {key}: {record.value} -> {merged}")
                record.value = merged
                record.confidence = candidate.confidence
                changed[key] = merged
            else:
                record.confidence = max(record.confidence, candidate.confidence)
            record.seen += 1
            record.last_turn = profile.turn

        if touched:
            profile.history = profile.history[-MAX_HISTORY:]
            self.write_text(user_id, self._render(user_id, profile))
        return changed

    def prompt_view(self, user_id: str, max_facts: int | None = None) -> str:
        """Facts as injected into the prompt: no bookkeeping metadata, no history.

        With `max_facts`, memory decay decides which facts survive the budget.
        """

        profile = self._load(user_id)
        if not profile.facts:
            return ""
        ranked = sorted(
            profile.facts.items(),
            key=lambda item: fact_score(item[1], profile.turn),
            reverse=True,
        )
        if max_facts is not None:
            ranked = ranked[:max_facts]
        kept = dict(ranked)
        ordered = [key for key in FACT_KEYS if key in kept] + [key for key in kept if key not in FACT_KEYS]
        return "\n".join(f"- {key}: {kept[key].value}" for key in ordered)

    # --- parsing / rendering -------------------------------------------------

    def _load(self, user_id: str) -> _Profile:
        profile = _Profile()
        path = self.path_for(user_id)
        if not path.exists():
            return profile

        section = ""
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            if not line or line.startswith("# User Profile"):
                continue
            turn_match = _TURN_LINE.match(line)
            if turn_match:
                profile.turn = int(turn_match.group(1))
                continue
            if line.startswith("## "):
                section = line[3:].strip().lower()
                if section not in {"facts", "history"}:
                    profile.extra.append(line)
                continue
            if section == "facts":
                fact_match = _FACT_LINE.match(line)
                if fact_match:
                    parts = [part.strip() for part in fact_match.group(2).split(" | ")]
                    record = FactRecord(parts[0])
                    for part in parts[1:]:
                        name, _, raw = part.partition("=")
                        try:
                            if name == "conf":
                                record.confidence = float(raw)
                            elif name == "seen":
                                record.seen = int(raw)
                            elif name == "last":
                                record.last_turn = int(raw)
                        except ValueError:
                            pass
                    profile.facts[fact_match.group(1).lower()] = record
                    continue
            if section == "history" and line.startswith("- "):
                profile.history.append(line[2:])
                continue
            profile.extra.append(line)
        return profile

    def _render(self, user_id: str, profile: _Profile) -> str:
        lines = [f"# User Profile: {user_id}", f"<!-- turn: {profile.turn} -->", "", "## Facts"]
        ordered = [key for key in FACT_KEYS if key in profile.facts]
        ordered += [key for key in profile.facts if key not in FACT_KEYS]
        for key in ordered:
            record = profile.facts[key]
            lines.append(
                f"- {key}: {record.value} | conf={record.confidence:.2f} | seen={record.seen} | last={record.last_turn}"
            )
        if profile.history:
            lines += ["", "## History"] + [f"- {entry}" for entry in profile.history]
        if profile.extra:
            lines += [""] + profile.extra
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def _first_sentence(text: str, limit: int) -> str:
    flat = re.sub(r"\s+", " ", text or "").strip()
    sentence = re.split(r"(?<=[.!?])\s+", flat, maxsplit=1)[0]
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence


def summarize_messages(
    messages: list[dict[str, str]],
    max_items: int = 6,
    previous_summary: str = "",
) -> str:
    """Create a compact summary of older messages.

    Heuristic and lossy on purpose: one short line per message (its first
    sentence), merged with the previous summary, keeping only the latest
    `max_items` lines so the summary itself can never grow without bound.
    """

    bullets = [line for line in (previous_summary or "").splitlines() if line.startswith("- ")]
    for message in messages:
        limit = 160 if message.get("role") == "user" else 80
        gist = _first_sentence(message.get("content", ""), limit)
        if gist:
            bullets.append(f"- {message.get('role', 'user')}: {gist}")
    return "\n".join(bullets[-max_items:])


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - recent messages are kept in full
    - when the thread exceeds `threshold_tokens`, older messages are folded into a summary
    - the number of compactions is tracked for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)
    summary_max_items: int = 8

    def _thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})  # type: ignore[union-attr]
        if self.context_tokens(thread_id) > self.threshold_tokens:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        if len(messages) <= self.keep_messages:
            return
        cut = len(messages) - self.keep_messages
        older, recent = messages[:cut], messages[cut:]
        thread["summary"] = summarize_messages(
            older,
            max_items=self.summary_max_items,
            previous_summary=str(thread["summary"]),
        )
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[call-overload]

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def context_tokens(self, thread_id: str) -> int:
        thread = self._thread(thread_id)
        total = estimate_tokens(str(thread["summary"]))
        for message in thread["messages"]:  # type: ignore[union-attr]
            total += estimate_tokens(message["content"])
        return total

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return int(self.state[thread_id]["compactions"])  # type: ignore[call-overload]
