"""Input and output guardrails.

Input:  PII is redacted before the question reaches retrieval, an LLM or the audit
        log; obvious prompt-injection attempts are refused; requests for personal
        advice are flagged so the answer carries a disclaimer.
Output: HTML is stripped, any PII is redacted, length is capped.

PII detection is pattern-based. It catches structured identifiers (emails, phone,
card, Aadhaar, PAN, SSN, and this project's policy/customer/claim IDs). It does
not detect names or street addresses; that needs an NER model such as Presidio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 2000
MAX_OUTPUT_CHARS = 2500

ADVICE_DISCLAIMER = "This summarises what the policy documents say. It is not personal financial, tax or legal advice."
INJECTION_REFUSAL = "I can only answer questions about the policy documents."


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return len(digits) >= 13 and checksum % 10 == 0


# Order matters: earlier patterns claim their spans first.
PII_PATTERNS: list[tuple[str, re.Pattern, object]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("CARD_NUMBER", re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), _luhn_ok),
    ("AADHAAR", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), None),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    ("PHONE", re.compile(r"(?<![\w+])\+?\d{10,13}\b"), None),
    ("PHONE", re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?\(?\d{2,5}\)?[\s.-]\d{3,5}(?:[\s.-]\d{3,5})?\b"), None),
    ("CLAIM_ID", re.compile(r"\bCL\d{7}\b"), None),
    ("POLICY_ID", re.compile(r"\bP\d{8}\b"), None),
    ("CUSTOMER_ID", re.compile(r"\bC\d{6}\b"), None),
]

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|system|all)\b.{0,30}"
        r"\b(instructions?|prompts?|rules|guidelines)\b",
        r"\b(reveal|show|print|repeat|output|leak)\b.{0,30}\b(system prompt|hidden instructions|your instructions)\b",
        r"\byou are now\b",
        r"\b(developer|jailbreak|dan) mode\b",
        r"\bpretend (to be|you are)\b",
    ]
]

ADVICE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bshould i\b",
        r"\bdo you (recommend|suggest|advise)\b",
        r"\bis it (worth|wise|better|a good idea)\b",
        r"\bwhich\b.{0,40}\b(should i|is best|is better|would you)\b",
        r"\bbest (fund|policy|plan|option)\b",
    ]
]


@dataclass
class PIIEntity:
    entity: str
    value: str
    start: int
    end: int


@dataclass
class InputCheck:
    text: str  # redacted, truncated
    entities: list[PIIEntity] = field(default_factory=list)
    blocked: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def advice_request(self) -> bool:
        return "advice_request" in self.flags


def find_pii(text: str) -> list[PIIEntity]:
    found: list[PIIEntity] = []
    for entity, pattern, validator in PII_PATTERNS:
        for m in pattern.finditer(text):
            if validator and not validator(m.group(0)):
                continue
            if any(m.start() < f.end and f.start < m.end() for f in found):
                continue
            found.append(PIIEntity(entity, m.group(0), m.start(), m.end()))
    return sorted(found, key=lambda e: e.start)


def redact(text: str) -> tuple[str, list[PIIEntity]]:
    entities = find_pii(text)
    for e in reversed(entities):
        text = f"{text[: e.start]}[{e.entity}]{text[e.end :]}"
    return text, entities


def check_input(text: str) -> InputCheck:
    flags: list[str] = []
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        flags.append("input_truncated")
    redacted, entities = redact(text)
    if entities:
        flags.append("pii_redacted")
    blocked = any(p.search(text) for p in INJECTION_PATTERNS)
    if blocked:
        flags.append("prompt_injection")
    if any(p.search(text) for p in ADVICE_PATTERNS):
        flags.append("advice_request")
    return InputCheck(redacted, entities, blocked, flags)


def check_output(text: str, advice_request: bool = False) -> tuple[str, list[str]]:
    flags: list[str] = []
    cleaned = re.sub(r"<[^>]{1,200}>", "", text)
    if cleaned != text:
        flags.append("html_stripped")
    cleaned, entities = redact(cleaned)
    if entities:
        flags.append("pii_redacted_in_output")
    if len(cleaned) > MAX_OUTPUT_CHARS:
        cleaned = cleaned[:MAX_OUTPUT_CHARS].rstrip() + " ..."
        flags.append("output_truncated")
    if advice_request:
        cleaned = f"{cleaned}\n\n{ADVICE_DISCLAIMER}"
        flags.append("advice_disclaimer_added")
    return cleaned, flags
