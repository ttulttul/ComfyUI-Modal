"""Normalize model reasoning and final-response channels across LLM backends."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)

_QWEN3_REASONING_PARSER = "qwen3"
_NO_REASONING_PARSER = "none"
_ARCHITECTURE_REASONING_PARSERS = {
    "Qwen3_5ForConditionalGeneration": _QWEN3_REASONING_PARSER,
}


class ReasoningProfile(Protocol):
    """Describe the profile fields needed to select a reasoning parser."""

    architecture: str
    reasoning_parser: str


class TokenDecoder(Protocol):
    """Describe the tokenizer operations used for token-aware extraction."""

    def get_vocab(self) -> dict[str, int]:
        """Return the token-to-id vocabulary."""

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        """Decode generated token IDs into text."""


@dataclass(frozen=True)
class ReasoningOutput:
    """Hold a clean final response and its separately extracted reasoning."""

    response: str
    reasoning: str
    reasoning_tokens: int
    parser: str


class ReasoningOutputParser(Protocol):
    """Define backend-neutral reasoning extraction from one generated candidate."""

    parser_name: str
    requires_boundary_tokens: bool

    def extract(
        self,
        text: str,
        token_ids: Sequence[int],
        *,
        native_reasoning: str | None = None,
    ) -> ReasoningOutput:
        """Separate reasoning from the final response."""


def reasoning_parser_name(profile: ReasoningProfile) -> str:
    """Return the configured model-output parser, with architecture fallback."""
    configured = profile.reasoning_parser.strip().lower()
    if configured:
        return configured
    return _ARCHITECTURE_REASONING_PARSERS.get(
        profile.architecture,
        _NO_REASONING_PARSER,
    )


def reasoning_chat_template_kwargs(profile: ReasoningProfile) -> dict[str, bool]:
    """Return chat-template settings that agree with the selected parser state."""
    if reasoning_parser_name(profile) == _QWEN3_REASONING_PARSER:
        return {"enable_thinking": True}
    return {}


class PassthroughReasoningParser:
    """Return ordinary non-reasoning model output unchanged."""

    parser_name = _NO_REASONING_PARSER
    requires_boundary_tokens = False

    def extract(
        self,
        text: str,
        token_ids: Sequence[int],
        *,
        native_reasoning: str | None = None,
    ) -> ReasoningOutput:
        """Use a native reasoning channel when an engine provides one."""
        del token_ids
        return ReasoningOutput(
            response=text.strip(),
            reasoning=(native_reasoning or "").strip(),
            reasoning_tokens=0,
            parser="native" if native_reasoning is not None else self.parser_name,
        )


class Qwen3ReasoningParser:
    """Split Qwen3 reasoning delimited by its chat-template thinking tokens."""

    parser_name = _QWEN3_REASONING_PARSER
    requires_boundary_tokens = True
    _start_token = "<think>"
    _end_token = "</think>"

    def __init__(self, tokenizer: TokenDecoder) -> None:
        """Retain the model tokenizer so special boundary tokens survive decoding."""
        self._tokenizer = tokenizer

    def _decode(self, token_ids: Sequence[int]) -> str:
        """Decode one channel while removing model control tokens."""
        return str(
            self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ).strip()

    def _boundary_ids(self) -> tuple[int | None, int | None]:
        """Return exact reasoning boundary IDs when the tokenizer exposes them."""
        vocabulary = self._tokenizer.get_vocab()
        return vocabulary.get(self._start_token), vocabulary.get(self._end_token)

    def _extract_from_tokens(
        self,
        token_ids: Sequence[int],
    ) -> ReasoningOutput | None:
        """Split at the exact end token without mistaking user-visible text for markup."""
        start_id, end_id = self._boundary_ids()
        if end_id is None:
            return None
        generated_ids = list(token_ids)
        try:
            boundary_index = generated_ids.index(end_id)
        except ValueError:
            reasoning_ids = generated_ids
            response_ids: list[int] = []
        else:
            reasoning_ids = generated_ids[:boundary_index]
            response_ids = generated_ids[boundary_index + 1 :]
            while response_ids and response_ids[0] == end_id:
                response_ids = response_ids[1:]
        if start_id is not None and start_id in reasoning_ids:
            reasoning_ids = reasoning_ids[reasoning_ids.index(start_id) + 1 :]
        return ReasoningOutput(
            response=self._decode(response_ids),
            reasoning=self._decode(reasoning_ids),
            reasoning_tokens=len(reasoning_ids),
            parser=self.parser_name,
        )

    def _extract_from_text(self, text: str) -> ReasoningOutput:
        """Fall back to Qwen's documented delimiters when token IDs are unavailable."""
        raw_text = text.strip()
        if self._end_token in raw_text:
            reasoning, response = raw_text.split(self._end_token, maxsplit=1)
        else:
            reasoning, response = raw_text, ""
        if self._start_token in reasoning:
            reasoning = reasoning.split(self._start_token, maxsplit=1)[1]
        return ReasoningOutput(
            response=response.strip(),
            reasoning=reasoning.strip(),
            reasoning_tokens=0,
            parser=self.parser_name,
        )

    def extract(
        self,
        text: str,
        token_ids: Sequence[int],
        *,
        native_reasoning: str | None = None,
    ) -> ReasoningOutput:
        """Prefer an engine-native channel, then exact tokens, then text delimiters."""
        if native_reasoning is not None:
            return ReasoningOutput(
                response=text.strip(),
                reasoning=native_reasoning.strip(),
                reasoning_tokens=0,
                parser="native",
            )
        token_result = self._extract_from_tokens(token_ids) if token_ids else None
        return token_result or self._extract_from_text(text)


def create_reasoning_parser(
    profile: ReasoningProfile,
    tokenizer: TokenDecoder,
) -> ReasoningOutputParser:
    """Create the model-aware output parser shared by every inference backend."""
    parser_name = reasoning_parser_name(profile)
    logger.debug(
        "Selected Modal LLM reasoning parser=%s architecture=%s.",
        parser_name,
        profile.architecture or "unspecified",
    )
    if parser_name == _NO_REASONING_PARSER:
        return PassthroughReasoningParser()
    if parser_name == _QWEN3_REASONING_PARSER:
        return Qwen3ReasoningParser(tokenizer)
    raise ValueError(f"Unsupported Modal LLM reasoning parser {parser_name!r}.")


__all__ = [
    "PassthroughReasoningParser",
    "Qwen3ReasoningParser",
    "ReasoningOutput",
    "ReasoningOutputParser",
    "create_reasoning_parser",
    "reasoning_chat_template_kwargs",
    "reasoning_parser_name",
]
