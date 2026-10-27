"""Answer generation. The LLM answerers run against stub clients, so no network or key is needed."""

from dataclasses import replace
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.answer import (
    NOT_FOUND_MESSAGE,
    AnthropicAnswerer,
    AzureOpenAIAnswerer,
    ExtractiveAnswerer,
    PolicyQA,
    build_answerer,
)
from app.config import Settings

# --- extractive -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question, expected, source",
    [
        ("What is the surrender charge in year 2 of the term life policy?", "8%", "term_life_insurance_policy.pdf"),
        ("How long is the free-look period?", "15 days", "term_life_insurance_policy.pdf"),
        ("What is the premium allocation charge in the first year of the ULIP?", "8%", "ulip_insurance_policy.pdf"),
        ("What is the waiting period for maternity benefits?", "2 years", "health_insurance_policy.pdf"),
        ("Under which tax sections are tax benefits available?", "80C", "general_terms_conditions.pdf"),
    ],
)
def test_extractive_answers_with_citation(retriever, question, expected, source):
    answer, _ = PolicyQA(retriever).ask(question)
    assert answer.found
    assert expected in answer.text
    assert answer.provider == "extractive"
    assert answer.citations and answer.citations[0].source_file == source
    assert "[1]" in answer.text


def test_extractive_says_not_found_for_off_topic_question(retriever):
    answer, _ = PolicyQA(retriever).ask("What is the capital of France?")
    assert not answer.found
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == []


def test_extractive_quotes_come_from_the_cited_chunk(retriever):
    answer, hits = PolicyQA(retriever).ask("How many days of post-hospitalisation expenses are covered?")
    for c in answer.citations:
        chunk_text = " ".join(hits[c.n - 1].chunk.lines)
        assert c.quote.split(": ")[-1] in chunk_text


# --- Anthropic (stub client) --------------------------------------------------------------


class StubAnthropic:
    def __init__(self, response=None, error=None):
        self.calls = []
        self._response, self._error = response, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


def text_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def cite(index, quote):
    return SimpleNamespace(type="search_result_location", search_result_index=index, cited_text=quote)


def claude_response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), *blocks],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=900, output_tokens=120),
        model="claude-opus-5-5",
    )


def test_claude_request_shape_and_citation_parsing(retriever):
    stub = StubAnthropic(
        claude_response(
            [
                text_block("In year 2 the surrender charge is "),
                text_block("8% of total premiums paid", [cite(0, "Year 2: 8% of total premiums paid")]),
                text_block("."),
            ]
        )
    )
    qa = PolicyQA(retriever, AnthropicAnswerer("claude-opus-5-5", client=stub))
    answer, hits = qa.ask("What is the surrender charge in year 2 of the term life policy?")

    request = stub.calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request["betas"]
    assert request["output_config"] == {"effort": "medium"}
    assert "temperature" not in request
    content = request["messages"][0]["content"]
    search_results = [b for b in content if b["type"] == "search_result"]
    assert len(search_results) == len(hits)
    assert all(b["citations"] == {"enabled": True} for b in search_results)
    assert content[-1]["text"].startswith("Question:")

    assert answer.found
    assert answer.text == "In year 2 the surrender charge is 8% of total premiums paid [1]."
    assert answer.citations[0].quote == "Year 2: 8% of total premiums paid"
    assert answer.citations[0].source_file == "term_life_insurance_policy.pdf"
    assert answer.usage["input_tokens"] == 900


def test_claude_not_found(retriever):
    stub = StubAnthropic(claude_response([text_block("NOT_FOUND")]))
    answer, _ = PolicyQA(retriever, AnthropicAnswerer("claude-opus-5-5", client=stub)).ask(
        "Is there a guaranteed minimum return on the ULIP Equity Growth Fund?"
    )
    assert len(stub.calls) == 1  # retrieval found something, so the model was asked
    assert not answer.found
    assert answer.text == NOT_FOUND_MESSAGE


def test_no_llm_call_when_nothing_relevant_is_retrieved(retriever):
    stub = StubAnthropic(claude_response([text_block("should not be used")]))
    answer, _ = PolicyQA(retriever, AnthropicAnswerer("claude-opus-5-5", client=stub)).ask("zebra xylophone quokka")
    assert not answer.found
    assert stub.calls == []


def test_uncited_claude_answer_falls_back_to_extractive(retriever):
    stub = StubAnthropic(claude_response([text_block("It is 8%, trust me.")]))
    answer, _ = PolicyQA(retriever, AnthropicAnswerer("claude-opus-5-5", client=stub)).ask(
        "What is the surrender charge in year 2 of the term life policy?"
    )
    assert answer.provider == "extractive"
    assert answer.fallback_reason == "llm_answer_without_citations"
    assert "8%" in answer.text


def test_claude_refusal_falls_back(retriever):
    stub = StubAnthropic(claude_response([], stop_reason="refusal"))
    answer, _ = PolicyQA(retriever, AnthropicAnswerer("claude-opus-5-5", client=stub)).ask(
        "How long is the free-look period?"
    )
    assert answer.provider == "extractive"
    assert "refusal" in answer.fallback_reason


def test_claude_api_error_falls_back(retriever):
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    stub = StubAnthropic(error=error)
    answer, _ = PolicyQA(retriever, AnthropicAnswerer("claude-opus-5-5", client=stub)).ask(
        "How long is the free-look period?"
    )
    assert answer.found and "15 days" in answer.text
    assert answer.fallback_reason.startswith("llm_error: APIConnectionError")


def test_citation_to_unknown_search_result_is_ignored(retriever):
    stub = StubAnthropic(claude_response([text_block("Something", [cite(99, "made up")])]))
    answer, _ = PolicyQA(retriever, AnthropicAnswerer("claude-opus-5-5", client=stub)).ask(
        "How long is the free-look period?"
    )
    assert answer.fallback_reason == "llm_answer_without_citations"


# --- Azure OpenAI (stub client) ------------------------------------------------------------


class StubAzure:
    def __init__(self, text):
        self.calls = []
        message = SimpleNamespace(content=text)
        self._response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=800, completion_tokens=40)
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_azure_keeps_valid_citations_and_drops_invented_ones(retriever):
    stub = StubAzure("The free-look period is 15 days [1]. See also [9].")
    answerer = AzureOpenAIAnswerer("https://example.openai.azure.com", "key", "gpt-deploy", client=stub)
    answer, hits = PolicyQA(retriever, answerer).ask("How long is the free-look period?")
    assert answer.provider == "azure_openai/gpt-deploy"
    assert answer.text == "The free-look period is 15 days [1]. See also."
    assert [c.n for c in answer.citations] == [1]
    assert stub.calls[0]["model"] == "gpt-deploy"
    assert "[1]" in stub.calls[0]["messages"][1]["content"]


def test_azure_not_found():
    stub = StubAzure("NOT_FOUND")
    answerer = AzureOpenAIAnswerer("https://example.openai.azure.com", "key", "gpt-deploy", client=stub)
    assert not answerer.answer("q", []).found


# --- provider selection ---------------------------------------------------------------------


def test_auto_provider_without_keys_is_extractive(retriever, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(build_answerer(Settings(), retriever), ExtractiveAnswerer)


def test_auto_provider_prefers_anthropic_when_key_set(retriever, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    answerer = build_answerer(Settings(), retriever)
    assert isinstance(answerer, AnthropicAnswerer)
    assert answerer.name == "anthropic/claude-opus-5-5"


def test_azure_provider_requires_settings(retriever):
    with pytest.raises(ValueError):
        build_answerer(replace(Settings(), llm_provider="azure_openai"), retriever)


def test_unknown_provider_rejected(retriever):
    with pytest.raises(ValueError):
        build_answerer(replace(Settings(), llm_provider="nope"), retriever)
