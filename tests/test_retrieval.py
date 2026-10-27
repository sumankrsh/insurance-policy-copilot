import pytest

from app.retrieval import normalize


def test_every_section_becomes_a_chunk(chunks):
    assert len(chunks) == 19
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.lines and c.section and c.page >= 1 for c in chunks)
    assert {c.source_file for c in chunks} == {
        "term_life_insurance_policy.pdf",
        "ulip_insurance_policy.pdf",
        "health_insurance_policy.pdf",
        "general_terms_conditions.pdf",
    }


def test_section_continuing_on_next_page_keeps_its_heading(chunks):
    exclusions = [
        c for c in chunks if c.chunk_id.startswith("term_life_insurance_policy") and "Exclusions" in c.section
    ]
    assert len(exclusions) == 1
    assert exclusions[0].page == 2
    assert exclusions[0].section == "Section 4: Exclusions"


def test_table_rows_stay_on_one_line(chunks):
    funds = next(c for c in chunks if c.section == "Section 2: Fund options")
    assert "Equity Growth Fund | High | 80-100% Equity | 1.35% p.a." in funds.lines


def test_bullets_are_stripped(chunks):
    assert not any(line.startswith("\x7f") for c in chunks for line in c.lines)


@pytest.mark.parametrize(
    "question, expected_chunk",
    [
        ("What is the surrender charge in year 2 of the term life policy?", "term_life_insurance_policy:p1:section-3"),
        ("How long is the free-look period?", "term_life_insurance_policy:p3:section-6"),
        ("What is the waiting period for pre-existing diseases?", "health_insurance_policy:p1:section-2"),
        ("What is the management fee for the Equity Growth Fund?", "ulip_insurance_policy:p1:section-2"),
        ("Which tax sections give tax benefits?", "general_terms_conditions:p2:regulatory-compliance"),
        ("Is cosmetic treatment covered by the health policy?", "health_insurance_policy:p1:section-3"),
    ],
)
def test_top_hit(retriever, question, expected_chunk):
    hits = retriever.search(question, k=3)
    assert hits[0].chunk.chunk_id.startswith(expected_chunk)
    assert hits[0].score >= hits[-1].score


def test_normalize_maps_number_words_and_drops_stop_words():
    assert normalize("What is the charge in the first year?") == "charge 1 year"
    assert normalize("Section 80C") == "section 80c"


def test_search_returns_at_most_k(retriever):
    assert len(retriever.search("premium", k=2)) == 2
