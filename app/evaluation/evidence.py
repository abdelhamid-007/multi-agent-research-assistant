"""Text helpers and an index of what the retrieved sources actually say."""
import re

STOPWORDS = frozenset("""
about above after again against also among because been before being between both
cannot could does doing down during each from further have having here into itself
just like more most much must only other over same should some such than that their
them then there these they this those through under until very were what when where
which while will with within without would your across can may might many often
""".split())

_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ed", "es", "s", "ly")
_NUMBER = re.compile(
    r"(?<![\w.])([$€£])?\s?(\d[\d,]*(?:\.\d+)?)\s?(%|percent\b|x\b|k\b|m\b|bn\b|million\b|billion\b|trillion\b|thousand\b)?",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[\d+\]")


def strip_citations(text):
    return re.sub(r"\s+([.,;:!?])", r"\1", _CITATION.sub("", text or ""))


def stem(word):
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text):
    """Stemmed content words (4+ letters, no stopwords) of a text."""
    words = re.findall(r"[a-z0-9]+", strip_citations(text).lower())
    return {stem(w) for w in words if len(w) >= 4 and w not in STOPWORDS}


def overlap(claim_tokens, evidence_tokens):
    """Share of the claim's content words that occur in the evidence."""
    return len(claim_tokens & evidence_tokens) / len(claim_tokens) if claim_tokens else 1.0


def _canonical(raw):
    number = raw.replace(",", "")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def numbers(text, statistics_only=False):
    """Canonical numbers in a text.

    With statistics_only, keeps figures a report must back with evidence:
    anything with a currency sign, % or magnitude word, or a bare value of 10
    or more that is not a year. Small counts ("3 steps") and years are ignored.
    """
    found = []
    for match in _NUMBER.finditer(strip_citations(text)):
        currency, raw, unit = match.groups()
        value = _canonical(raw)
        if statistics_only and not (currency or unit):
            try:
                number = float(value)
            except ValueError:
                continue
            is_year = number.is_integer() and 1900 <= number <= 2100
            if number < 10 or is_year:
                continue
        found.append(value)
    return found


class EvidenceIndex:
    """What the verified sources say, per source and overall.

    Numbers are checked against the tool-provided text only (title and
    snippet); words are also matched against the findings extracted from
    each source.
    """

    def __init__(self, research):
        self.source_ids = {s.id for s in research.sources}
        self._source_text, self._finding_text = {}, {}
        for s in research.sources:
            self._source_text[s.id] = f"{s.title} {s.snippet}"
            self._finding_text[s.id] = ""
        for f in research.findings:
            for i in f.source_ids:
                self._finding_text[i] += f" {f.claim} {f.evidence}"
        self._tokens = {i: tokens(self._source_text[i] + self._finding_text[i]) for i in self.source_ids}
        self._numbers = {i: set(numbers(self._source_text[i])) for i in self.source_ids}
        self.all_tokens = set().union(*self._tokens.values()) if self._tokens else set()
        self.all_numbers = set().union(*self._numbers.values()) if self._numbers else set()

    def tokens_for(self, ids):
        return set().union(*(self._tokens[i] for i in ids if i in self._tokens)) if ids else set()

    def numbers_for(self, ids):
        return set().union(*(self._numbers[i] for i in ids if i in self._numbers)) if ids else set()
