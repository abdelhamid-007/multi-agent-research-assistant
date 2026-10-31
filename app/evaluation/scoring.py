"""Weights and thresholds of the automated evaluation."""

WEIGHTS = {
    "factual_accuracy": 0.30,
    "citation_reliability": 0.25,
    "research_coverage": 0.20,
    "logical_structure": 0.15,
    "recommendations": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

LABELS = {
    "factual_accuracy": "Factual accuracy",
    "citation_reliability": "Citation reliability",
    "research_coverage": "Research coverage",
    "logical_structure": "Logical structure",
    "recommendations": "Recommendations",
}

# A draft can only be approved with an overall score of at least PASS_SCORE,
# no dimension below MIN_DIMENSION_SCORE, and no blocking flag.
PASS_SCORE = 70.0
MIN_DIMENSION_SCORE = 40.0

# Partial credit for a claim, by how well the sources support it.
CLAIM_CREDIT = {
    "supported": 1.0,
    "weak": 0.5,
    "uncited_supported": 0.5,
    "uncited_stat": 0.5,
    "miscited": 0.25,
    "unsupported": 0.0,
    "unsupported_critical": 0.0,
}

MIN_REPORT_WORDS = 150
MAX_REPORT_WORDS = 950
THIN_SECTION_WORDS = 8
SUPPORT_OVERLAP = 0.35          # share of a cited claim's content words found in its sources
UNCITED_OVERLAP = 0.5           # same, for an uncited claim against all sources
FINDING_COVERED_OVERLAP = 0.6   # share of a finding's words that must appear in the report
SOURCE_UTILIZATION_TARGET = 0.5  # citing half of the sources counts as full utilisation
