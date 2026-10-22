# Version 2 — Cyclone ML Research Checkpoint
Date: 2026-10-22

## Real datasets
- EM-DAT: historical disaster events and impact outcomes.
- NOAA IBTrACS: historical tropical-cyclone observations.

## Locally observed matching results
- EM-DAT country records evaluated: 1,249
- High-confidence country matches: 608
- Unique physical cyclones: 386
- Event-country identifier linkage: 1,249/1,249 passed

## Implementation
- Conservative EM-DAT–IBTrACS matching implemented.
- Complete event dates and identifiers validated.
- Cutoff-safe feature engineering and cyclone validation scripts committed.
- 44 automated ML tests passed locally.

## Pending research
- Run the cutoff eligibility and missingness audit.
- Train baseline and IBTrACS-enriched models locally.
- Compare validation accuracy, macro F1 and per-class metrics.
- Keep the 2023–2025 test period excluded from model selection.

## Scientific limitations
Matching coverage is not prediction accuracy. IBTrACS contains retrospective
best-track information, and matching a cyclone does not independently prove
country-level impact. No Version 2 performance improvement is claimed yet.
