# Publication readiness report

## Decision

**NOT YET READY FOR JOURNAL SUBMISSION.**

The manuscript is structurally complete and the all-hazard validation section is
traceable to committed artifacts. The central Version 2 cyclone claims are not
yet publication-verifiable because their Git-ignored local outputs are absent.
Submitting before resolving this gap would make the title and event-aware
contribution stronger than the archived evidence.

## Completed quality gates

- Twenty-seven references have verified bibliographic identities. Twenty-four
  DOI records were checked through Crossref; JMLR and arXiv records were checked
  at their permanent pages.
- The Delforge journal metadata and accessible full author manuscript were
  reviewed. Talan's publisher full text and Crossref abstract were reviewed.
- All-hazard sample counts, validation metrics, class reports, and confusion
  matrices were recovered from committed JSON files.
- Figures are generated only from the committed preprocessing report and
  validation artifacts.
- Project results and literature results are labelled separately.
- No reserved-period number is reported.
- The abstract states the missing cyclone-evidence limitation and does not claim
  live prediction, algorithmic novelty, acceptance, or operational readiness.
- Citation numbering is consecutive and every numbered bibliography entry is
  cited in the manuscript.

## Required before submission

1. **Archive the cyclone experiment evidence.** Recover the cutoff audit,
   baseline/enriched comparison, feature-block ablation, magnitude ablation, and
   algorithm-comparison JSON outputs. Verify hashes, denominators, class support,
   cohort fingerprints, SID counts, confusion matrices, and paired intervals.
2. **Resolve event dependence beyond cyclones.** Audit whether country records
   from the same non-cyclone physical event cross temporal boundaries. If they
   do, redesign the split or explicitly narrow the claim.
3. **Freeze the development protocol.** The same 2020–2022 validation cohort has
   supported several comparisons. Specify the final candidate and stop tuning
   before any independent evaluation.
4. **Decide the reserved-period policy.** Establish who authorized the existing
   final-test artifact, verify its hashes and execution record, and decide
   prospectively whether it is admissible. Do not rerun or select from test
   results.
5. **Confirm data rights and provenance.** Record EM-DAT access date/version,
   redistribution terms, IBTrACS version/checksum, and exact local source hashes.
6. **Complete author declarations.** Replace author, affiliation, funding,
   ethics, conflict-of-interest, and CRediT placeholders with verified details.
7. **Choose a target journal and apply its template.** Reformat length,
   references, figure resolution, highlights, declarations, and data statement
   only after journal selection.
8. **Independent reproduction.** A second environment should regenerate the
   aggregate development artifacts from authorized sources and compare hashes or
   documented tolerances.

## Claims that must remain excluded

- 90% performance, state-of-the-art performance, or superiority to published
  systems;
- validated real-time or operational GDACS impact prediction;
- causal explanations for model performance;
- treating GDACS exposure as observed affected population;
- directly comparing classification F1 with Talan's regression RMSLE or
  Zhang's spatial classification scores;
- claiming that removal of magnitude makes the remaining features live-safe;
- any cyclone confidence interval or algorithm ranking until its artifact is
  recovered.

## Recommended submission path

The shortest defensible path is to restore and audit the existing local cyclone
outputs rather than run new searches. If those artifacts pass, update Sections
7.3–8.2, generate the preregistered ablation/algorithm figures, and repeat the
citation/metric audit. If they fail, submit a narrower all-hazard temporal
validation paper with a title that does not imply completed IBTrACS comparison.
