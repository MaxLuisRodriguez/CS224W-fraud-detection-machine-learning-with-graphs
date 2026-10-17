# When More Transactions Hide the Evidence

Research project for CS224W: avoidable search failure in budgeted graph investigation.

**Question:** Can reasoning about shared accounts reduce avoidable evidence-search
failures as transaction graphs grow, beyond strong simple and nonmyopic policies
at the same query budget?

The first experiment compares evidence discovery against an exact reference that
knows the graph distribution but sees only legal query responses. This distinguishes
loss forced by limited information from additional policy failure. Ordinary graph
growth, symbolic witnesses, and strong simple policies come before learned methods.

The [active project plan](docs/PROJECT_PLAN.md) specifies the experiment, comparisons,
44-hour pilot, financial-data gate, and publication criteria. The
[one-page framing](docs/RESEARCH_QUESTION.md) states the question and motivation;
the [decision record](docs/RESEARCH_DECISION.md) explains the September 24 revision.

For another chat: [self-contained handoff](docs/RESEARCH_AUDIT_HANDOFF.md).
For a portable snapshot: [September 24 plan](docs/PROJECT_PLAN_2026-09-24.md)
or [plain text](docs/PROJECT_PLAN_2026-09-24.txt).

## Status

Planning and data only, ready for implementation from scratch. Public IBM HI-Small
transaction and pattern files, with download metadata and checksums, remain in
`data/raw/`. No model results, validated robustness claims, or conference-ready
contribution have been established.

The unfinished implementation scaffold and its Python environment were moved to
the sibling folder `../CS224W-fraud-detection-machine-learning-with-graphs-scaffold-archive-20260923/`.
Its `ARCHIVE.md` records what was moved and how to recover it. No files were deleted.

Raw data, local environments, and experiment artifacts are excluded from Git.
Dataset labels and evaluator-only annotations must never enter investigation-tool
observations.
