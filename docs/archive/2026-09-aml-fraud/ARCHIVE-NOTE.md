# Archived: AML joint-evidence-acquisition plans (September 2026)

These documents planned a money-laundering alert-triage study on IBM AMLworld
(HI-Small). They are superseded by [the active plan](../../PROJECT_PLAN.md),
which keeps the core research idea — choosing which evidence to fetch for a
*batch* of related decisions under a budget — and moves it to sales
prospecting on public RelBench databases.

What carried over:

- Budgeted, batch-level acquisition where one lookup can inform several targets.
- Equal-information fairness: every policy shares the same cache and budget.
- Strong baselines (coverage, per-target uncertainty, oracle-greedy) before claiming novelty.
- Strict temporal cutoffs; labels never enter observations.
- The related-work audit (BRiG-AFA, CAMA, ACO, collective active inference, successor features).

What was dropped:

- IBM AMLworld as the primary dataset, the custom investigation gateway, and
  the hand-built finite "tiny worlds" (RelBench `rel-f1` now plays the
  small-debugging role).
- The 20–24-week, $20k program framing.

The raw IBM files under `data/raw/` are Git-ignored and no longer used.
