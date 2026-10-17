# Final research question — September 25, 2026

> Can jointly choosing evidence requests across related fraud alerts improve money-laundering detection under a fixed investigation budget, beyond investigating alerts individually with the same shared cache?

## The motivation

A queue can contain several alerts involving the same accounts. One history request may inform several of those cases. Some requests are useful only because they reveal what to inspect next. The system must decide where to spend the queue's limited retrieval budget.

A cache prevents paying for the same request twice. It does not necessarily make a case-local policy choose a request whose main benefit is spread across other cases. The research tests whether joint, multi-step valuation improves on that workflow—and on strong global acquisition and planning methods.

## Where shared accounts come in

Suppose alerts concern A → X, B → X, and C → Y. X is visibly shared before further investigation. Querying X may reveal Z, and querying Z may change the risk estimates for both X-related alerts. The proposed policy values the sequence across the queue.

This is a hypothesis, not a guaranteed win. X could be an ordinary merchant and the extra queries useless. Shared-account status is not a fraud label. Financial experiments use the dataset's laundering labels, hidden from the tools; benign high-overlap cases are essential.

## What changes from the old plan

The headline is now queue-level classification and acquisition, rather than finding a temporal diamond. Graph growth is a stress test. Exact finite-world policies remain diagnostics of achievable performance without clairvoyant access. Every main baseline has the same cache and released records.

## Publication research question

> Can we learn which unresolved cases an evidence request could help, including through later requests, and use that information to match dense joint planning with substantially less computation as the queue grows?

The candidate selectively computes action–case effects, estimates omitted contributions, and expands the calculation when the next request is ambiguous. It updates after new evidence connects cases. This must outperform ordinary sparse routing and efficient scalar joint models, not just an expensive dense model.

The dense per-alert value model is an inherited baseline: successor features and reward decomposition cover its central idea, while August 2026 BRiG-AFA overlaps its Bellman training recipe. The [related-work audit](RELATED_WORK_AUDIT_2026-09-25.md) identifies these precedents and remaining checks against acquisition pruning. Adaptive influence refinement is a research candidate, not a cleared novelty claim. No result currently establishes main-track readiness.

## What would reject the research claim

- Gains disappear once every baseline shares the cache and evidence.
- A coverage heuristic or global greedy method closes the useful gap.
- Existing nonmyopic acquisition reproduces the result at comparable cost.
- Scalar joint value learning matches the proposed adaptive method with less computation.
- The influence-refinement mechanism is already supplied by acquisition-pruning or factored-planning work.
- The retrieval constraint or alert overlap is artificial in the intended application.
- Financial labels can be predicted only using shortcuts or unavailable future data.
- Results fail on another financial generator or depend on duplicated cases.

The [complete plan](PROJECT_PLAN.md) supplies the audit, conditional method, baselines, 50-hour prototype, 20–24-week study, and staged $20,000 AWS/Azure budget. No implementation or experimental gains are claimed.
