# Worth the Lookup

**Learning what to research before the call: budgeted relational retrieval for call-list prioritization.** Stanford CS224W project (Fall 2026), **project type 1: Real-world application of GNNs** (blog post + Colab).

A sales rep's call list is a graph. Prospects share accounts, events, colleagues, and activity, so one lookup can sharpen several predictions. Lookups cost money (enrichment credits), tokens (LLM research agents), and time. This project asks:

> Given a call list of *N* prospects and a budget of *B* lookups, which lookups should a research agent make so that a graph neural network ranks the list best?

- **Data:** public [RelBench](https://relbench.stanford.edu/) relational databases (auto-downloaded; nothing collected by hand).
- **Models:** PyTorch Geometric heterogeneous temporal GNNs trained to work under partial retrieval, plus batch-aware acquisition policies.
- **Compute:** AWS (EC2 GPU + S3) in `us-west-2`.

## Documents

- [Project plan](docs/PROJECT_PLAN.md): audit of the previous plan, problem, method, experiments, rubric map, AWS budget, schedule to the Oct 20 / Nov 10 / Dec 10 deadlines.
- [Build modules](docs/modules/README.md): step-by-step build with PASS/STOP checks, starting with [Module 1: orientation and setup](docs/modules/01-orientation-and-setup.md).
- [Related work](docs/RELATED_WORK.md): closest prior work and what remains open.
- [Archive](docs/archive/2026-09-aml-fraud/ARCHIVE-NOTE.md): the superseded September AML/fraud plans.

## Status

Planning complete. First data finding: on random 128-prospect call lists, 34% of events are shared on `rel-event` but only 0.1% of ads on `rel-avito`. No model results yet.
