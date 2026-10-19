# P8 · Assemble and submit the proposal (~4 h over Sun Oct 18 – Tue Oct 20)

> **Handout (§4.1):** submit "who is in your group" and the proposal, in NeurIPS'22 style, aiming for **2 pages excluding references**, with **no abstract**.
>
> **Output:** `proposal/proposal.pdf`, submitted by **Tue Oct 20, 2026, 11:59pm PT**.

P1–P7 left you a `proposal/notes.md` with every section in your own words, Fig. 1, and seed-averaged results. `proposal/` also holds a complete LaTeX draft. This module turns the two into your final PDF.

## 8.1 Who is in your group: confirm solo with staff

The handout says solo projects are "discouraged; please see Ed". Post on Ed **today** if you haven't. A draft:

> *Hi! I'm planning a solo project of type 1 (real-world application of GNNs): budgeted lookups for ranking a sales call list, on public RelBench data (`rel-event`, `rel-avito`; `rel-f1` for debugging). Is a solo project OK, and does the scope look right for one person? Thanks, Max Rodriguez*

Keep the post to the public project; nothing about SCOUT or any employer. In the proposal, write the group line as *"Max Rodriguez (solo)"*, adding *"confirmed with course staff on Ed"* only once they have confirmed.

## 8.2 Load the draft into Overleaf and revise it

`proposal/` already holds a complete draft:

- `proposal.tex`: every section, Tables 1–2, Fig. 1, and Eqs. (1)–(5);
- `refs.bib`: 13 entries, author lists checked against arXiv, PMLR, or AAAI;
- `neurips_2022.sty`: the official style file;
- `figures/fig1_call_list.pdf`.

The same four files are zipped at `C:\Users\maxlr\Downloads\cs224w-proposal-overleaf.zip`. This laptop has no LaTeX, so compile on **Overleaf**: **New Project → Upload Project →** the zip.

How the draft handles the style file:

- `\usepackage[final]{neurips_2022}` shows your name. Without `[final]` the PDF is anonymized for review.
- `[final]` also prints "36th Conference on Neural Information Processing Systems (NeurIPS 2022)" in the footer, which is false for a course proposal. The `\renewcommand{\@noticestring}{...}` lines right after it replace that with a CS224W line.
- `\PassOptionsToPackage{numbers, compress}{natbib}` makes citations numeric (`[1, 2]`), which saves space.

**PASS:** Overleaf compiles a PDF with your name under the title, no abstract, the CS224W footer on page 1, both tables, Fig. 1, and a numbered reference list.

**STOP:**

- `File 'neurips_2022.sty' not found`: it must sit next to `proposal.tex`, not inside a folder.
- `Citation undefined`: the bib file must be named `refs.bib`; recompile twice.
- Anything else: paste the first error from the Overleaf log.

**Then revise it section by section**, using what you built:

1. **Rewrite each paragraph in your own words** from your `proposal/notes.md` (P1–P7). Keep the facts; make the sentences yours.
2. **Replace Table 2** with your own P7 seed means. The draft's numbers came from test runs of the same code; yours will differ slightly because the sampler isn't seeded. Update every sentence in §4 that quotes them.
3. **Resolve every `% TODO`** (search the file for "TODO").
4. **Recompile and check the length:** about 2 pages before the references. If it runs over, shorten the related-work paragraph first.

## 8.3 Outline and page budget

| § | Section | Budget | Source in `notes.md` | Must contain |
|---|---|---|---|---|
| — | Title block | — | 8.1 | Your name, "(solo)" |
| 1 | Application and motivation | ~⅓ page | P1 | The user, the cost of lookups, the graph, the RQ, why GNNs |
| 2 | Dataset, task, metric | ~½ page | P2, P3, P4 | Table 1 (rows, label, horizon, splits, positive rate); public statement; metrics with *K* defined; headroom and sharing numbers; citations |
| 3 | Graph ML approach | ~¾ page | P5, P6, P7 | Model with citations; partial observation; Eqs. (1)–(6) built, (7)–(8) planned; Fig. 1; why it fits; baselines named |
| 4 | Evaluation plan and preliminary results | ~⅓ page | P7 | Frontiers and the analyses list; H1–H5 one line each; seed-averaged results, labeled |
| — | References | not counted | `refs.bib` | Everything cited, including the data |

Two pages is tight, and Fig. 1 alone takes about a third of a page. If you run over, cut in this order: the sales-analogue table (keep one sentence), Eq. (8) (describe it in words), Fig. 2 (keep the table), the baseline list (one line).

**Keep the vocabulary consistent.** Pick one word per idea and use it everywhere, including figure captions:

| Term | Meaning in this proposal |
|---|---|
| Target / prospect | A row of the entity table being scored (a user; a driver on `rel-f1`) |
| Call list *Q* | *N* targets scored together; *N* = 128 by default |
| Lookup | One costed action: `list` (pull a prospect's history) or `get` (enrich an entity) |
| Budget *B* | Total lookups for the whole list, reported as lookups per prospect |
| Stub | An entity known to exist but not yet fetched; enters the model as `[MASK]` |
| Revealed set *O* | Everything fetched so far; the model sees only *O* |
| Policy π | The rule that picks the next lookup |
| Frontier | Ranking quality vs. budget, one curve per policy |
| Full information | Every row revealed: the standard RelBench GNN, the upper end of every frontier |
| Oracle-greedy | Picks lookups using the true labels; a diagnostic ceiling, never deployable |

## 8.4 Requirement checklist

Each row points to the module whose output is your evidence. Tick each one before submitting.

| ☐ | Requirement (handout §1.3, §4.1) | Proposal § | Evidence | Module |
|---|---|---|---|---|
| ☐ | Who is in your group | Title | Ed confirmation | P8 §8.1 |
| ☐ | Application domain, "an application that a company would actually use" | 1 | Problem statement, RQ, hypotheses | [P1](P1-application-domain.md) |
| ☐ | Which dataset(s)? | 2 | `peek_data.py` output | [P2](P2-datasets-and-public.md) |
| ☐ | **Is the dataset public?** *(strict)* | 2 | Downloads with no login; HF dataset not gated; original Kaggle sources cited | [P2](P2-datasets-and-public.md) |
| ☐ | Describe dataset / task / metric; cite, especially the data | 2 | `proposal/cards/*.md`; `metrics.py` + tests; RelBench and Kaggle citations | [P3](P3-dataset-task-metric.md) |
| ☐ | Why did you choose the dataset? | 2 | Leaderboard gaps; `sharing_stats.py` output | [P4](P4-why-these-datasets.md) |
| ☐ | Which graph ML model? (cite) | 3 | `inspect_graph.py` output; `refs.bib` | [P5](P5-which-model.md) |
| ☐ | Describe the model, with figures and equations | 3 | Eqs. (1)–(8) mapped to code; `fig1_call_list.pdf` (and optional Fig. 2) | [P6](P6-describe-the-model.md) |
| ☐ | Why is the model appropriate for the dataset? | 3 | Leakage test; seed-averaged results | [P7](P7-why-appropriate.md) |
| ☐ | Novelty or comprehensive analyses (the type-1 bar, §1) | 3–4 | Claim boundaries in [RELATED_WORK.md](../RELATED_WORK.md); analyses list in [PROJECT_PLAN §6](../PROJECT_PLAN.md#6-experiments-and-analyses) | P1, P7 |
| ☐ | Format: NeurIPS'22 style, ~2 pages excluding references, no abstract | all | `proposal.pdf` | P8 §8.2 |
| ☐ | Preliminary results *(not required; strengthens it)* | 4 | Sharing statistics; LightGBM and RDL validation numbers with seeds stated | P4, P7 |

## 8.5 Timeline

Per handout §4.1, the proposal is 13.33% of the project grade (4% of the course), the milestone is credit/no credit, and the final submission is 80% of the project (24% of the course).

| When | Do |
|---|---|
| **Sat Oct 17** | P0–P7. End of day: `notes.md` has every section, both figures exist, the seed runs are averaged |
| **Sun Oct 18** | **Revise.** Load the draft in Overleaf (8.2); rewrite it section by section from `notes.md`; put your P7 numbers in Table 2; compile after each section |
| **Mon Oct 19** | **Polish.** Cut to 2 pages; check every number against its source (8.6); read the claim boundaries in RELATED_WORK.md; have someone read §1 cold |
| **Tue Oct 20** | **Submit**, on the site the handout names, by 11:59pm PT. Aim to submit in the afternoon so an upload problem isn't fatal |
| Tue Nov 10 | Milestone: code plus report draft (build track B1–B3) |
| Thu Dec 10 | Final: private Medium draft link, Medium account name, and Colab |

## 8.6 Before you submit

- [ ] Your name is under the title; the footer says CS224W, not NeurIPS 2022.
- [ ] No abstract.
- [ ] About 2 pages before the references; figures readable when printed in grayscale.
- [ ] **Every number has a source you produced or cite:** cards (P3), sharing statistics (P4), leaderboard gaps (P4, cited), LightGBM and RDL results (P7).
- [ ] **Every preliminary number says what it is:** `rel-f1` validation (588 rows), how many seeds, and "rows revealed at random". No single-seed number is presented as a finding.
- [ ] Claims follow RELATED_WORK.md: Zhou (2026) cited as the closest work; mask training and random-removal robustness cited as known, not claimed.
- [ ] Nothing about SCOUT internals, Bonterra, interviews, or customer data. The proposal and the repo are both public.
- [ ] References include RelBench, Fey et al. 2024, Hamilton et al. 2017, Fey & Lenssen 2019, and Zhou (2026). The two Kaggle sources are credited (the draft uses a footnote). Add GloVe (Pennington et al. 2014) if the text mentions text embeddings.

## 8.7 Commit

Download the compiled PDF from Overleaf into `proposal/proposal.pdf`, then:

```powershell
git add proposal
git status      # expect proposal/ files only: no outputs/, .venv, or *.local.json
git commit -m "P8: proposal source, figures, and PDF"
git push
```

**Self-check**

1. Which numbers in your proposal are preliminary, and which words in the text say so?
2. Which of your contributions overlaps Zhou (2026), and what is the defensible difference?
3. Your draft runs to 2.5 pages. What do you cut first?

**Next:** the build track starts after Oct 20: [B1 · Cloud GPU and full-information baselines](B1-cloud-gpu-and-baseline.md). The full list is in the [module index](README.md).
