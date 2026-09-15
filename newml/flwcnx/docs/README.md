# Documentation index

Read in this order if you are new to the project.

## The deliverables

| file | what it is |
|---|---|
| [`paper/paper.md`](paper/paper.md) | The research paper. A reproduction and evaluation study. The five findings are in section 1.1 and the prior-art accounting is in section 6. |
| [`paper/report.md`](paper/report.md) | The project report. The whole system, the four objectives, how the data was obtained and what that cost. |
| [`deliverables/review-1-deck.html`](deliverables/review-1-deck.html) | Review presentation. |
| [`deliverables/review-1-content.md`](deliverables/review-1-content.md) | Speaker content for the deck. |

The demonstration is `scripts/run_demo.py`; it writes a self-contained HTML page
alongside a per-decision audit trail.

## The honest parts, which are the point

| file | what it is |
|---|---|
| [`limitations.md`](limitations.md) | Written while building, not retrofitted at the end. Section 1 predicted the drift failure before it was measured, and was rewritten around the measurement afterwards. |
| [`novelty-review.md`](novelty-review.md) | An independent literature check, commissioned against the brief below. It withdrew the project's methods claim, found a paper the internal check missed, and corrected an author attribution error. |
| [`novelty-assessment.md`](novelty-assessment.md) | The internal check that preceded it, kept because the review corrects it. |
| [`literature-review-brief.md`](literature-review-brief.md) | The brief the review was run against. Written adversarially: the instruction was not to conclude novelty from failure to find prior work. |

Two claims made during this project were withdrawn after being checked. Both
withdrawals are documented in place rather than edited away:

- the **methods claim** (`novelty-assessment.md`, `novelty-review.md`);
- the **offset-spread predictor** (`../results/summary/gate-negative-result.md`).

## Data provenance

| file | what it is |
|---|---|
| [`data.md`](data.md) | The StarNet schema, pinned against their loader rather than the paper text, including the places the two disagree. |
| [`data-access.md`](data-access.md) | Every route tried while the traces were unavailable, the workaround taken, and what it cost. Ends with how the files finally arrived, mislabelled, and how they were identified from their contents. |
| [`supplied-dataset.md`](supplied-dataset.md) | Inspection of the dataset supplied with the brief, before any feature code was written against it. Carries its own in-place correction. |
| [`wetlinks-full.md`](wetlinks-full.md) | The full WetLinks release, which the supplied files were a subset of. |

## Progress and process

| file | what it is |
|---|---|
| [`progress.md`](progress.md) | One entry per phase. Nothing is recorded that was not actually run. |
| [`references.bib`](references.bib) | 40 entries. Every one is cited somewhere in the code or the deliverables. |
| [`email-draft.md`](email-draft.md) | The message sent to the StarNet authors while the traces were unavailable. |

## A note on how to read the results

Every number in `results/summary/` comes from a run that wrote a `result.json`
with its configuration snapshot beside it. Figures and tables are generated from
those files and recompute nothing, so any figure can be traced back to the exact
settings that produced it.

Two of the summary files disagree with each other:
`starnet-regime-grid.md` and `osnabruck-capacity.md` reach opposite conclusions
about whether regime conditioning helps. That is the honest state of the
evidence and both are kept. Neither is the real one.
