# TakeMeter — Planning

A fine-tuned classifier that scores discourse quality on an NBA community by sorting comments into three kinds of "take." This document is the design thinking behind the project: the labels, the edge-case rules, the data plan, the metrics, and the AI tool plan. It was written before any data was collected.

## 1. Community

**Choice: r/nba (and adjacent NBA discussion threads).**

NBA Reddit is a good fit because the discourse is high-volume, text-heavy, and varies enormously in quality within the *same thread*. A post-game thread will contain, side by side, a 200-word breakdown of a defensive scheme, a one-line "X is the best player alive and it's not close," and a pure "LETS GOOOO." The "hot take vs. real analysis" distinction is not something I'm imposing from outside — it is a thing the community itself argues about constantly ("that's a hot take," "actually cooking with this analysis"). That makes the labels grounded in how participants actually talk, which is the requirement that matters most.

It is also trivial to collect: Reddit exposes public thread JSON with no authentication, so 200+ real comments are a few requests away.

## 2. Labels

Three mutually exclusive labels. Random-guess baseline is ~33%, which keeps any fine-tuning gain legible.

### `analysis`
Makes a structured argument backed by **specific, verifiable** evidence — statistics, historical comparison, or tactical/film observation. If you stripped the opinion framing, a reasoned claim would still be standing.

- *Example:* "Jokic's assist numbers are inflated by Denver's pace, but even adjusting for that his assist rate leads all centers since prime CP3 — the gravity is real."
- *Example:* "They keep switching the 1-5 pick and Gobert can't recover to the perimeter, that's why the corner three is wide open every possession."

### `hot_take`
A bold, confident **opinion** asserted without real support. It may *cite* a stat, but the stat is decorative or cherry-picked rather than load-bearing — just enough to sound credible. It asserts rather than argues.

- *Example:* "Embiid is a top-3 player of all time, period. People will look back and realize."
- *Example:* "LeBron is overrated, his Finals record is below .500 and that says everything."

### `reaction`
An in-the-moment **emotional response** to a specific event, play, or game. Little to no argument — the comment is expressing a feeling.

- *Example:* "I CANNOT believe he hit that fadeaway are you kidding me"
- *Example:* "welp. season over. see everyone next year 💀"

## 3. Hard edge cases

The whole project hinges on the boundary rules, so they are written down here and applied consistently during annotation.

### Edge case A — the one-stat post (`hot_take` vs `analysis`)
The most common ambiguity: a confident claim with a single statistic attached.

> "LeBron is overrated — his playoff win rate against top seeds is below .500."

**Decision rule:** If the evidence would genuinely support the claim even with the opinion framing stripped out → `analysis`. If the stat is vague, cherry-picked, or decorative (selected for effect, not as part of a reasoning chain) → `hot_take`. The example above uses one cherry-picked stat with accusatory framing → **`hot_take`**.

### Edge case B — emotional claim (`reaction` vs `hot_take`)
An emotional outburst that also smuggles in a claim.

> "LeBron is WASHED holy crap I can't watch this"

**Decision rule:** Strip the emotion — is there a claim left standing on its own? If yes and it's the dominant content → `hot_take`. If the comment is primarily an exclamation tied to the moment → `reaction`. "Washed" survives as a (weak) claim → leans **`hot_take`**; "I can't watch this anymore" alone → `reaction`.

### Edge case C — sarcasm / rhetorical questions
> "Oh sure, trading three first-rounders for a 34-year-old, that'll age well."

**Decision rule:** Sarcasm that encodes an actual argument (here: the trade is bad because of age + draft cost) is judged on the *argument underneath it*. If the underlying argument is evidenced → `analysis`; if it's just a confident jab → `hot_take`; if it's purely venting with no argument → `reaction`. This one carries an implicit cost/age argument but no real evidence → **`hot_take`**.

> Additional difficult examples encountered during annotation are logged in Section "Hard annotation decisions (log)" below.

## 4. Data collection plan

- **Source:** Public comment JSON from r/nba threads (and adjacent NBA discussion threads), pulled via Reddit's unauthenticated `.json` endpoints. Public content only.
- **Variety strategy:** Class mix follows thread type, so I scrape across kinds — game/live threads (skew `reaction`), post-game and "[Highlight]" threads (mix), and daily-discussion / post-game-analysis threads (most of the `analysis` and `hot_take`).
- **Target:** ~65–70 examples per label, ≥200 total. Hard floor of 20% per class (well under the 70% imbalance ceiling).
- **Cleaning:** Drop deleted/removed/bot comments, drop ultra-short noise that isn't a genuine reaction, dedupe.
- **If a label is underrepresented after 200** (almost certainly `analysis`): scrape more from analysis-heavy threads and bias toward longer comments (length correlates with sustained argument), then re-balance before training.
- **Output:** one labeled CSV `data/takemeter_dataset.csv` with columns `text`, `label`, `notes`. Not pre-split — the notebook does the 70/15/15 split.

## 5. Evaluation metrics

- **Overall accuracy** (both models) — headline number, but insufficient alone because it hides per-class collapse on a possibly-imbalanced set.
- **Per-class precision / recall / F1, plus macro-F1** — macro-F1 is the *real* headline: it weights each class equally, so a model that punts on the rare `analysis` class is penalized instead of rewarded by the majority class.
- **Confusion matrix** — to identify *which* boundary blurs. Expectation: `analysis`↔`hot_take` is the limiting confusion.

These are the right metrics because the task is a balanced-importance multiclass problem where the rare, hard class (`analysis`) is exactly the one I care about getting right — accuracy would let a lazy model hide, macro-F1 + the confusion matrix will not.

## 6. Definition of success

- **Good enough for deployment in a real community tool:** fine-tuned **macro-F1 ≥ 0.70**, beating the Groq zero-shot baseline by a meaningful margin (**≥10 points accuracy**), with **no class at F1 ≈ 0**.
- **The real bar I'm watching:** `analysis` recall **≥ 0.65**, since the `analysis`↔`hot_take` boundary is the hardest and the most consequential (mislabeling a hot take as analysis is the failure that would most annoy real users).
- Anything below the deployment bar is still a *successful project* if the evaluation honestly diagnoses *why* — the brief values a diagnosable failure over an opaque success.

## AI Tool Plan

There is no application code to generate here, so AI tools help at three specific points:

1. **Label stress-testing (before annotating).** Give an LLM the label definitions + edge-case rules and ask it to generate 5–10 posts deliberately sitting on the `analysis`/`hot_take` and `reaction`/`hot_take` boundaries. If I can't classify its output cleanly with my own rules, the definitions get tightened *before* I annotate 200 examples.
2. **Annotation assistance (pre-labeling).** `scripts/prelabel_groq.py` sends each scraped comment to Groq (`llama-3.3-70b-versatile`) with the label definitions and gets a *suggested* label. I then read and correct **every** suggested label by hand — pre-labeling reorders the work (review beats label-from-scratch) but does not replace judgment. The pre-labeled column is kept separate from my final `label` column so I can measure how often I overrode the model, and this is disclosed in the README AI-usage section. **Critically, this pre-labeling model is NOT the baseline** — the baseline (notebook Sec 5) is a separate clean zero-shot run on the locked test set, so pre-labeling cannot contaminate it.
3. **Failure analysis (after evaluation).** Paste the fine-tuned model's misclassifications into an LLM and ask it to surface patterns (label pair confused, post length, sarcasm, low-information posts). Then re-read the flagged examples myself to confirm or discard each pattern before it goes in the report.

## Hard annotation decisions (log)

Populated during Milestone 3 — at least three genuine judgment calls with what I decided and why.

- _TBD during annotation._
