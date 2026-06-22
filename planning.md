# TakeMeter Planning

A fine-tuned classifier that scores discourse quality on an NBA community by sorting comments into three kinds of take. This document is the design thinking behind the project. It covers the labels, the edge-case rules, the data plan, the metrics, and the AI tool plan. I wrote it before collecting any data.

## 1. Community

I chose r/nba and a few adjacent NBA discussion threads.

NBA Reddit fits well because the discourse runs high volume, leans heavily on text, and varies enormously in quality inside the same thread. A post-game thread will hold, side by side, a 200-word breakdown of a defensive scheme, a one-line "X is the best player alive and it's not close," and a pure "LETS GOOOO." The hot take versus real analysis split is not something I'm imposing from outside. It's a thing the community itself argues about constantly ("that's a hot take," "actually cooking with this analysis"), which is what makes the labels grounded in how participants really talk. That's the requirement that matters most.

It's also easy to collect. Reddit serves public thread JSON with no authentication, so 200 or more real comments are a few requests away.

## 2. Labels

Three mutually exclusive labels. Random-guess accuracy sits near 33%, which keeps any fine-tuning gain easy to read.

### `analysis`
Makes a structured argument backed by specific, verifiable evidence such as statistics, historical comparison, or tactical and film observation. Strip the opinion framing and a reasoned claim is still standing.

- Example. "Jokic's assist numbers are inflated by Denver's pace, but even adjusting for that his assist rate leads all centers since prime CP3, the gravity is real."
- Example. "They keep switching the 1-5 pick and Gobert can't recover to the perimeter, that's why the corner three is wide open every possession."

### `hot_take`
A bold, confident opinion asserted without real support. It may cite a stat, but the stat is decorative or cherry-picked rather than load-bearing, just enough to sound credible. It asserts instead of arguing.

- Example. "Embiid is a top-3 player of all time, period. People will look back and realize."
- Example. "LeBron is overrated, his Finals record is below .500 and that says everything."

### `reaction`
An in-the-moment emotional response to a specific event, play, or game. Little to no argument. The comment is expressing a feeling.

- Example. "I CANNOT believe he hit that fadeaway are you kidding me"
- Example. "welp. season over. see everyone next year 💀"

## 3. Hard edge cases

The whole project hinges on the boundary rules, so I write them down here and apply them the same way every time during annotation.

### Edge case A, the one-stat post (`hot_take` vs `analysis`)
The most common ambiguity is a confident claim with a single statistic attached.

> "LeBron is overrated, his playoff win rate against top seeds is below .500."

Rule. If the evidence would genuinely support the claim even with the opinion framing stripped out, it's `analysis`. If the stat is vague, cherry-picked, or decorative, selected for effect rather than as part of a reasoning chain, it's `hot_take`. The example above uses one cherry-picked stat with accusatory framing, so it lands as `hot_take`.

### Edge case B, the emotional claim (`reaction` vs `hot_take`)
An emotional outburst that also smuggles in a claim.

> "LeBron is WASHED holy crap I can't watch this"

Rule. Strip the emotion and ask whether a claim is left standing on its own. If yes and it's the dominant content, it's `hot_take`. If the comment is mostly an exclamation tied to the moment, it's `reaction`. "Washed" survives as a weak claim, so it leans `hot_take`. "I can't watch this anymore" alone is `reaction`.

### Edge case C, sarcasm and rhetorical questions
> "Oh sure, trading three first-rounders for a 34-year-old, that'll age well."

Rule. Sarcasm that encodes an actual argument (here, the trade is bad because of age plus draft cost) is judged on the argument underneath it. If that argument is evidenced, it's `analysis`. If it's just a confident jab, it's `hot_take`. If it's purely venting with no argument, it's `reaction`. This one carries an implicit cost-and-age argument but no real evidence, so it's `hot_take`.

I log any other difficult examples I hit during annotation in the "Hard annotation decisions" section below.

## 4. Data collection plan

I pull public comment JSON from r/nba threads, and a few adjacent NBA discussion threads, through Reddit's unauthenticated `.json` endpoints. Public content only.

Class mix follows thread type, so I scrape across kinds to get variety. Game and live threads skew toward `reaction`. Post-game and "[Highlight]" threads give a mix. Daily-discussion and post-game-analysis threads carry most of the `analysis` and `hot_take`.

I'm targeting about 65 to 70 examples per label, 200 or more total, with a hard floor of 20% per class (well under the 70% imbalance ceiling). Cleaning drops deleted, removed, and bot comments, drops ultra-short noise that isn't a genuine reaction, and dedupes.

If a label is underrepresented after the first 200, and that's almost certainly going to be `analysis`, I scrape more from analysis-heavy threads and bias toward longer comments, since length correlates with sustained argument, then re-balance before training.

The output is one labeled CSV at `data/takemeter_dataset.csv` with columns `text`, `label`, and `notes`. It is not pre-split. The notebook does the 70/15/15 split.

## 5. Evaluation metrics

Overall accuracy for both models is the headline number, but it isn't enough on its own because it hides per-class collapse on a set that may be imbalanced.

Per-class precision, recall, and F1, plus macro-F1, is the real headline. Macro-F1 weights each class equally, so a model that punts on the rare `analysis` class gets penalized instead of rewarded by the majority class.

The confusion matrix tells me which boundary blurs. My expectation is that `analysis` and `hot_take` will be the limiting confusion.

These are the right metrics because the task is a balanced-importance multiclass problem where the rare, hard class (`analysis`) is exactly the one I care about getting right. Accuracy would let a lazy model hide. Macro-F1 plus the confusion matrix will not.

## 6. Definition of success

Good enough to deploy in a real community tool means fine-tuned macro-F1 of 0.70 or higher, beating the Groq zero-shot baseline by a meaningful margin (10 or more points of accuracy), with no class sitting at F1 near zero.

The real bar I'm watching is `analysis` recall of 0.65 or higher, since the `analysis` versus `hot_take` boundary is the hardest and the most consequential. Mislabeling a hot take as analysis is the failure that would annoy real users most.

Anything below the deployment bar is still a successful project if the evaluation honestly diagnoses why. The brief values a diagnosable failure over an opaque success.

## AI Tool Plan

There's no application code to generate here, so AI tools help at three specific points.

Label stress-testing, before annotating. I give an LLM the label definitions and edge-case rules and ask it to generate five to ten posts that deliberately sit on the `analysis`/`hot_take` and `reaction`/`hot_take` boundaries. If I can't classify its output cleanly with my own rules, I tighten the definitions before annotating 200 examples.

Annotation assistance, pre-labeling. `scripts/prelabel_groq.py` sends each scraped comment to Groq (`llama-3.3-70b-versatile`) with the label definitions and gets back a suggested label. I then read and correct every suggested label by hand. Pre-labeling reorders the work, since reviewing beats labeling from scratch, but it does not replace judgment. I keep the pre-labeled column separate from my final `label` column so I can measure how often I overrode the model, and I disclose this in the README AI-usage section. The pre-labeling model is not the baseline. The baseline (notebook Section 5) is a separate clean zero-shot run on the locked test set, so pre-labeling can't contaminate it.

Failure analysis, after evaluation. I paste the fine-tuned model's misclassifications into an LLM and ask it to surface patterns, such as which label pair gets confused, post length, sarcasm, or low-information posts. Then I re-read the flagged examples myself to confirm or discard each pattern before it goes in the report.

## Hard annotation decisions

I fill this in during Milestone 3 with at least three genuine judgment calls, each with what I decided and why. Empty for now until annotation starts.
