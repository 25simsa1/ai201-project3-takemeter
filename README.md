# TakeMeter

A fine-tuned text classifier that scores discourse quality on r/nba by sorting comments into three kinds of take. This README is the final report. The design notes and working decisions live in [planning.md](planning.md).

## Community

I chose r/nba and a handful of adjacent NBA discussion threads. NBA Reddit is a good fit because the discourse is high volume, text heavy, and varies enormously in quality inside the same thread. A post-game thread holds a 200-word breakdown of a defensive scheme right next to a one-line "LETS GOOOO". The hot take versus real analysis split is something the community argues about constantly ("that's a hot take", "actually cooking with this analysis"), so the labels are grounded in how people there actually talk. It is also easy to collect, since Reddit serves public thread JSON.

## Label taxonomy

Three mutually exclusive labels. Random-guess accuracy is about 33%.

**analysis** is a structured argument backed by specific, verifiable evidence such as stats, historical comparison, or tactical/film observation. Strip the opinion framing and a reasoned claim still stands.
- "In 2009 he was listed at 6'9 250. That's the average weight of an NFL linebacker. Was still the fastest, highest flying player on the floor and did it for 39 minutes a night."
- "They keep switching the 1-5 pick and Gobert can't recover to the perimeter, that's why the corner three is wide open every possession."

**hot_take** is a bold, confident opinion asserted without real support. It may cite a stat, but the stat is decorative or cherry-picked, not load-bearing. It asserts instead of arguing.
- "Embiid is a top-3 player of all time, period. People will look back and realize."
- "Best pure athlete across all of sports history tbh."

**reaction** is an in-the-moment emotional response to a specific event, play, or game. Little to no argument, it expresses a feeling.
- "THANK THE BASKETBALL GODS"
- "Holy choke of the decade"

## Data collection and annotation

**Source.** I saved the public comment JSON for six r/nba threads chosen to span the three labels: the LeBron-athleticism thread, the "OKC eliminated" thread, the Shai questionable-foul highlight, the Popovich/Kerr story, the OG Anunoby tip-in highlight, and an "if you could be any player ever" thread. Game and highlight threads feed reaction, while the LeBron and any-player threads feed analysis and hot_take. Reddit blocks unauthenticated programmatic requests with a 403, so I fetched the `.json` in a logged-in browser and ran the scraper in an offline mode that reads the saved files.

**Pipeline.** A small scraper (`scripts/scrape_reddit.py`) flattens the comment trees and filters out deleted, bot, and ultra-short noise, yielding 2,178 raw comments. I sampled this down with a length-stratified draw (short comments feed reaction, long comments feed analysis), then pre-labeled the sample with Groq `llama-3.3-70b-versatile` (`scripts/prelabel_groq.py`) and reviewed every label by hand. The Groq suggestion is kept frozen in a separate column so overrides are measurable.

**Final distribution** (`data/takemeter_dataset.csv`, 219 rows):

| label | count | share |
|---|---|---|
| reaction | 81 | 37% |
| analysis | 73 | 33% |
| hot_take | 65 | 30% |

No class exceeds 70% and every class clears 20%. The hand review changed 5 of Groq's labels and filled 12 it left blank, for an override rate of about 2%. That low rate is honest for this dataset. It is a stat-heavy LeBron/Wilt thread, so most of Groq's calls were defensible. It also turns out to be the single most important fact about the whole project (see Reflection).

**Three difficult cases** (full list in planning.md):

1. "The call he got, where he literally jumped left into the guy at his side... call the game according to the rules." Sat between reaction and analysis. It describes a specific play to argue a foul was wrong, so it is not pure emotion, but the evidence is one play described emotively rather than a structured argument. Decided **hot_take**.
2. "Everyone needs to get paid and owners dont want to pay $500M in luxury tax... See Tatum's leg, MPJ/Gordon/Murray injuries... Dynasties are very difficult." Sat between hot_take and analysis. The claim is backed by a stack of specific roster, injury, and CBA examples doing real work. Decided **analysis**.
3. "Lol what is this post. Russell is one of the most athletic players ever... Wilt benched more, had a higher vert." Sat between hot_take and analysis. Despite the dismissive opener, it cites specific comparative evidence that carries the claim. Decided **analysis**.

## Fine-tuning approach

**Base model.** `distilbert-base-uncased` with a fresh 3-class classification head.

**Setup.** The dataset is split 70/15/15 (train 153, validation 33, test 33), stratified by label with seed 42. Text is tokenized at max length 256. Training runs through the `Trainer` API and keeps the best validation-accuracy checkpoint. I ran it locally on an Apple-Silicon GPU (MPS) via `scripts/train_eval.py` rather than in the Colab notebook, because the Groq daily token cap forced me to compute the baseline from saved predictions, and a single local script kept the whole evaluation reproducible. The method is identical to the notebook (same model, split, and arguments).

**Key hyperparameter decision.** The first training run was a failure that taught the most. With the notebook defaults (3 epochs, batch 16, `warmup_steps=50`) the model sat at exactly 0.333 accuracy across all three epochs and the loss barely moved. The cause was the warmup schedule. Three epochs over 153 examples at batch 16 is only about 30 optimizer steps, but `warmup_steps=50` is larger than that, so the learning rate ramped from zero and never finished warming up. The weights barely updated and the model stayed at chance, predicting one class for everything. I fixed it by raising epochs to 10 and dropping the batch size to 8 (about 190 steps), setting `warmup_steps=20` (roughly 10% of training), and nudging the learning rate to 3e-5. Validation accuracy then climbed to 0.79 and training loss fell to 0.46.

## Baseline

The baseline is zero-shot Groq `llama-3.3-70b-versatile` at temperature 0, prompted with the same label definitions and one example per label and told to reply with only the label name. The exact prompt is in `scripts/prelabel_groq.py` (`SYSTEM_PROMPT`). Because the live Groq daily token limit was exhausted during pre-labeling and Developer-tier upgrades were unavailable, I collected the baseline predictions from the pre-labeling pass, which is the same model with the same prompt run on the same comments. The 33 test rows are a subset, so reading their stored Groq predictions gives a valid zero-shot baseline on the locked test set with no leakage from training (the fine-tuned model never saw the test rows). Two of the 33 test responses were not one of the three labels and count as unparseable, leaving 31 scored.

## Evaluation report

### Overall

| model | accuracy | macro-F1 |
|---|---|---|
| Groq zero-shot baseline | 0.935 (31/33 parseable) | 0.933 |
| Fine-tuned DistilBERT | 0.667 | 0.663 |

Fine-tuning **regressed** by 27 points against the baseline. That looks bad, but it is the expected and informative result here, explained in the Reflection.

### Per-class

Fine-tuned DistilBERT:

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| analysis | 0.86 | 0.55 | 0.67 | 11 |
| hot_take | 0.46 | 0.60 | 0.52 | 10 |
| reaction | 0.77 | 0.83 | 0.80 | 12 |

Groq baseline:

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| analysis | 0.83 | 1.00 | 0.91 | 10 |
| hot_take | 1.00 | 0.80 | 0.89 | 10 |
| reaction | 1.00 | 1.00 | 1.00 | 11 |

### Confusion matrix, fine-tuned model (rows = true, columns = predicted)

|  | pred analysis | pred hot_take | pred reaction |
|---|---|---|---|
| **true analysis** | 6 | 5 | 0 |
| **true hot_take** | 1 | 6 | 3 |
| **true reaction** | 0 | 2 | 10 |

The off-diagonal mass is concentrated on the analysis/hot_take boundary: 5 of 11 true analysis comments were called hot_take, and the two classes trade errors in both directions. reaction is the cleanest class (10 of 12 right, F1 0.80). This is exactly the hard boundary predicted in planning.md.

### Three wrong predictions, analyzed

1. **"NBA changed the rules to nerf Shaq. He is the only player in the past 30 years that the NBA actively worked against..."** True analysis, predicted hot_take at 0.74. The comment makes a verifiable historical claim (the hack-a-Shaq rule change), which is what makes it analysis, but its surface form is a single confident assertion. The model keys on the assertive register rather than the buried evidence. This is the core analysis-to-hot_take failure.

2. **"Bruh Bron, especially at Miami, was a 1-5 player... Wilt could've never played the point like that."** True hot_take, predicted analysis at 0.91, the model's most confident error. It names specific abilities (guarding 1-5, Dwight-level strength), so the model reads it as evidenced. But the specifics are decorative assertions, not an argument, so a human reads it as a hot take. The model has learned that "mentions specific basketball details" means analysis, which is not the same boundary I intended.

3. **"Best pure athlete across all of sports history tbh."** True hot_take, predicted reaction at 0.66. This is a bold claim with zero support, a textbook hot take, but it is only eight words. The model has tied short length to reaction, so brevity overrides the claim. Short hot takes are a systematic blind spot.

### Sample classifications, fine-tuned model

| comment | predicted | confidence | true |
|---|---|---|---|
| "THANK THE BASKETBALL GODS" | reaction | 0.93 | reaction ✓ |
| "I think he was more athletic in Miami probably too. He moved similarly to this but was just so damn big." | analysis | 0.92 | analysis ✓ |
| "Holy choke of the decade" | reaction | 0.91 | reaction ✓ |
| "Best pure athlete across all of sports history tbh." | reaction | 0.66 | hot_take ✗ |
| "Bruh Bron, especially at Miami, was a 1-5 player..." | analysis | 0.91 | hot_take ✗ |

The first one is a clean, reasonable call: "THANK THE BASKETBALL GODS" is a pure emotional exclamation with no argument, the model is correctly and confidently reaction. The pattern across the correct high-confidence rows is that the model is strongest when surface form and label agree (short exclamation = reaction, measured stat-cite = analysis) and weakest when they pull apart.

## Reflection, what the model learned versus what I intended

I intended the model to learn the difference between *arguing* and *asserting* a basketball opinion. What it actually learned is a set of surface proxies for that difference: long and stat-flavored reads as analysis, short and punchy reads as reaction, and confident-but-unsupported lands wherever the surface points. When those proxies line up with the real distinction the model is right, and when they come apart (a short hot take, an assertion dressed in specifics) it fails. The confusion matrix shows this cleanly, with the analysis/hot_take boundary, the one genuinely about reasoning rather than surface, carrying most of the errors.

The larger lesson is methodological and it is the most important finding of the project. My labels were bootstrapped from Groq's own zero-shot predictions and I only overrode about 2% of them. That means the test set's ground truth is, for the most part, Groq's answers. So when I grade Groq against that test set it scores 93.5%, not because it is a near-perfect judge of discourse but because it is being graded against its own output. The baseline is circular. A small model trained on 153 of those same labels then cannot beat the model that wrote the labels. The honest reading is not "fine-tuning failed" but "this comparison was confounded the moment I used the baseline model to pre-label the data." A clean experiment would label fully by hand, with no LLM pre-labeling, so the baseline is independent of the ground truth. I would do that next time, or at least raise the override rate by labeling a larger fraction from scratch.

Against my own success criteria from planning.md (macro-F1 at least 0.70 and beating the baseline), the project did not pass: macro-F1 is 0.66 and the baseline wins. But the failure is fully diagnosable, which planning.md argued was the real goal, and it points at a specific fix.

## Spec reflection

**One way the spec helped.** planning.md committed me to a single decision rule for the analysis/hot_take boundary ("strip the opinion framing, does a reasoned claim survive on specific evidence"). Having that written down kept my 219 hand labels consistent, and it let me predict before training that analysis/hot_take would be the model's hardest boundary. The confusion matrix confirmed it, which would have been hard to claim credibly without the rule on paper first.

**One way the implementation diverged.** The plan assumed I would fine-tune and run the baseline inside the Colab notebook. In practice Reddit blocked programmatic scraping, the Groq daily token cap was exhausted by pre-labeling, and Developer-tier upgrades were unavailable, so I added an offline file-reading mode to the scraper, computed the baseline from the stored pre-labeling predictions, and ran the fine-tune locally on MPS through one reproducible script. The end artifacts (the dataset, the metrics, the confusion matrix) are exactly what the plan specified.

## AI usage

1. **Label and prompt design.** I used Claude to pressure-test my three label definitions and to draft the Groq classification prompt from them. I kept the structure but tightened the analysis/hot_take rule into the "strip the framing" test that ended up in planning.md.

2. **Annotation assistance (disclosed).** I used Groq `llama-3.3-70b-versatile` to pre-label every collected comment, then reviewed all 219 by hand against my definitions, correcting 5 labels and filling 12 the model left blank. This pre-labeling is the reason the baseline comparison is confounded, which I treat as the central finding rather than hiding it. The frozen `suggested_label` column in `data/reviewed.csv` is the audit trail.

3. **Failure analysis.** I used Claude to help spot the error pattern across the misclassified test cases. It flagged the analysis-to-hot_take direction and the short-hot-take blind spot, which I then verified by re-reading the specific comments and checking the confusion matrix before writing them up here.

## Repository

- `planning.md` - design, label rules, edge cases, metrics, AI tool plan
- `data/takemeter_dataset.csv` - 219 labeled examples (text, label, notes)
- `data/reviewed.csv` - annotation trail with Groq suggestion vs final label
- `scripts/scrape_reddit.py`, `scripts/prelabel_groq.py`, `scripts/dataset_stats.py` - data pipeline (tested)
- `scripts/train_eval.py` - fine-tune and evaluation
- `outputs/evaluation_results.json`, `outputs/confusion_matrix.png`, `outputs/predictions.csv` - results
