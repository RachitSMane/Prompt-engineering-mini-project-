# ReviewIQ — demo talk track

> **Superseded figures (audit 2026-11-14).** This file was written for the 2026-11-12 state. Current results: judge v2 valid for 15 / 15 summaries (v1 5 / 15 preserved); negative control on 15 products with 30 planted claims, 18 / 30 detected and 12 missed (the judge is not reliable); evidence audit 13 / 15; D2 0 / 300 verified human labels; tests 269 passed, 1 skipped (final pass; judge v3 and the reference-approval workflow added). Use `MENTOR_DEMO_GUIDE.md` instead of this file.

Target: about 6 minutes. Numbers are the ones the app shows (verified 2026-11-12). Selections are in `DEMO_GUIDE.md` §4.

---

### 0. Opening (20 s), on the Overview page

> "ReviewIQ turns raw product reviews into structured, validated data and then into grounded product summaries. Everything
> you will see ran on a **local** model, Qwen 2.5 3B through Ollama on this laptop, with no cloud APIs. I'm running the app in
> **read-only demo mode**, the amber banner at the top: it shows the saved experiment results and blocks every model call and
> every write, so nothing I click can change the results in my report."

### 1. Overview (40 s)

> "The dataset is 1,500 Amazon appliance reviews across 15 products. 285 of them have been processed by the model into
> validated records. Of those, 38.6 % are positive and 53.7 % negative; it's a replacement-parts category, so complaints
> dominate. The aspect chart shows what customers talk about: performance and compatibility lead. These charts are computed
> with pandas from the saved records; the model is only used for extraction."

Point at the deliverables row: "D1 extraction, D2 prompt comparison on 299 labelled reviews, D3 summaries and judge."

### 2. Review Analysis (90 s), Deliverable 1

> "This is the structured extraction. I'm loading the **saved** results; the green callout says they came from qwen2.5:3b
> and that this view makes zero LLM calls. 285 reviews passed schema validation and 14 failed even after repair. Those
> failures are reported, not hidden or silently fixed."

Filter product B09WWM1WLS, sentiment positive, click the first row.

> "Each record follows a Pydantic schema: overall sentiment, aspects with a category from a fixed taxonomy, a sentiment per
> aspect, and a **verbatim evidence quote** from the review. Here, the reviewer praises performance but complains about rough
> edges, and the model captured both: quality durability negative and ease of use positive, each with its quote. Evidence is
> checked against the review text, so the model can't invent a quote."

Open "Repair log": "If the first output is invalid JSON or breaks the schema, we try deterministic local fixes first, then
up to two repair prompts. 77 records were fixed locally and 3 needed an LLM repair."

Mention downloads: "JSON and CSV exports are the D1 deliverable."

### 3. Prompt Experiments (90 s), Deliverable 2

> "I compared three prompting strategies on the same 299 reviews: zero-shot, few-shot and chain-of-thought. Few-shot wins
> clearly: macro-F1 0.603 versus 0.438 for zero-shot and 0.423 for chain-of-thought, and accuracy 81 % versus 61 % and 58 %.
> I use macro-F1 as the primary metric because the classes are imbalanced; accuracy alone would reward predicting the
> majority class."

Open "Confusion matrices":
> "The confusion matrices show why. Zero-shot and chain-of-thought push many positive reviews into **neutral**: CoT labels
> 51 of the positive reviews neutral, so its neutral precision is about 0.05. Few-shot fixes that (102 positives correct).
> All three struggle with **mixed**; even few-shot only recalls 31 % of mixed reviews, mostly calling them negative.
> Neutral has only 5 reviews, so its per-class scores are very noisy."

**Say the limitation:**
> "An honest caveat: the reference labels are mostly **AI-assisted**: 298 of 299, with one human label. The app flags this
> in the provenance callout. So these scores measure agreement with those reference labels, not with independent human
> annotation. Validating with human labels is the most important next step."

Open "Efficiency": "Few-shot also costs more: about 2,900 input tokens and 54 s per review on this laptop, versus 41 s for
zero-shot."

### 4. Product Intelligence (75 s), Deliverable 3 part 1

Select "Napco…" (B09WWM1WLS).

> "This is the prompt chain. Stage 1 is extraction, stage 2 is pure pandas aggregation (no LLM; it's labelled
> 'Calculated statistics'), and stage 3 is the LLM summary, generated from those statistics and the evidence quotes."

Expand a summary point:
> "Every point in the summary has to cite review IDs, and clicking it shows the actual reviews it's based on, so you can
> check each claim."

Point at the greyed-out Generate buttons: "In demo mode generation is disabled; these summaries are the saved ones."

### 5. Summary Evaluation (75 s), Deliverable 3 part 2

> "To evaluate the summaries I used LLM-as-a-judge, with a **different** model family, Llama 3.2, so the summariser isn't
> grading itself. It scores faithfulness, coverage, consistency and hallucination from 1 to 5. Alongside it there are
> deterministic checks that need no LLM: for example, whether every percentage in the summary exists in the statistics, and
> whether the cited IDs belong to this product."

**Say the limitations, plainly:**
> "The judge turned out to be the weakest component. For 10 of the 15 summaries Llama 3.2 never produced valid structured
> output, even after repairs. I kept those as recorded failures instead of re-running until I got a score. Even the 5
> successful judgements contradict themselves: here it gives hallucination 2 out of 5, which means many unsupported claims,
> but its own explanation says there are none."

> "I also ran a negative control: I deliberately corrupted summaries, inverting the opinion, inventing a feature and
> removing complaints. The corrupted versions got a slightly lower average score in all 3 pairs that could be judged, but
> the **faithfulness** score never dropped (0 of 3), and it didn't catch the invented feature. So I don't treat these judge
> scores as reliable evidence of summary quality; the deterministic checks and the cited evidence are more trustworthy."

Switch to "Whirlpool 3949238": "This is what a recorded judge failure looks like: no score and the actual error, shown openly."

### 6. Human Annotation (30 s)

> "This is the annotation tool used to build the 300-review reference set: 299 complete and 1 flagged as unreadable and
> excluded. As I said, most labels were AI-assisted, and the tool records that provenance for every label. In demo mode the
> labels can be browsed and exported but not changed."

### Close (15 s)

> "To sum up: structured extraction with validation and repair works on a 3B local model; few-shot prompting clearly beats
> zero-shot and chain-of-thought; the summaries are grounded in cited evidence. The two open problems are human-validated
> labels and a more reliable judge. I'm happy to take questions."
