# D1 — Structured extraction export · run `20261111-030912_qwen2.5-3b`

Derived from the run's saved predictions (no new LLM calls). Every exported record was reloaded and re-validated with the Pydantic schema, including evidence grounding against its own review text.

| strategy | coverage | valid | failed (output) | failed (provider) | first-attempt valid | needed repair | repaired OK | re-validated | mean s | p95 s | input tokens | output tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero_shot | 299/299 | 282 | 17 | 0 | 273 | 26 | 9 | 282/282 | 41.19 | 143.29 | 615,159 | 64,126 |
| few_shot | 299/299 | 285 | 14 | 0 | 282 | 17 | 3 | 285/285 | 53.77 | 109.1 | 862,456 | 68,146 |
| cot | 299/299 | 279 | 20 | 0 | 274 | 25 | 5 | 279/279 | 47.53 | 135.68 | 686,970 | 66,718 |

## zero_shot

- Model: `ollama/qwen2.5:3b`
- Repair attempts distribution (attempts: reviews): `{'0': 273, '1': 7, '2': 19}`
- Records with deterministic (local) fixes: 12
- Most frequent validation errors before repair: `{'aspects.0.evidence': 24, 'complaint_categories.0': 22, 'aspects.1.evidence': 8, 'overall_sentiment': 7, 'aspect_name': 7, 'aspect_category': 7, 'sentiment': 7, 'evidence': 7, 'complaint_categories.1': 5, 'aspects.2.evidence': 5}`
- Failures: 17
  - `R3936467669fb`: Output still invalid after 2 repair attempt(s): aspects.1.evidence: String should have at most 300 characters
  - `Rb6fd13816de5`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: String should have at most 300 characters
  - `Rec16a339ebbc`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `R4bd55a149bd3`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: String should have at most 300 characters
  - `R02f621435777`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence "It's a miracle this thing didn't break my Keurig! Didn't fit properly and almost" 
  - `Rf28a59584713`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'very quiet, very movable, easy to use and install' is not a verbatim quote from th
  - `Re02ef86668af`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `Rb3975bb1c59a`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `R1d1832973ec9`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'The lights fit our structure, like a glove.' is not a verbatim quote from the revi
  - `Rd1bbcb595b1f`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: String should have at most 300 characters; aspects.4.evidence: String should have at most 30
  - `Rbd2a0049f12f`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `R755425de6ae7`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'They are sturdy & easy to clean. They have provided us with space saving.' is not 
  - `R1c492cdbe101`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence "They're great for storing lots of eggs. They're perfect for the storage of those" 
  - `R5f83464aea35`: Output still invalid after 2 repair attempt(s): overall_sentiment: Field required; aspect_name: Extra inputs are not permitted; aspect_category: Extra inputs ar
  - `R3b4b64dd2026`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'The second or third time we’ve replaced the heating element for our Samsung drye' 
  - `R6e47845099ff`: Output still invalid after 2 repair attempt(s): complaint_categories.1: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `R681a812a58d9`: Output still invalid after 2 repair attempt(s): overall_sentiment: Field required; aspect_name: Extra inputs are not permitted; aspect_category: Extra inputs ar

## few_shot

- Model: `ollama/qwen2.5:3b`
- Repair attempts distribution (attempts: reviews): `{'0': 282, '1': 3, '2': 14}`
- Records with deterministic (local) fixes: 85
- Most frequent validation errors before repair: `{'invalid JSON': 10, 'aspects.0.aspect_category': 9, 'complaint_categories.0': 7, 'aspects.0.evidence': 7, 'aspects.1.evidence': 3, 'aspects.3.evidence': 3, 'aspects.2.evidence': 3, 'aspects.1.aspect_category': 3, 'complaint_categories.2': 1, 'aspects.5.evidence': 1}`
- Failures: 14
  - `R3936467669fb`: Output still invalid after 2 repair attempt(s): aspects.1.evidence: String should have at most 300 characters
  - `R12153e2344dc`: Output still invalid after 2 repair attempt(s): aspects.0.aspect_category: Input should be 'performance', 'quality_durability', 'design_build', 'ease_of_use', '
  - `R22c2ae3f5f5e`: Output still invalid after 2 repair attempt(s): invalid JSON: Expecting value at line 162 column 18
  - `Rf28a59584713`: Output still invalid after 2 repair attempt(s): aspects.3.evidence: evidence "I don't think it pulls fabric like commercial or home units. It has shot up my e" 
  - `R7221d5e772ba`: Output still invalid after 2 repair attempt(s): invalid JSON: Unterminated string starting at at line 167 column 5
  - `Rfe95c8abb789`: Output still invalid after 2 repair attempt(s): invalid JSON: Unterminated string starting at at line 153 column 5
  - `Rd1d55023e6fd`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `R1d1832973ec9`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'Great purchase! Recommended purchase' is not a verbatim quote from the review text
  - `R71e993199614`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `Reb1ecc83b4e7`: Output still invalid after 2 repair attempt(s): aspects.2.evidence: evidence 'We love our 77-L egg container. We love the high sides and the spacing between t' 
  - `R4a5c174051f0`: Output still invalid after 2 repair attempt(s): aspects.0.aspect_category: Input should be 'performance', 'quality_durability', 'design_build', 'ease_of_use', '
  - `R82e08ead8ec9`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'It just makes a weird noise. It just does not spin or anything' is not a verbatim 
  - `R3b4b64dd2026`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'The second or third time we’ve replaced the heating element for our Samsung drye' 
  - `R59874d4d9366`: Output still invalid after 2 repair attempt(s): aspects.0.aspect_category: Input should be 'performance', 'quality_durability', 'design_build', 'ease_of_use', '

## cot

- Model: `ollama/qwen2.5:3b`
- Repair attempts distribution (attempts: reviews): `{'0': 274, '1': 3, '2': 22}`
- Records with deterministic (local) fixes: 9
- Most frequent validation errors before repair: `{'aspects.0.evidence': 19, 'complaint_categories.0': 15, 'aspects.2.evidence': 12, 'rationale': 9, 'overall_sentiment': 6, 'aspect_name': 6, 'aspect_category': 6, 'sentiment': 6, 'evidence': 6, 'invalid JSON': 6}`
- Failures: 20
  - `R3936467669fb`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'The timer stopped advancing and the dryer runs at full heat until you open the d' 
  - `Rf0680f2c3f3e`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `Rb6fd13816de5`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'The area where the water enters the ice maker is different. You can, however, sw' 
  - `R1cec8053b10c`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `R4bd55a149bd3`: Output still invalid after 2 repair attempt(s): aspects.2.evidence: String should have at most 300 characters
  - `R02f621435777`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence "It's a miracle this thing didn't break my Keurig! Didn't fit properly and almost" 
  - `Rb3ddf496c9fa`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence "They were dented and still work, but I wasn't going to send them back just for t" 
  - `Rf28a59584713`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'very quiet, very movable, easy to use and install' is not a verbatim quote from th
  - `R00181bfd96a7`: Output still invalid after 2 repair attempt(s): aspects.2.aspect_category: Input should be 'performance', 'quality_durability', 'design_build', 'ease_of_use', '
  - `R1a135c0c927e`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `R6fae6e36a9fd`: Output still invalid after 2 repair attempt(s): rationale: String should have at most 400 characters
  - `Rd1d55023e6fd`: Output still invalid after 2 repair attempt(s): rationale: String should have at most 400 characters
  - `R1d1832973ec9`: Output still invalid after 2 repair attempt(s): aspects.2.evidence: evidence 'They even sent a marketing, yet personal Thank You Card, welcoming us to the fam' 
  - `R0c25bf5618a0`: Output still invalid after 2 repair attempt(s): complaint_categories.0: Input should be 'product_defect', 'poor_durability', 'performance_issue', 'noise_issue',
  - `R906ab7a68d21`: Output still invalid after 2 repair attempt(s): invalid JSON: Unterminated string starting at at line 154 column 5
  - `R755425de6ae7`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence 'They are providing space saving these containers have provided.' is not a verbatim
  - `R1c492cdbe101`: Output still invalid after 2 repair attempt(s): aspects.0.evidence: evidence "They're great for storing lots of eggs. They're perfect for the storage of those" 
  - `Ra5dbe3d71703`: Output still invalid after 2 repair attempt(s): rationale: String should have at most 400 characters
  - `Rc6cb1c5385f5`: Output still invalid after 2 repair attempt(s): aspects.0.aspect_category: Input should be 'performance', 'quality_durability', 'design_build', 'ease_of_use', '
  - `Rbe5f5a6a52de`: Output still invalid after 2 repair attempt(s): invalid JSON: Expecting ',' delimiter at line 8 column 63
