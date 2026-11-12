# Data

## Source

**Amazon Reviews 2023**, McAuley Lab, UC San Diego. Website: https://amazon-reviews-2023.github.io/

> Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for
> Retrieval and Recommendation.* arXiv:2403.03952.

* Category used: **Appliances** (2,128,605 raw reviews). Configure with `DATASET_CATEGORY`.
* Files (downloaded automatically by `scripts/prepare_dataset.py`):
  * `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz` (≈270 MB)
  * `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz` (≈66 MB, used for product titles)
* **Licence / terms:** the dataset is released by the McAuley Lab for research use. It is used here for
  non-commercial academic work. Raw and processed review files are **not** committed to git (see
  `.gitignore`); anyone can regenerate them deterministically with the script. The 300-review annotation
  sample in `data/annotations/` is committed so that the reference labels can be read alongside the text.
  Check the dataset website for current terms before redistributing.

## Fields used

| Raw field | Project column | Notes |
|---|---|---|
| `parent_asin` | `product_id` | Groups colour/size variants of the same product |
| meta `title` | `product_title` | Cleaned, truncated to 200 characters |
| `text` | `review_text` | HTML tags removed, entities unescaped, whitespace collapsed |
| `title` | `review_title` | Not shown to the LLM |
| `rating` | `rating` | Used only for sampling and display, **never** given to the LLM |
| `timestamp` (ms) | `timestamp` | Converted to `YYYY-MM-DD`; used for aspect trends |
| `user_id`, `asin`, `timestamp` | `review_id` | `"R" + sha1(user_id|asin|timestamp)[:12]` (stable, anonymised) |
| `verified_purchase`, `helpful_vote` | same | Kept for reference |

## Preparation pipeline (`utils/data_loader.prepare_dataset`)

1. **Pass 1:** count valid reviews (5–400 words after cleaning) per product.
2. Keep candidate products with 200–5,000 valid reviews and a title (1,373 products).
3. Randomly select **15 products** (seed 42).
4. **Pass 2:** for the selected products, drop invalid lengths (1,406), duplicate user+product reviews (70)
   and duplicate texts within a product (6).
5. Sample up to **100 reviews per product** (seeded) → **1,500 reviews** in `processed/reviews.csv`.
6. Write the statistics to `processed/dataset_card.json`.

Final rating distribution: 1★ 150 · 2★ 55 · 3★ 64 · 4★ 145 · 5★ 1,086.

## Folders

| Folder | Content | In git? |
|---|---|---|
| `raw/` | Downloaded `.jsonl.gz` files | no |
| `processed/` | `reviews.csv`, `dataset_card.json` | no (regenerate) |
| `annotations/` | `annotation_sample.csv` (300 reviews), `annotation_template.csv`, `annotations.json` (reference labels: 1 human, 299 AI-assisted; see `meta.annotation_method`) | yes |

Regenerate everything with:

```bash
python scripts/prepare_dataset.py
python scripts/build_annotation_sample.py   # keeps an existing sample unless --force
```
