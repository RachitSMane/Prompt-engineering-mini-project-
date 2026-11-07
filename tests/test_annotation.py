import pandas as pd
import pytest
from pydantic import ValidationError

from annotation.annotation_manager import (
    ANNOTATION_CSV_COLUMNS,
    AnnotationStore,
    build_annotation_sample,
    load_sample,
    parse_aspects_cell,
    save_sample,
    write_template,
)
from schemas.evaluation_schema import AnnotatedAspect, Annotation


def _reviews(n=1000):
    return pd.DataFrame(
        {
            "review_id": [f"R{i:04d}" for i in range(n)],
            "product_id": [f"P{i % 7}" for i in range(n)],
            "product_title": "t",
            "rating": [5.0 if i % 10 < 6 else float(1 + (i // 10) % 4) for i in range(n)],  # 60% 5-star, ~100 of each other rating
            "review_text": [f"synthetic review {i}" for i in range(n)],
        }
    )


def _ann(rid="R0001", **kw):
    data = dict(review_id=rid, product_id="P1", overall_sentiment="mixed", annotator="tester",
                aspects=[AnnotatedAspect(aspect_category="noise", sentiment="negative")], complaint_categories=["noise_issue"])
    data.update(kw)
    return Annotation(**data)


def test_sample_is_reproducible_stratified_and_unique():
    a = build_annotation_sample(_reviews(), 300, seed=42)
    b = build_annotation_sample(_reviews(), 300, seed=42)
    assert len(a) == 300 and a["review_id"].is_unique
    assert a["review_id"].tolist() == b["review_id"].tolist()
    counts = a["rating"].value_counts()
    assert counts.to_dict() == {1.0: 60, 2.0: 60, 3.0: 60, 4.0: 60, 5.0: 60}  # balanced despite skewed source


def test_existing_sample_not_overwritten(tmp_path):
    path = tmp_path / "s.csv"
    s = build_annotation_sample(_reviews(), 50, seed=1)
    assert save_sample(s, path) is True
    assert save_sample(build_annotation_sample(_reviews(), 50, seed=2), path) is False
    assert load_sample(path)["review_id"].tolist() == s["review_id"].tolist()


def test_store_persists_across_instances(tmp_path):
    path = tmp_path / "annotations.json"
    store = AnnotationStore(path)
    store.save(_ann())
    store.save(_ann("R0002", overall_sentiment="positive", aspects=[], complaint_categories=[]))
    reloaded = AnnotationStore(path)
    assert len(reloaded.all()) == 2
    assert reloaded.get("R0001").aspects[0].sentiment.value == "negative"
    reloaded.delete("R0002")
    assert AnnotationStore(path).get("R0002") is None


def test_progress_and_flagged_excluded(tmp_path):
    store = AnnotationStore(tmp_path / "a.json")
    store.save(_ann("R1"))
    store.save(_ann("R2", status="flagged"))
    prog = store.progress(["R1", "R2", "R3"])
    assert prog == {"total": 3, "complete": 1, "flagged": 1, "remaining": 1}
    assert [a.review_id for a in store.completed()] == ["R1"]


def test_duplicate_category_rejected():
    with pytest.raises(ValidationError):
        _ann(aspects=[AnnotatedAspect(aspect_category="noise", sentiment="negative"), AnnotatedAspect(aspect_category="noise", sentiment="positive")])


def test_csv_export_import_roundtrip(tmp_path):
    store = AnnotationStore(tmp_path / "a.json")
    store.save(_ann())
    df = store.to_dataframe()
    assert list(df.columns) == ANNOTATION_CSV_COLUMNS
    assert df.loc[0, "aspects"] == "noise:negative"
    fresh = AnnotationStore(tmp_path / "b.json")
    res = fresh.import_csv(df)
    assert res["imported"] == 1 and not res["errors"]
    assert fresh.get("R0001").complaint_categories[0].value == "noise_issue"


def test_import_reports_bad_rows_and_skips_blank(tmp_path):
    df = pd.DataFrame(
        [
            {"review_id": "A", "product_id": "P", "overall_sentiment": "positive", "aspects": "price_value:positive", "complaint_categories": "", "annotator": "x"},
            {"review_id": "B", "product_id": "P", "overall_sentiment": "", "aspects": "", "complaint_categories": "", "annotator": ""},
            {"review_id": "C", "product_id": "P", "overall_sentiment": "great", "aspects": "", "complaint_categories": "", "annotator": "x"},
            {"review_id": "D", "product_id": "P", "overall_sentiment": "negative", "aspects": "noise-bad", "complaint_categories": "", "annotator": "x"},
        ]
    )
    res = AnnotationStore(tmp_path / "a.json").import_csv(df)
    assert res["imported"] == 1 and res["skipped_unlabelled"] == 1 and len(res["errors"]) == 2


def test_template_has_all_sample_rows(tmp_path):
    s = build_annotation_sample(_reviews(), 20, seed=3)
    write_template(s, tmp_path / "t.csv")
    t = pd.read_csv(tmp_path / "t.csv")
    assert list(t.columns) == ANNOTATION_CSV_COLUMNS and len(t) == 20 and t["overall_sentiment"].isna().all()


def test_parse_aspects_cell():
    out = parse_aspects_cell("Noise:Negative; performance:positive")
    assert [(a.aspect_category.value, a.sentiment.value) for a in out] == [("noise", "negative"), ("performance", "positive")]
