from __future__ import annotations

import sys

import pandas as pd

from common import (
    PROCESSED_DIR,
    SCALES,
    VAR_LABELS,
    ensure_directories,
    judgment_alpha,
)


def cronbach_alpha(data: pd.DataFrame) -> float:
    clean = data.dropna()
    item_count = clean.shape[1]
    item_variances = clean.var(axis=0, ddof=1)
    total_variance = clean.sum(axis=1).var(ddof=1)
    return item_count / (item_count - 1) * (1 - item_variances.sum() / total_variance)


def reliability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale_name, items in SCALES.items():
        alpha = cronbach_alpha(df[items])
        rows.append(
            {
                "scale": scale_name,
                "label": VAR_LABELS[scale_name],
                "items": ", ".join(items),
                "n_items": len(items),
                "cronbach_alpha": round(alpha, 3),
                "judgment": judgment_alpha(alpha),
            }
        )
    return pd.DataFrame(rows)


def descriptive_table(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    desc = df[variables].agg(["count", "mean", "std", "min", "max", "skew"]).T
    desc["kurtosis"] = df[variables].kurtosis()
    desc["missing_count"] = df[variables].isna().sum()
    desc.insert(0, "label", [VAR_LABELS.get(var, var) for var in variables])
    return desc.round(3)


def main() -> None:
    try:
        ensure_directories()
        input_path = PROCESSED_DIR / "preprocessed_data.csv"
        if not input_path.exists():
            raise FileNotFoundError("processed/preprocessed_data.csv가 없습니다. 01 스크립트를 먼저 실행하세요.")

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        main_vars = ["psych_safety", "knowledge_sharing", "innovative_behavior"]
        item_vars = [item for items in SCALES.values() for item in items]

        reliability = reliability_table(df)
        desc_main = descriptive_table(df, main_vars)
        desc_items = descriptive_table(df, item_vars)

        reliability.to_csv(PROCESSED_DIR / "reliability.csv", index=False, encoding="utf-8-sig")
        desc_main.to_csv(PROCESSED_DIR / "descriptive_statistics_main.csv", encoding="utf-8-sig")
        desc_items.to_csv(PROCESSED_DIR / "descriptive_statistics_items.csv", encoding="utf-8-sig")
        print("[완료] 02_reliability_and_descriptive.py: 신뢰도와 기술통계 결과를 저장했습니다.")
    except Exception as exc:
        print(f"[오류] 02_reliability_and_descriptive.py 실행 실패: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
