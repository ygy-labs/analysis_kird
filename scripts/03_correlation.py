from __future__ import annotations

import sys

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from common import (
    CONTROL_VARS,
    PROCESSED_DIR,
    PSYCH_SAFETY_ITEMS,
    VAR_LABELS,
    ensure_directories,
    p_label,
)


def pearson_matrix(df: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = df[variables].corr().round(3)
    p_values = pd.DataFrame(index=variables, columns=variables, dtype=float)
    for var1 in variables:
        for var2 in variables:
            if var1 == var2:
                p_values.loc[var1, var2] = 0.0
            else:
                _, p_value = stats.pearsonr(df[var1], df[var2])
                p_values.loc[var1, var2] = p_value
    return corr, p_values


def pairwise_correlation(df: pd.DataFrame, predictors: list[str], outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            r_value, p_value = stats.pearsonr(df[predictor], df[outcome])
            rows.append(
                {
                    "predictor": predictor,
                    "predictor_label": VAR_LABELS.get(predictor, predictor),
                    "outcome": outcome,
                    "outcome_label": VAR_LABELS.get(outcome, outcome),
                    "r": round(r_value, 3),
                    "p": p_label(p_value),
                }
            )
    return pd.DataFrame(rows)


def partial_corr(x: pd.Series, y: pd.Series, controls: pd.DataFrame) -> tuple[float, float]:
    controls_const = sm.add_constant(controls)
    x_resid = sm.OLS(x, controls_const).fit().resid
    y_resid = sm.OLS(y, controls_const).fit().resid
    return stats.pearsonr(x_resid, y_resid)


def partial_correlation_table(df: pd.DataFrame, predictors: list[str], outcomes: list[str], controls: list[str]) -> pd.DataFrame:
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            r_value, p_value = partial_corr(df[predictor], df[outcome], df[controls])
            rows.append(
                {
                    "predictor": predictor,
                    "predictor_label": VAR_LABELS.get(predictor, predictor),
                    "outcome": outcome,
                    "outcome_label": VAR_LABELS.get(outcome, outcome),
                    "partial_r": round(r_value, 3),
                    "p": p_label(p_value),
                    "controls": ", ".join(controls),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    try:
        ensure_directories()
        input_path = PROCESSED_DIR / "preprocessed_data.csv"
        if not input_path.exists():
            raise FileNotFoundError("processed/preprocessed_data.csv가 없습니다. 01 스크립트를 먼저 실행하세요.")

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        main_vars = ["psych_safety", "knowledge_sharing", "innovative_behavior"]
        outcomes = ["knowledge_sharing", "innovative_behavior"]
        control_dummy_cols = [
            col
            for control in CONTROL_VARS.values()
            for col in df.columns
            if col.startswith(f"{control}_")
        ]

        scale_corr, scale_p = pearson_matrix(df, main_vars)
        item_corr, item_p = pearson_matrix(df, PSYCH_SAFETY_ITEMS)
        item_outcome_corr = pairwise_correlation(df, PSYCH_SAFETY_ITEMS, outcomes)
        partial_item_outcome_corr = partial_correlation_table(df, PSYCH_SAFETY_ITEMS, outcomes, control_dummy_cols)
        partial_scale_corr = partial_correlation_table(df, ["psych_safety"], outcomes, control_dummy_cols)

        scale_corr.to_csv(PROCESSED_DIR / "correlation_scales_matrix.csv", encoding="utf-8-sig")
        scale_p.to_csv(PROCESSED_DIR / "correlation_scales_p_values.csv", encoding="utf-8-sig")
        item_corr.to_csv(PROCESSED_DIR / "correlation_psych_safety_items_matrix.csv", encoding="utf-8-sig")
        item_p.to_csv(PROCESSED_DIR / "correlation_psych_safety_items_p_values.csv", encoding="utf-8-sig")
        item_outcome_corr.to_csv(PROCESSED_DIR / "correlation_psych_safety_items_outcomes.csv", index=False, encoding="utf-8-sig")
        partial_item_outcome_corr.to_csv(
            PROCESSED_DIR / "partial_correlation_psych_safety_items_outcomes_controls.csv",
            index=False,
            encoding="utf-8-sig",
        )
        partial_scale_corr.to_csv(PROCESSED_DIR / "partial_correlation_scales_controls.csv", index=False, encoding="utf-8-sig")
        print("[완료] 03_correlation.py: 상관분석과 부분상관분석 결과를 저장했습니다.")
    except Exception as exc:
        print(f"[오류] 03_correlation.py 실행 실패: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
