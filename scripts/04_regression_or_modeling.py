from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (
    CONTROL_VARS,
    PROCESSED_DIR,
    PSYCH_SAFETY_ITEMS,
    VAR_LABELS,
    ensure_directories,
    p_label,
)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std


def fit_ols(df: pd.DataFrame, outcome: str, predictors: list[str]):
    x = sm.add_constant(df[predictors])
    y = df[outcome]
    return sm.OLS(y, x).fit()


def standardized_betas(df: pd.DataFrame, outcome: str, predictors: list[str]) -> pd.Series:
    standardized = pd.DataFrame({col: zscore(df[col]) for col in predictors})
    standardized[outcome] = zscore(df[outcome])
    model = fit_ols(standardized, outcome, predictors)
    return model.params.drop("const")


def coefficient_table(model, beta: pd.Series, outcome: str, model_name: str) -> pd.DataFrame:
    conf_int = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append(
            {
                "outcome": outcome,
                "model": model_name,
                "term": term,
                "label": VAR_LABELS.get(term, term),
                "B": round(model.params[term], 3),
                "SE": round(model.bse[term], 3),
                "std_beta": "" if term == "const" else round(beta.get(term, np.nan), 3),
                "t": round(model.tvalues[term], 3),
                "p": p_label(model.pvalues[term]),
                "CI_lower": round(conf_int.loc[term, 0], 3),
                "CI_upper": round(conf_int.loc[term, 1], 3),
            }
        )
    return pd.DataFrame(rows)


def model_fit_row(model, outcome: str, model_name: str, previous_r2: float | None) -> dict[str, object]:
    r2 = model.rsquared
    return {
        "outcome": outcome,
        "model": model_name,
        "R2": round(r2, 3),
        "Adj_R2": round(model.rsquared_adj, 3),
        "Delta_R2": "" if previous_r2 is None else round(r2 - previous_r2, 3),
        "F": round(model.fvalue, 3),
        "F_p": p_label(model.f_pvalue),
        "n": int(model.nobs),
    }


def main() -> None:
    try:
        ensure_directories()
        input_path = PROCESSED_DIR / "preprocessed_data.csv"
        if not input_path.exists():
            raise FileNotFoundError("processed/preprocessed_data.csv가 없습니다. 01 스크립트를 먼저 실행하세요.")

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        outcomes = ["knowledge_sharing", "innovative_behavior"]
        control_dummy_cols = [
            col
            for control in CONTROL_VARS.values()
            for col in df.columns
            if col.startswith(f"{control}_")
        ]

        all_coefficients = []
        all_fit_rows = []
        item_model_coefficients = []

        for outcome in outcomes:
            model1_predictors = control_dummy_cols
            model2_predictors = control_dummy_cols + ["psych_safety"]
            item_predictors = control_dummy_cols + PSYCH_SAFETY_ITEMS

            model1 = fit_ols(df, outcome, model1_predictors)
            model2 = fit_ols(df, outcome, model2_predictors)
            item_model = fit_ols(df, outcome, item_predictors)

            all_fit_rows.append(model_fit_row(model1, outcome, "Model 1: controls", None))
            all_fit_rows.append(model_fit_row(model2, outcome, "Model 2: controls + psych_safety", model1.rsquared))

            beta_model2 = standardized_betas(df, outcome, model2_predictors)
            beta_item = standardized_betas(df, outcome, item_predictors)
            all_coefficients.append(coefficient_table(model2, beta_model2, outcome, "Model 2: controls + psych_safety"))
            item_model_coefficients.append(coefficient_table(item_model, beta_item, outcome, "Item model: controls + Q12 items"))

        pd.concat(all_coefficients, ignore_index=True).to_csv(
            PROCESSED_DIR / "regression_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(all_fit_rows).to_csv(PROCESSED_DIR / "regression_model_fit.csv", index=False, encoding="utf-8-sig")
        pd.concat(item_model_coefficients, ignore_index=True).to_csv(
            PROCESSED_DIR / "regression_psych_safety_item_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("[완료] 04_regression_or_modeling.py: 회귀분석 결과를 저장했습니다.")
    except Exception as exc:
        print(f"[오류] 04_regression_or_modeling.py 실행 실패: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
