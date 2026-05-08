from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


DATA_PATH = Path(
    r"C:\Users\rhksd\Desktop\Data\활동조사_재직자_데이터\활동조사_재직자_데이터_어승수.xlsx"
)
OUTPUT_DIR = Path("outputs")


SCALES = {
    "psych_safety": ["Q12_1", "Q12_2", "Q12_3", "Q12_4"],
    "knowledge_sharing": ["Q19_7", "Q19_8", "Q19_9", "Q19_10"],
    "innovative_behavior": ["Q28_4", "Q28_5", "Q28_6", "Q28_7"],
}

CONTROL_VARS = {
    "SQ1": "gender",
    "SQ2_1": "age_group",
    "SQ3": "org_type",
    "QT1": "job_type",
}


def cronbach_alpha(data: pd.DataFrame) -> float:
    clean = data.dropna()
    item_count = clean.shape[1]
    item_variances = clean.var(axis=0, ddof=1)
    total_variance = clean.sum(axis=1).var(ddof=1)
    return item_count / (item_count - 1) * (1 - item_variances.sum() / total_variance)


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
    raw = pd.read_excel(DATA_PATH, sheet_name="DATA", header=None)
    labels = raw.iloc[1].tolist()
    codes = raw.iloc[2].tolist()
    data = raw.iloc[3:].copy()
    data.columns = codes
    label_map = dict(zip(codes, labels))
    return data, label_map


def prepare_analysis_data(data: pd.DataFrame) -> pd.DataFrame:
    selected = [item for items in SCALES.values() for item in items] + list(CONTROL_VARS)
    analysis = data[selected].copy()
    analysis = analysis.apply(pd.to_numeric, errors="coerce")

    for scale_name, items in SCALES.items():
        analysis[scale_name] = analysis[items].mean(axis=1)

    analysis = analysis.rename(columns=CONTROL_VARS)
    model_vars = list(SCALES) + list(CONTROL_VARS.values())
    return analysis[model_vars].dropna()


def descriptive_statistics(analysis: pd.DataFrame) -> pd.DataFrame:
    scale_names = list(SCALES)
    desc = analysis[scale_names].agg(["count", "mean", "std", "min", "max"]).T
    return desc.round(3)


def reliability_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale_name, items in SCALES.items():
        item_data = data[items].apply(pd.to_numeric, errors="coerce")
        rows.append(
            {
                "variable": scale_name,
                "items": ", ".join(items),
                "n_items": len(items),
                "cronbach_alpha": round(cronbach_alpha(item_data), 3),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scale_names = list(SCALES)
    corr = analysis[scale_names].corr().round(3)
    p_values = pd.DataFrame(np.ones((len(scale_names), len(scale_names))), index=scale_names, columns=scale_names)

    for row in scale_names:
        for col in scale_names:
            if row != col:
                _, p_value = stats.pearsonr(analysis[row], analysis[col])
                p_values.loc[row, col] = p_value

    return corr, p_values


def run_regression(analysis: pd.DataFrame, dependent_var: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    predictors = ["psych_safety", "gender", "age_group", "org_type", "job_type"]
    x = sm.add_constant(analysis[predictors])
    y = analysis[dependent_var]
    return sm.OLS(y, x).fit()


def regression_table(model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "B": model.params,
            "SE": model.bse,
            "t": model.tvalues,
            "p": model.pvalues,
            "CI_lower": model.conf_int()[0],
            "CI_upper": model.conf_int()[1],
        }
    )
    rounded = table.copy()
    for col in ["B", "SE", "t", "CI_lower", "CI_upper"]:
        rounded[col] = rounded[col].round(3)
    rounded["p"] = rounded["p"].map(p_label)
    return rounded


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data, label_map = load_data()
    numeric_data = data.copy()
    for items in SCALES.values():
        numeric_data[items] = numeric_data[items].apply(pd.to_numeric, errors="coerce")

    analysis = prepare_analysis_data(data)
    reliability = reliability_statistics(numeric_data)
    desc = descriptive_statistics(analysis)
    corr, corr_p = correlation_matrix(analysis)

    model_ks = run_regression(analysis, "knowledge_sharing")
    model_ib = run_regression(analysis, "innovative_behavior")
    reg_ks = regression_table(model_ks)
    reg_ib = regression_table(model_ib)

    reliability.to_csv(OUTPUT_DIR / "reliability.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", encoding="utf-8-sig")
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv", encoding="utf-8-sig")
    corr_p.to_csv(OUTPUT_DIR / "correlation_p_values.csv", encoding="utf-8-sig")
    reg_ks.to_csv(OUTPUT_DIR / "regression_knowledge_sharing.csv", encoding="utf-8-sig")
    reg_ib.to_csv(OUTPUT_DIR / "regression_innovative_behavior.csv", encoding="utf-8-sig")

    print("Sample size:", len(analysis))
    print("\nReliability")
    print(reliability.to_string(index=False))
    print("\nDescriptive statistics")
    print(desc.to_string())
    print("\nCorrelation matrix")
    print(corr.to_string())
    print("\nRegression: knowledge sharing")
    print(reg_ks.to_string())
    print(f"R2={model_ks.rsquared:.3f}, Adj.R2={model_ks.rsquared_adj:.3f}, F p={p_label(model_ks.f_pvalue)}")
    print("\nRegression: innovative behavior")
    print(reg_ib.to_string())
    print(f"R2={model_ib.rsquared:.3f}, Adj.R2={model_ib.rsquared_adj:.3f}, F p={p_label(model_ib.f_pvalue)}")


if __name__ == "__main__":
    main()
