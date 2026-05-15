from __future__ import annotations

import sys

import pandas as pd

from common import (
    CONTROL_VARS,
    PROCESSED_DIR,
    SCALES,
    VAR_LABELS,
    ensure_directories,
    resolve_raw_data_path,
)


def load_raw_excel() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = resolve_raw_data_path()
    raw = pd.read_excel(raw_path, sheet_name="DATA", header=None)
    labels = raw.iloc[1].tolist()
    codes = raw.iloc[2].tolist()
    data = raw.iloc[3:].copy()
    data.columns = codes

    mapping = pd.DataFrame(
        {
            "original_order": range(1, len(codes) + 1),
            "original_code": codes,
            "korean_label": labels,
        }
    )
    return data, mapping


def preprocess(data: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_vars = [item for items in SCALES.values() for item in items]
    control_codes = list(CONTROL_VARS)
    required_vars = item_vars + control_codes

    missing_columns = [col for col in required_vars if col not in data.columns]
    if missing_columns:
        raise KeyError(f"원자료에 필요한 컬럼이 없습니다: {missing_columns}")

    selected = data[required_vars].copy()
    before_rows = len(selected)

    for col in required_vars:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")

    missing_before_drop = selected.isna().sum().reset_index()
    missing_before_drop.columns = ["variable", "missing_count_before_drop"]

    # 본 설문 설계에서 사용한 문항들은 역문항으로 확인되지 않아 별도 역코딩을 적용하지 않았다.
    for scale_name, items in SCALES.items():
        selected[scale_name] = selected[items].mean(axis=1)

    selected = selected.rename(columns=CONTROL_VARS)
    analysis_required = item_vars + list(SCALES) + list(CONTROL_VARS.values())
    preprocessed = selected[analysis_required].dropna().copy()
    after_rows = len(preprocessed)

    dummy_frames = []
    dummy_mapping_rows = []
    for control in CONTROL_VARS.values():
        dummies = pd.get_dummies(preprocessed[control].astype("Int64").astype(str), prefix=control, drop_first=True)
        dummies = dummies.astype(int)
        dummy_frames.append(dummies)
        for col in dummies.columns:
            dummy_mapping_rows.append(
                {
                    "original_variable": control,
                    "dummy_variable": col,
                    "reference_category": sorted(preprocessed[control].dropna().astype(int).unique())[0],
                    "encoded_category": col.split("_")[-1],
                }
            )

    if dummy_frames:
        preprocessed = pd.concat([preprocessed] + dummy_frames, axis=1)

    mapping["analysis_name"] = mapping["original_code"].map(CONTROL_VARS).fillna(mapping["original_code"])
    mapping["analysis_label"] = mapping["analysis_name"].map(VAR_LABELS)

    summary = pd.DataFrame(
        [
            {"item": "raw_rows", "value": before_rows, "note": "원자료 응답자 수"},
            {"item": "analysis_rows", "value": after_rows, "note": "결측 제거 후 분석 표본"},
            {"item": "removed_rows", "value": before_rows - after_rows, "note": "분석 변수 결측으로 제거된 행"},
            {"item": "reverse_coding", "value": 0, "note": "확인된 역문항 없음"},
            {"item": "scale_variables_created", "value": len(SCALES), "note": "문항 평균 척도 수"},
            {"item": "dummy_variables_created", "value": sum(len(frame.columns) for frame in dummy_frames), "note": "통제변수 더미 수"},
        ]
    )

    missing_before_drop.to_csv(PROCESSED_DIR / "missing_summary_before_drop.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dummy_mapping_rows).to_csv(PROCESSED_DIR / "control_dummy_mapping.csv", index=False, encoding="utf-8-sig")
    return preprocessed, mapping, summary


def main() -> None:
    try:
        ensure_directories()
        data, mapping = load_raw_excel()
        preprocessed, column_mapping, summary = preprocess(data, mapping)

        preprocessed.to_csv(PROCESSED_DIR / "preprocessed_data.csv", index=False, encoding="utf-8-sig")
        column_mapping.to_csv(PROCESSED_DIR / "column_mapping.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(PROCESSED_DIR / "preprocess_summary.csv", index=False, encoding="utf-8-sig")
        print("[완료] 01_load_and_preprocess.py: 전처리 데이터와 요약 파일을 저장했습니다.")
    except Exception as exc:
        print(f"[오류] 01_load_and_preprocess.py 실행 실패: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
