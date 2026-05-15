from __future__ import annotations

import platform
import sys
from pathlib import Path

import pandas as pd

from common import FIGURES_DIR, PROCESSED_DIR, PROJECT_ROOT, VAR_LABELS, ensure_directories


README_PATH = PROJECT_ROOT / "README.md"


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} 파일이 없습니다. 앞 단계 스크립트를 먼저 실행하세요.")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def matrix_to_markdown(df: pd.DataFrame) -> str:
    df = df.copy()
    df.insert(0, "변수", df.index)
    return df_to_markdown(df.reset_index(drop=True))


def existing_figures_markdown() -> str:
    figures = [
        ("주요 변수 분포", "processed/figures/01_main_variable_distributions.png"),
        ("주요 변수 상관관계 히트맵", "processed/figures/02_correlation_heatmap.png"),
        ("심리적 안전감 하위 문항 부분상관", "processed/figures/03_partial_correlations_by_item.png"),
        ("심리적 안전감 표준화 회귀계수", "processed/figures/04_regression_standardized_beta.png"),
    ]
    lines = []
    for title, rel_path in figures:
        if (PROJECT_ROOT / rel_path).exists():
            lines.append(f"**{title}**\n\n![{title}]({rel_path})")
    return "\n\n".join(lines)


def main() -> None:
    try:
        ensure_directories()

        preprocess_summary = read_csv("preprocess_summary.csv")
        reliability = read_csv("reliability.csv")
        desc_main = read_csv("descriptive_statistics_main.csv")
        desc_items = read_csv("descriptive_statistics_items.csv")
        scale_corr = read_csv("correlation_scales_matrix.csv", index_col=0)
        item_corr = read_csv("correlation_psych_safety_items_matrix.csv", index_col=0)
        partial_item = read_csv("partial_correlation_psych_safety_items_outcomes_controls.csv")
        model_fit = read_csv("regression_model_fit.csv")
        reg_coef = read_csv("regression_coefficients.csv")
        item_coef = read_csv("regression_psych_safety_item_coefficients.csv")

        desc_main_named = desc_main.rename(columns={"Unnamed: 0": "variable"})
        desc_items_named = desc_items.rename(columns={"Unnamed: 0": "variable"})
        scale_corr_named = scale_corr.rename(index=VAR_LABELS, columns=VAR_LABELS)
        item_corr_named = item_corr.rename(index=VAR_LABELS, columns=VAR_LABELS)

        psych_rows = reg_coef[reg_coef["term"] == "psych_safety"].copy()
        best_item_rows = item_coef[item_coef["term"] == "Q12_3"].copy()

        readme = f"""# 심리적 안전감이 지식공유와 혁신행동에 미치는 영향 - 분석 보고서

## 참고사항

이 저장소는 KIRD 「과학기술 인재개발 활동조사」 재직자 데이터를 활용하여 심리적 안전감이 지식공유와 혁신행동에 미치는 영향을 분석하는 소규모 분석 프로젝트입니다.

- 원자료 엑셀 파일은 데이터 사용 제한 가능성이 있어 저장소에 포함하지 않습니다.
- 분석 실행 후 생성되는 CSV 파일은 `processed/`에 저장됩니다.
- 그래프 이미지는 `processed/figures/`에 저장됩니다.
- README의 수치는 `scripts/06_generate_readme.py`가 실제 분석 결과 CSV를 읽어 작성합니다.
- 다른 컴퓨터에서 실행할 경우 `data/raw_data.xlsx`를 두거나 환경변수 `KIRD_RAW_DATA`를 지정해야 합니다.
- `processed/preprocessed_data.csv`는 행 단위 파생 데이터이므로 공개 GitHub 업로드에서는 제외하고, 로컬에서 재생성하는 것을 권장합니다.

실행 순서는 아래와 같습니다.

```powershell
python scripts\\01_load_and_preprocess.py
python scripts\\02_reliability_and_descriptive.py
python scripts\\03_correlation.py
python scripts\\04_regression_or_modeling.py
python scripts\\05_visualization.py
python scripts\\06_generate_readme.py
```

## 폴더 구조

| 경로 | 설명 |
|---|---|
| `data/` | 원자료 위치 안내. 원자료는 GitHub 업로드 제외 |
| `scripts/` | 번호 순서대로 실행 가능한 분석 스크립트 |
| `processed/` | 전처리 데이터와 분석 결과 CSV 저장 위치 |
| `processed/figures/` | 주요 시각화 이미지 저장 위치 |
| `README.md` | 분석 보고서 |
| `requirements.txt` | 재현 실행에 필요한 Python 패키지 목록 |

## 환경 준비

- Python 버전: `{platform.python_version()}`에서 검증
- 주요 패키지: pandas, numpy, scipy, statsmodels, openpyxl, pillow

패키지 설치:

```bash
pip install -r requirements.txt
```

## 분석 파이프라인

| 실행 순서 | 스크립트 | 역할 | 주요 산출물 |
|---:|---|---|---|
| 1 | `scripts/01_load_and_preprocess.py` | 데이터 로드, 컬럼 매핑, 결측 제거, 척도 평균 생성, 통제변수 더미 인코딩 | `processed/preprocessed_data.csv`, `processed/column_mapping.csv`, `processed/preprocess_summary.csv` |
| 2 | `scripts/02_reliability_and_descriptive.py` | 신뢰도 및 기술통계 계산 | `processed/reliability.csv`, `processed/descriptive_statistics_main.csv`, `processed/descriptive_statistics_items.csv` |
| 3 | `scripts/03_correlation.py` | Pearson 상관, 하위 문항 상관, 통제변수 포함 부분상관 | `processed/correlation_scales_matrix.csv`, `processed/partial_correlation_psych_safety_items_outcomes_controls.csv` |
| 4 | `scripts/04_regression_or_modeling.py` | 위계적 다중회귀 및 하위 문항 회귀분석 | `processed/regression_model_fit.csv`, `processed/regression_coefficients.csv` |
| 5 | `scripts/05_visualization.py` | 주요 그래프 생성 | `processed/figures/*.png` |
| 6 | `scripts/06_generate_readme.py` | 분석 결과 CSV를 읽어 README 보고서 생성 | `README.md` |

전처리 요약:

주의: `processed/preprocessed_data.csv`는 로컬 재현 실행 시 생성되는 행 단위 분석 데이터입니다. 데이터 사용 제한 가능성을 고려하여 공개 GitHub 업로드에서는 제외하는 것을 권장합니다.

{df_to_markdown(preprocess_summary)}

## 1. 변수 선정

| 구분 | 원본 컬럼명 | 분석용 변수명 | 척도 또는 타입 | 선정 이유 |
|---|---|---|---|---|
| 독립변수 | `Q12_1`, `Q12_2`, `Q12_3`, `Q12_4` | `psych_safety` | 1~5점 문항 평균 | 조직 내 심리적 안전감 수준 측정 |
| 종속변수 | `Q19_7`, `Q19_8`, `Q19_9`, `Q19_10` | `knowledge_sharing` | 1~5점 문항 평균 | 동료 간 지식공유 행동 측정 |
| 종속변수 | `Q28_4`, `Q28_5`, `Q28_6`, `Q28_7` | `innovative_behavior` | 1~5점 문항 평균 | 업무 개선과 새로운 아이디어 적용 행동 측정 |
| 통제변수 | `SQ1` | `gender` | 범주형 더미 인코딩 | 성별 차이 통제 |
| 통제변수 | `SQ2_1` | `age_group` | 범주형 더미 인코딩 | 연령대 차이 통제 |
| 통제변수 | `SQ3` | `org_type` | 범주형 더미 인코딩 | 기관 유형 차이 통제 |
| 통제변수 | `QT1` | `job_type` | 범주형 더미 인코딩 | 연구개발직/연구지원직 차이 통제 |

역문항 처리는 적용하지 않았습니다. 현재 제공된 문항명 기준으로 명시적인 역문항이 확인되지 않았으므로, 추가 검토 필요 항목으로 남깁니다.

## 2. 이론적 배경 및 분석 가설

사용자가 제공한 공모문과 조사 개요에 따르면 이 데이터는 과학기술 분야 재직자의 인재개발 활동, 개인 특성, 조직 환경, 성과 변수를 포함합니다. 본 프로젝트는 그중 심리적 안전감, 지식공유, 혁신행동의 관계에 초점을 둡니다.

- 가설 1: 심리적 안전감은 지식공유에 정적인 영향을 미칠 것이다.
- 가설 2: 심리적 안전감은 혁신행동에 정적인 영향을 미칠 것이다.
- 가설 3: 성별, 연령대, 기관 유형, 직무구분을 통제한 뒤에도 심리적 안전감의 영향은 유지될 것이다.

심리적 안전감과 혁신행동의 이론적 연결에 대한 외부 선행연구 인용은 본 프로젝트 범위에서 별도로 수집하지 않았습니다. 따라서 참고문헌 보강은 추가 검토 필요입니다.

## 3. 신뢰도 분석

{df_to_markdown(reliability)}

세 척도의 Cronbach's alpha는 모두 0.70 이상으로 나타났습니다. 따라서 문항 평균을 척도 점수로 사용하는 것은 수용 가능하다고 판단했습니다.

## 4. 기술통계 분석

### 4.1 독립변수 및 종속변수

{df_to_markdown(desc_main_named)}

주요 변수의 평균은 모두 3점 이상으로 나타났습니다. 왜도는 음수 방향으로 나타나 높은 점수 쪽 응답이 상대적으로 많은 편입니다. 첨도는 과도하게 크지 않아 극단적인 분포 문제는 크지 않은 것으로 보입니다.

### 4.2 문항 수준 기술통계

{df_to_markdown(desc_items_named)}

## 5. 상관분석

### 5.1 주요 척도 상관계수

{matrix_to_markdown(scale_corr_named)}

심리적 안전감은 지식공유 및 혁신행동과 모두 정적 상관을 보였습니다. 지식공유와 혁신행동 사이에도 정적 상관이 확인되었습니다.

### 5.2 심리적 안전감 하위 문항 상관계수

{matrix_to_markdown(item_corr_named)}

심리적 안전감 하위 문항 간 상관은 전반적으로 중간 이상 수준입니다. 문항 간 관련성이 높지만 0.80을 넘는 수준은 아니므로, 심각한 중복 문항이라고 단정하기는 어렵습니다. 다만 회귀분석에서 하위 문항을 동시에 투입할 경우 다중공선성 가능성은 추가 검토가 필요합니다.

### 5.3 통제변수 포함 부분상관

{df_to_markdown(partial_item[["predictor", "predictor_label", "outcome", "outcome_label", "partial_r", "p"]])}

성별, 연령대, 소속기관 유형, 직무구분을 더미변수로 통제한 뒤에도 심리적 안전감 하위 문항은 지식공유와 혁신행동에 대체로 유의한 정적 관련을 보였습니다.

## 6. 주요 분석 결과

### 6.1 위계적 다중회귀 모형 설명력

{df_to_markdown(model_fit)}

통제변수만 투입한 모형과 통제변수에 심리적 안전감을 추가한 모형을 비교했습니다. 심리적 안전감 추가 후 지식공유와 혁신행동 모두에서 설명력이 증가했습니다.

### 6.2 심리적 안전감 척도 회귀계수

{df_to_markdown(psych_rows[["outcome", "label", "B", "SE", "std_beta", "t", "p", "CI_lower", "CI_upper"]])}

### 6.3 심리적 안전감 하위 문항 회귀계수

{df_to_markdown(item_coef[item_coef["term"].isin(["Q12_1", "Q12_2", "Q12_3", "Q12_4"])][["outcome", "term", "label", "B", "SE", "std_beta", "t", "p"]])}

하위 문항을 동시에 투입한 모형에서는 `Q12_3` 구성원 간 도움 요청의 자유로움이 지식공유와 혁신행동 모두에서 유의한 변수로 나타났습니다.

## 7. 분석 결과 해석

심리적 안전감은 통제변수를 고려한 뒤에도 지식공유와 혁신행동에 유의한 정적 영향을 보였습니다. 특히 지식공유 모형에서 추가 설명력 증가가 더 크게 나타났습니다.

가장 일관되게 영향력이 큰 하위 문항은 `Q12_3` 구성원 간 도움 요청의 자유로움입니다. 이는 과학기술 분야 재직자가 자유롭게 도움을 요청할 수 있는 환경에서 지식공유가 활발해지고, 새로운 아이디어를 시도하는 혁신행동도 높아질 가능성을 시사합니다.

예상과 일치한 결과는 심리적 안전감이 두 종속변수 모두와 정적 관계를 보였다는 점입니다. 예상과 다른 결과 또는 추가 검토가 필요한 결과는 하위 문항 동시 투입 시 일부 문항의 회귀계수가 유의하지 않게 나타난 점입니다. 이는 하위 문항 간 상관이 존재하기 때문일 수 있으므로 다중공선성 진단은 추가 검토가 필요합니다.

실무적 또는 정책적 시사점은 교육훈련 프로그램뿐 아니라 조직 내 질문, 도움 요청, 상호 존중이 가능한 환경을 조성하는 것이 인재개발 성과와 혁신행동을 높이는 데 중요할 수 있다는 점입니다.

## 8. 데이터 출처·한계

- 데이터 출처: 사용자가 제공한 KIRD 「과학기술 인재개발 활동조사」 재직자 데이터
- 표본 한계: 현재 분석은 제공된 재직자 데이터 2,000명에 한정됩니다.
- 변수 한계: 통제변수의 범주명은 코드북을 통해 추가 확인이 필요합니다.
- 분석 방법 한계: 횡단면 설문자료이므로 인과관계를 단정할 수 없습니다.
- 측정 한계: 자기보고식 설문이므로 동일방법편의 가능성이 있습니다.
- 문항 처리 한계: 역문항 여부는 문항명 기준으로 판단했으며, 공식 코드북 대조는 추가 검토 필요입니다.

## 참고문헌 또는 참고자료

- 사용자가 제공한 KIRD 「과학기술 인재개발 활동조사」 데이터 활용 논문 공모문 및 조사 개요
- 추가 작성 필요

## 주요 시각화

{existing_figures_markdown()}

## 문제가 생길 때

| 문제 유형 | 해결 방법 |
|---|---|
| 파일 경로 오류 | 원자료를 `data/raw_data.xlsx`로 복사하거나 PowerShell에서 `$env:KIRD_RAW_DATA = "원자료경로"`를 설정합니다. |
| ModuleNotFoundError | `pip install -r requirements.txt`를 실행합니다. 가상환경을 사용 중이면 활성화 여부를 확인합니다. |
| UnicodeDecodeError | CSV 파일은 `utf-8-sig`로 저장됩니다. 엑셀에서 깨질 경우 데이터 가져오기 기능에서 UTF-8을 선택합니다. |
| 한글 깨짐 | Windows에서는 Malgun Gothic을 사용하도록 설정했습니다. 다른 OS에서는 한글 폰트 설치가 필요할 수 있습니다. |
| 그래프 저장 오류 | `processed/figures/` 폴더 권한을 확인하고, `05_visualization.py`를 다시 실행합니다. |
| 결측치 처리 오류 | `processed/missing_summary_before_drop.csv`를 확인하여 어떤 변수에서 결측이 발생했는지 점검합니다. |
"""

        README_PATH.write_text(readme, encoding="utf-8")
        print("[완료] 06_generate_readme.py: README.md 분석 보고서를 생성했습니다.")
    except Exception as exc:
        print(f"[오류] 06_generate_readme.py 실행 실패: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
