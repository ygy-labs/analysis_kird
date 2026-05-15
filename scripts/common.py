from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
FIGURES_DIR = PROCESSED_DIR / "figures"

DEFAULT_RAW_DATA_PATH = Path(
    r"C:\Users\rhksd\Desktop\Data\활동조사_재직자_데이터\활동조사_재직자_데이터_어승수.xlsx"
)
LOCAL_RAW_DATA_PATH = DATA_DIR / "raw_data.xlsx"

PSYCH_SAFETY_ITEMS = ["Q12_1", "Q12_2", "Q12_3", "Q12_4"]
KNOWLEDGE_SHARING_ITEMS = ["Q19_7", "Q19_8", "Q19_9", "Q19_10"]
INNOVATIVE_BEHAVIOR_ITEMS = ["Q28_4", "Q28_5", "Q28_6", "Q28_7"]

SCALES = {
    "psych_safety": PSYCH_SAFETY_ITEMS,
    "knowledge_sharing": KNOWLEDGE_SHARING_ITEMS,
    "innovative_behavior": INNOVATIVE_BEHAVIOR_ITEMS,
}

CONTROL_VARS = {
    "SQ1": "gender",
    "SQ2_1": "age_group",
    "SQ3": "org_type",
    "QT1": "job_type",
}

VAR_LABELS = {
    "psych_safety": "심리적 안전감",
    "knowledge_sharing": "지식공유",
    "innovative_behavior": "혁신행동",
    "Q12_1": "구성원 실수 책임 공유",
    "Q12_2": "구성원 다양성 존중",
    "Q12_3": "구성원 간 도움 요청의 자유로움",
    "Q12_4": "나의 노력에 대한 존중",
    "Q19_7": "동료들과 새로운 정보 공유 정도",
    "Q19_8": "동료들과 업무에 대한 공유 정도",
    "Q19_9": "특정지식 필요 시, 동료를 통한 습득",
    "Q19_10": "특정 배움 필요 시, 동료에게 가르침 요청 정도",
    "Q28_4": "업무 문제점 해결을 위한 신규 아이디어 개발",
    "Q28_5": "업무 수행에 활용되는 신규 방법 등을 찾으려는 노력",
    "Q28_6": "혁신적 아이디어 공감",
    "Q28_7": "혁신적 아이디어 적용을 위한 노력",
    "gender": "성별",
    "age_group": "연령대",
    "org_type": "소속기관 유형",
    "job_type": "직무구분",
}


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def resolve_raw_data_path() -> Path:
    env_path = os.environ.get("KIRD_RAW_DATA")
    candidates = [
        Path(env_path) if env_path else None,
        LOCAL_RAW_DATA_PATH,
        DEFAULT_RAW_DATA_PATH,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    checked = "\n".join(str(path) for path in candidates if path)
    raise FileNotFoundError(
        "원자료 엑셀 파일을 찾지 못했습니다. 확인한 경로:\n"
        f"{checked}\n"
        "해결 방법: data/raw_data.xlsx로 파일을 복사하거나 KIRD_RAW_DATA 환경변수를 설정하세요."
    )


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def judgment_alpha(alpha: float) -> str:
    if alpha >= 0.9:
        return "매우 높음"
    if alpha >= 0.8:
        return "양호"
    if alpha >= 0.7:
        return "수용 가능"
    return "추가 검토 필요"
