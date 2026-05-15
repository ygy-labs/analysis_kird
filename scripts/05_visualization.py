from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from common import FIGURES_DIR, PROCESSED_DIR, VAR_LABELS, ensure_directories


WIDTH = 1200
HEIGHT = 760
BG = "#FFFFFF"
TEXT = "#1F2933"
GRID = "#E5E7EB"
TEAL = "#2F6F73"
PURPLE = "#725AC1"
GREEN = "#5AA469"
ORANGE = "#D9822B"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_title(draw: ImageDraw.ImageDraw, title: str) -> None:
    draw.text((40, 28), title, fill=TEXT, font=font(28, bold=True))


def save_histograms(df: pd.DataFrame) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw_title(draw, "주요 변수 분포")
    variables = ["psych_safety", "knowledge_sharing", "innovative_behavior"]
    colors = [TEAL, PURPLE, GREEN]
    panel_w = 340
    panel_h = 460
    top = 150
    lefts = [60, 430, 800]
    bins = [1 + i * 0.25 for i in range(17)]

    for var, color, left in zip(variables, colors, lefts):
        values = df[var].dropna().tolist()
        counts = [0] * (len(bins) - 1)
        for value in values:
            idx = min(max(int((value - 1) / 0.25), 0), len(counts) - 1)
            counts[idx] += 1
        max_count = max(counts)
        draw.text((left, top - 42), VAR_LABELS[var], fill=TEXT, font=font(20, bold=True))
        draw.line((left, top + panel_h, left + panel_w, top + panel_h), fill=TEXT, width=2)
        draw.line((left, top, left, top + panel_h), fill=TEXT, width=2)
        bar_w = panel_w / len(counts)
        for i, count in enumerate(counts):
            h = 0 if max_count == 0 else count / max_count * (panel_h - 25)
            x0 = left + i * bar_w + 2
            y0 = top + panel_h - h
            x1 = left + (i + 1) * bar_w - 2
            draw.rectangle((x0, y0, x1, top + panel_h), fill=color)
        for label, x in [("1", left), ("3", left + panel_w / 2), ("5", left + panel_w)]:
            draw.text((x - 6, top + panel_h + 12), label, fill=TEXT, font=font(14))
        draw.text((left + 118, top + panel_h + 42), "점수", fill=TEXT, font=font(15))

    img.save(FIGURES_DIR / "01_main_variable_distributions.png")


def color_scale(value: float) -> tuple[int, int, int]:
    # Blue-white-red scale centered at zero.
    value = max(min(value, 1), -1)
    if value >= 0:
        r = 255
        g = int(255 - value * 115)
        b = int(255 - value * 115)
    else:
        r = int(255 + value * 115)
        g = int(255 + value * 80)
        b = 255
    return r, g, b


def save_correlation_heatmap() -> None:
    corr = pd.read_csv(PROCESSED_DIR / "correlation_scales_matrix.csv", index_col=0, encoding="utf-8-sig")
    labels = [VAR_LABELS.get(col, col) for col in corr.columns]
    img = Image.new("RGB", (800, 720), BG)
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), "주요 변수 상관관계 히트맵", fill=TEXT, font=font(26, bold=True))
    cell = 145
    left = 245
    top = 175
    for i, label in enumerate(labels):
        draw.text((left + i * cell + 10, top - 55), label, fill=TEXT, font=font(16, bold=True))
        draw.text((40, top + i * cell + 55), label, fill=TEXT, font=font(16, bold=True))
    for r, row in enumerate(corr.index):
        for c, col in enumerate(corr.columns):
            value = float(corr.loc[row, col])
            x0 = left + c * cell
            y0 = top + r * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color_scale(value), outline=BG)
            draw.text((x0 + 48, y0 + 58), f"{value:.3f}", fill=TEXT, font=font(18, bold=True))
    img.save(FIGURES_DIR / "02_correlation_heatmap.png")


def save_partial_correlations() -> None:
    partial = pd.read_csv(
        PROCESSED_DIR / "partial_correlation_psych_safety_items_outcomes_controls.csv",
        encoding="utf-8-sig",
    )
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw_title(draw, "통제변수 포함 심리적 안전감 하위 문항 부분상관")
    left = 120
    top = 135
    plot_w = 940
    plot_h = 470
    max_val = 0.35
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=TEXT, width=2)
    draw.line((left, top, left, top + plot_h), fill=TEXT, width=2)
    for tick in [0, 0.1, 0.2, 0.3]:
        y = top + plot_h - tick / max_val * plot_h
        draw.line((left - 6, y, left + plot_w, y), fill=GRID, width=1)
        draw.text((left - 55, y - 10), f"{tick:.1f}", fill=TEXT, font=font(14))
    groups = ["Q12_1", "Q12_2", "Q12_3", "Q12_4"]
    outcomes = ["knowledge_sharing", "innovative_behavior"]
    bar_w = 48
    group_gap = plot_w / len(groups)
    for i, item in enumerate(groups):
        center = left + group_gap * i + group_gap / 2
        for j, outcome in enumerate(outcomes):
            value = float(partial[(partial["predictor"] == item) & (partial["outcome"] == outcome)]["partial_r"].iloc[0])
            x0 = center - bar_w - 5 + j * (bar_w + 10)
            y0 = top + plot_h - value / max_val * plot_h
            draw.rectangle((x0, y0, x0 + bar_w, top + plot_h), fill=TEAL if outcome == "knowledge_sharing" else ORANGE)
            draw.text((x0 - 2, y0 - 25), f"{value:.3f}", fill=TEXT, font=font(13))
        draw.text((center - 45, top + plot_h + 18), item, fill=TEXT, font=font(15, bold=True))
    draw.rectangle((820, 80, 840, 100), fill=TEAL)
    draw.text((850, 78), "지식공유", fill=TEXT, font=font(16))
    draw.rectangle((940, 80, 960, 100), fill=ORANGE)
    draw.text((970, 78), "혁신행동", fill=TEXT, font=font(16))
    img.save(FIGURES_DIR / "03_partial_correlations_by_item.png")


def save_regression_betas() -> None:
    coef = pd.read_csv(PROCESSED_DIR / "regression_coefficients.csv", encoding="utf-8-sig")
    focus = coef[coef["term"] == "psych_safety"].copy()
    img = Image.new("RGB", (850, 620), BG)
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), "심리적 안전감의 표준화 회귀계수", fill=TEXT, font=font(26, bold=True))
    left = 150
    top = 135
    plot_w = 560
    plot_h = 360
    max_val = max(0.35, math.ceil(float(focus["std_beta"].max()) * 10) / 10)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=TEXT, width=2)
    draw.line((left, top, left, top + plot_h), fill=TEXT, width=2)
    for tick in [0, 0.1, 0.2, 0.3]:
        y = top + plot_h - tick / max_val * plot_h
        draw.line((left - 6, y, left + plot_w, y), fill=GRID, width=1)
        draw.text((left - 55, y - 10), f"{tick:.1f}", fill=TEXT, font=font(14))
    for i, (_, row) in enumerate(focus.iterrows()):
        value = float(row["std_beta"])
        x0 = left + 110 + i * 220
        y0 = top + plot_h - value / max_val * plot_h
        draw.rectangle((x0, y0, x0 + 90, top + plot_h), fill=PURPLE)
        draw.text((x0 + 12, y0 - 28), f"{value:.3f}", fill=TEXT, font=font(16, bold=True))
        draw.text((x0 - 25, top + plot_h + 18), VAR_LABELS.get(row["outcome"], row["outcome"]), fill=TEXT, font=font(16, bold=True))
    img.save(FIGURES_DIR / "04_regression_standardized_beta.png")


def main() -> None:
    try:
        ensure_directories()
        input_path = PROCESSED_DIR / "preprocessed_data.csv"
        if not input_path.exists():
            raise FileNotFoundError("processed/preprocessed_data.csv가 없습니다. 01 스크립트를 먼저 실행하세요.")

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        save_histograms(df)
        save_correlation_heatmap()
        save_partial_correlations()
        save_regression_betas()
        print("[완료] 05_visualization.py: 주요 시각화 이미지를 저장했습니다.")
    except Exception as exc:
        print(f"[오류] 05_visualization.py 실행 실패: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
