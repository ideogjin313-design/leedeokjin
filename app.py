import ast
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="QA/QC 배치 모니터링", layout="wide")
pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], .main { background-color: #ffffff !important; color: #0f172a; }
    [data-testid="stHeader"] { background: #ffffff !important; }
    [data-testid="stToolbar"] { right: 0.5rem; }
    h1, h2, h3, h4, h5, h6, p, label, span { color: #0f172a; }
    .stCaption { color: #475569 !important; }
    [data-testid="stMarkdownContainer"] p { color: #0f172a; }
    [data-baseweb="select"] > div { background: #ffffff !important; color: #0f172a !important; border-color: #cbd5e1 !important; }
    .stNumberInput input, .stTextInput input { background: #ffffff !important; color: #0f172a !important; }
    [data-testid="stPlotlyChart"] > div { background: #ffffff !important; border-radius: 8px; }
    .js-plotly-plot .plot-container { background: #ffffff !important; }
    .js-plotly-plot .svg-container { background: #ffffff !important; }
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
    .stButton > button {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stButton > button:hover { background: #f8fafc !important; }
    [data-baseweb="slider"] > div { background: #e2e8f0 !important; }
    .stSlider [data-baseweb="slider"] div[style*="rgb(26, 108, 255)"],
    .stSlider [data-baseweb="slider"] div[style*="rgb(26,108,255)"] {
        background-color: #8b2f2f !important;
        border-color: #8b2f2f !important;
    }
    [data-baseweb="slider"] [role="slider"] { background: #8b2f2f !important; border-color: #8b2f2f !important; }
    html, body, .stApp, [data-testid="stAppViewContainer"] { color-scheme: only light !important; }
    div[data-testid="stCheckbox"] { background: transparent !important; }
    div[data-testid="stCheckbox"] * { color: #0f172a !important; }
    div[data-testid="stCheckbox"] input[type="checkbox"] {
        accent-color: #dc2626 !important;
        background: #ffffff !important;
        background-color: #ffffff !important;
        color-scheme: only light !important;
        forced-color-adjust: none !important;
        filter: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stCheckbox"] input[type="checkbox"]::before,
    div[data-testid="stCheckbox"] input[type="checkbox"]::after {
        forced-color-adjust: none !important;
    }
    [data-testid="stWidgetLabel"] { color: #0f172a !important; }
    [data-testid="stToggle"] label { color: #0f172a !important; }
    [data-testid="stToggle"] div[role="switch"] {
        background: #e2e8f0 !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stExpander { background: #ffffff !important; border-radius: 10px; }
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] details {
        background: #ffffff !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        background: #ffffff !important;
        color: #0f172a !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: #f8fafc !important;
    }
    [data-testid="stExpander"] summary svg {
        color: #0f172a !important;
        fill: #0f172a !important;
    }
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    [data-testid="stDataFrame"] [role="grid"] { background: #ffffff !important; }
    [data-testid="stDataFrame"] [role="row"] { background: transparent !important; }
    [data-testid="stDataFrame"] [role="columnheader"] { background: #f1f5f9 !important; color:#0f172a !important; }
    [data-testid="stDataFrame"] [role="gridcell"] { background: transparent !important; }
    [data-testid="stTable"] { background: #ffffff !important; }
    [data-testid="stDataFrame"] * { color: #0f172a !important; }
    [data-testid="stDataFrame"] thead tr th { background-color: #f1f5f9 !important; color:#0f172a !important; font-weight: 700 !important; }
    [data-testid="stDataFrame"] tbody tr:nth-child(even) td { background-color: #f8fafc !important; }
    [data-testid="stDataFrame"] tbody tr:nth-child(odd) td { background-color: #ffffff !important; }
    [data-testid="stDataFrame"] [role="gridcell"] { color:#0f172a !important; }
    .stAlert { border-radius: 10px; }
    .block-container { padding-top: 2.4rem; padding-bottom: 0.4rem; }
    .panel {
        background: #ffffff;
        border: 1px solid #e7e9ef;
        border-radius: 14px;
        padding: 8px 10px;
        margin-bottom: 6px;
        min-height: 84px;
    }
    .kpi-title { font-size: 0.94rem; color: #70757f; margin-bottom: 6px; line-height: 1.25; text-align: center; }
    .kpi-value { font-size: 1.28rem; font-weight: 800; color: #1f2937; white-space: normal; line-height: 1.3; text-align: center; }
    .section-title { font-size: 1.02rem; font-weight: 700; margin-bottom: 8px; }
    .panel-title { font-size: 1.08rem; font-weight: 800; color: #0f172a; margin-bottom: 3px; letter-spacing: 0.2px; }
    .panel-subtitle { font-size: 0.82rem; color: #5b6577; }
    .plain-title {
        font-size: 1.34rem;
        font-weight: 800;
        color: #0f172a;
        margin: 8px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #dbe1ea;
    }
    .log-card { background: #ffffff; border: 1px solid #e7e9ef; border-radius: 12px; padding: 6px 10px; }
    .log-item { margin-bottom: 4px; color: #1f2937; font-size: 0.92rem; line-height: 1.25; font-weight: 600; }
    .log-item b { display: inline-block; min-width: 64px; color: #334155; font-size: 0.94rem; font-weight: 800; }
    .hero {
        background: #ffffff;
        color: #0f172a;
        border-radius: 16px;
        padding: 12px 16px;
        border: 1px solid #dbe1ea;
        margin-bottom: 6px;
        margin-top: 10px;
    }
    .hero-title { font-size: 2.55rem; line-height: 1.15; font-weight: 950; letter-spacing: 0.2px; margin-bottom: 4px; padding-top: 0; color:#0f172a; }
    .hero-sub { font-size: 0.90rem; color: #475569; }
    .priority-card { border-radius: 14px; padding: 12px 14px; border: 1px solid #d1d5db; }
    .priority-red { background: #fff1f2; border-color: #fecdd3; }
    .priority-yellow { background: #fffbeb; border-color: #fde68a; }
    .priority-green { background: #ecfdf5; border-color: #86efac; }
    .priority-title { font-size: 0.98rem; font-weight: 800; margin-bottom: 6px; color: #111827; }
    .priority-text { font-size: 0.9rem; color: #111827; line-height: 1.45; font-weight: 600; }
    .status-card {
        background: #ffffff;
        border: 1px solid #e7e9ef;
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 8px;
        box-shadow: none;
        min-height: 190px;
    }
    .status-card-title {
        font-size: 1.44rem;
        font-weight: 900;
        color: #111111;
        padding-bottom: 10px;
        border-bottom: 1px solid #dbe1ea;
        margin-bottom: 12px;
    }
    .status-line {
        font-size: 1.16rem;
        font-weight: 800;
        line-height: 2.0;
        color: #1d1311;
    }
    .status-summary-box {
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 10px 12px 12px 12px;
        margin: 4px 0 8px 0;
        background: #ffffff;
    }
    .status-summary-title {
        font-size: 1.62rem;
        font-weight: 900;
        color: #0f172a;
        margin: 0 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #dbe1ea;
    }
    .checklist-title {
        font-size: 1.52rem;
        font-weight: 900;
        color: #0f172a;
        margin: 8px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #dbe1ea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("final_merged(2).csv")
    df = df.replace(["NA", "N/A", "", "nan"], np.nan)
    # 문자열로 섞여 들어오는 수치 컬럼을 숫자로 정규화
    for col in ["ID", "Viscosity", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Stability_Test" in df.columns:
        df["Stability_Passed"] = (
            df["Stability_Test"].astype(str).str.upper().map(
                {
                    "TRUE": True,
                    "1": True,
                    "1.0": True,
                    "FALSE": False,
                    "0": False,
                    "0.0": False,
                    "NAN": False,
                }
            )
        ).fillna(False)
    else:
        df["Stability_Passed"] = False
    return df


def get_batch_curve(batch_row: pd.Series):
    raw = batch_row.get("Rheology_Data")
    if pd.isna(raw):
        return None, None
    try:
        parsed = ast.literal_eval(raw)
        shear_rate = pd.to_numeric(pd.Series(parsed[0].get("shear_rate", [])), errors="coerce").to_numpy(dtype=float)
        avg_visc = pd.to_numeric(pd.Series(parsed[1].get("avg_viscosity", [])), errors="coerce").to_numpy(dtype=float)
        if len(shear_rate) > 0 and len(shear_rate) == len(avg_visc):
            mask = np.isfinite(shear_rate) & np.isfinite(avg_visc) & (shear_rate > 0) & (avg_visc > 0)
            shear_rate = shear_rate[mask]
            avg_visc = avg_visc[mask]
            if len(shear_rate) > 2:
                order = np.argsort(shear_rate)
                return shear_rate[order], avg_visc[order]
    except Exception:
        return None, None
    return None, None


def parse_rheology_mean(raw_value):
    if pd.isna(raw_value):
        return None
    try:
        parsed = ast.literal_eval(raw_value)
        avg_visc = parsed[1].get("avg_viscosity", [])
        if avg_visc:
            return float(np.mean(avg_visc))
    except Exception:
        return None
    return None


def ingredient_name_ko(name: str) -> str:
    n = str(name).lower()
    exact = {
        "arlyponf": "아릴폰 F",
        "arlypontt": "아릴폰 TT",
        "luviquatexcellence": "루비쿼트 엑설런스",
        "salcaresuper7": "살케어 슈퍼7",
        "dehyquartcc6": "디하이쿼트 CC6",
        "dehyquartaca": "디하이쿼트 ACA",
        "dehyquartcc7benz": "디하이쿼트 CC7 벤즈",
        "dehytonab30": "디하이톤 AB30",
        "dehytonpk45": "디하이톤 PK45",
        "dehytonmc": "디하이톤 MC",
        "dehytonml": "디하이톤 ML",
        "plantacare2000": "플랜타케어 2000",
        "plantacare818": "플랜타케어 818",
        "plantaponlc7": "플랜타폰 LC7",
        "plantaponacg50": "플랜타폰 ACG50",
        "plantaponaminoscgl": "플랜타폰 아미노 SCGL",
        "plantaponaminokgl": "플랜타폰 아미노 KGL",
        "texaponsb3kc": "텍사폰 SB3 KC",
    }
    if n in exact:
        return exact[n]
    return str(name)


def ingredient_category(name: str) -> str:
    n = str(name).lower()
    exact = {
        "arlyponf": "가용화 보조",
        "arlypontt": "가용화 보조",
        "luviquatexcellence": "컨디셔닝 폴리머",
        "salcaresuper7": "컨디셔닝 폴리머",
        "dehyquartcc6": "컨디셔닝 폴리머",
        "dehyquartaca": "컨디셔닝 폴리머",
        "dehyquartcc7benz": "컨디셔닝 폴리머",
        "dehytonab30": "양쪽성 계면활성제",
        "dehytonpk45": "양쪽성 계면활성제",
        "dehytonmc": "양쪽성 계면활성제",
        "dehytonml": "양쪽성 계면활성제",
        "plantacare2000": "비이온성 계면활성제",
        "plantacare818": "비이온성 계면활성제",
        "plantaponlc7": "음이온성 계면활성제",
        "plantaponacg50": "음이온성 계면활성제",
        "plantaponaminoscgl": "아미노산계 계면활성제",
        "plantaponaminokgl": "아미노산계 계면활성제",
        "texaponsb3kc": "음이온성 계면활성제",
    }
    if n in exact:
        return exact[n]
    if any(k in n for k in ["sles", "sls", "sulf", "betaine", "ampho", "texapon"]):
        return "계면활성제"
    if any(k in n for k in ["chloride", "salt", "na cl"]):
        return "염/점도조절"
    if any(k in n for k in ["citric", "acid", "ph"]):
        return "pH조절"
    if any(k in n for k in ["quat", "polyquaternium", "guar", "condition"]):
        return "컨디셔닝"
    if any(k in n for k in ["preserv", "benzo", "sorb", "phenoxy"]):
        return "보존"
    if any(k in n for k in ["fragrance", "perfume", "menthol", "cool"]):
        return "향/기능"
    return "원료사전 확인 필요"


def ingredient_label(name: str) -> str:
    return f"{str(name)}({ingredient_category(name)})"


def normalize_ing(name: str) -> str:
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")


REDLINE_RULES = {
    # name_key: (redline_max_success, risk_level, note)
    "texaponsb3kc": (16.24, "안전", "고농도에서도 안정성이 우수한 편"),
    "plantaponacg50": (13.68, "안전", "약 13.7%까지 안정적"),
    "plantaponlc7": (13.58, "위험", "레드라인 초과 시 제형 파괴 위험"),
    "plantacare818": (14.45, "안전", "14.5% 부근까지 안정"),
    "plantacare2000": (13.57, "위험", "레드라인 초과 시 실패 사례 증가"),
    "dehytonmc": (13.30, "주의", "13.3% 부근 경계"),
    "dehytonpk45": (13.21, "주의", "13.2% 부근 경계"),
    "dehytonml": (14.27, "주의", "14.2% 부근 경계"),
    "dehytonab30": (13.09, "주의", "13.1% 부근 경계"),
    "plantaponaminoscgl": (13.44, "안전", "13.4% 부근까지 안정"),
    "plantaponaminokgl": (12.76, "위험", "아미노산계 중 취급 주의"),
    "dehyquartaca": (13.30, "주의", "13.3% 부근 경계"),
    "luviquatexcellence": (3.02, "위험", "3.0% 부근 초과 시 분리 가능성"),
    "dehyquartcc6": (4.00, "매우 위험", "4.0% 초과 시 안정성 급감"),
    "dehyquartcc7benz": (3.40, "매우 위험", "3.4% 초과 시 실패 사례"),
    "salcaresuper7": (3.18, "주의", "3.18% 부근 경계"),
    "arlyponf": (5.27, "안전", "점도 형성에 긍정적 기여"),
    "arlypontt": (4.82, "최고 위험", "5% 접근 시 실패 급증"),
}


def grade_from_score(score: float) -> str:
    if score < 1.0:
        return "묽음"
    if score <= 3.0:
        return "안정"
    return "고점도"


def build_viscosity_by_id(df: pd.DataFrame) -> tuple[pd.Series, str]:
    vis = df.groupby("ID")["Viscosity"].mean().dropna()
    if len(vis) >= 4:
        return vis, "Viscosity"

    # Viscosity 결측이 많은 경우 Rheology_Data에서 평균 점도를 대체 사용
    rows = []
    for bid, group in df.groupby("ID"):
        candidate = None
        for raw in group["Rheology_Data"].dropna():
            candidate = parse_rheology_mean(raw)
            if candidate is not None:
                break
        if candidate is not None:
            rows.append((bid, candidate))

    if rows:
        alt = pd.Series({k: v for k, v in rows})
        return alt.dropna(), "Rheology_Data(avg_viscosity)"
    return vis, "없음"


def build_curve_mean_by_id(df: pd.DataFrame) -> pd.Series:
    rows = []
    for bid, group in df.groupby("ID"):
        mean_val = None
        for raw in group["Rheology_Data"].dropna():
            mean_val = parse_rheology_mean(raw)
            if mean_val is not None:
                break
        if mean_val is not None:
            rows.append((bid, mean_val))
    return pd.Series({k: v for k, v in rows}, dtype=float) if rows else pd.Series(dtype=float)


def get_target_reference_curve(df: pd.DataFrame, batch_df: pd.DataFrame, target_type: str) -> tuple[pd.Series, str]:
    curve_mean_by_id = build_curve_mean_by_id(df)
    stable_meta = batch_df[batch_df["Stability_Passed"] == True]
    ref_ids = stable_meta[stable_meta["Rheology_Type"] == target_type]["ID"].tolist()
    reference = curve_mean_by_id[curve_mean_by_id.index.isin(ref_ids)]
    source = f"안정성 합격 + 목표 타입({target_type})"
    if len(reference) < 10:
        reference = curve_mean_by_id[curve_mean_by_id.index.isin(stable_meta["ID"])]
        source = "안정성 합격 전체"
    return reference.dropna(), source


def compute_grade_score(value: float, ref: pd.Series) -> float:
    q10, q25, q75, q90 = ref.quantile([0.10, 0.25, 0.75, 0.90])
    if q90 <= q10:
        return 2.0
    if value <= q10:
        return 0.0
    if value < q25:
        return float(np.interp(value, [q10, q25], [0.0, 1.0]))
    if value <= q75:
        return float(np.interp(value, [q25, q75], [1.0, 3.0]))
    if value < q90:
        return float(np.interp(value, [q75, q90], [3.0, 4.0]))
    return 4.0


def estimate_curve_mean_knn(after_vec: pd.Series, pivot_all: pd.DataFrame, curve_mean_by_id: pd.Series, k: int = 8) -> float | None:
    valid_ids = [idx for idx in pivot_all.index if idx in curve_mean_by_id.index]
    if len(valid_ids) < 3:
        return None
    mat = pivot_all.loc[valid_ids].to_numpy(dtype=float)
    target = after_vec.reindex(pivot_all.columns).fillna(0).to_numpy(dtype=float)
    d = np.linalg.norm(mat - target, axis=1)
    order = np.argsort(d)[: max(1, min(k, len(d)))]
    d_sel = d[order]
    y_sel = curve_mean_by_id.reindex(valid_ids).iloc[order].to_numpy(dtype=float)
    w = 1.0 / (d_sel + 1e-6)
    return float(np.sum(w * y_sel) / np.sum(w))


def find_auto_simulation_plan(
    df: pd.DataFrame,
    batch_df: pd.DataFrame,
    selected_batch_id,
    sim_base: pd.DataFrame,
    target_type: str,
    sim_delta_min: float,
    sim_delta_max: float,
    guide_map: dict[str, float] | None = None,
    n_random: int = 2500,
):
    if sim_base.empty:
        return None, "시뮬레이션 대상 성분이 없습니다."

    model, label_encoder, feature_names, model_msg = load_tabpfn_artifacts()
    if model_msg:
        return None, f"모델 로드 실패로 자동 탐색 불가: {model_msg}"

    pivot_all = df.pivot_table(index="ID", columns="name", values="amount", aggfunc="sum").fillna(0)
    if selected_batch_id not in pivot_all.index:
        return None, "선택 ID의 벡터를 만들 수 없습니다."

    stable_ids = batch_df[batch_df["Stability_Passed"] == True]["ID"].tolist()
    stable_pivot = pivot_all[pivot_all.index.isin(stable_ids)]
    curve_mean = build_curve_mean_by_id(df)
    ref_curve, _ = get_target_reference_curve(df, batch_df, target_type)

    ings = sim_base["name"].astype(str).tolist()
    base_amt = sim_base["amount"].astype(float).to_numpy()

    guide_delta = np.zeros(len(ings), dtype=float)
    guide_map = guide_map or {}
    for i, ing in enumerate(ings):
        cur = float(base_amt[i])
        target = float(guide_map.get(ing, cur))
        if cur > 0:
            d = ((target / cur) - 1.0) * 100.0
        else:
            d = 0.0
        guide_delta[i] = float(np.clip(d, sim_delta_min, sim_delta_max))

    rng = np.random.default_rng(42)
    candidates = [np.zeros(len(ings), dtype=float), guide_delta]
    for _ in range(int(n_random)):
        candidates.append(rng.uniform(sim_delta_min, sim_delta_max, size=len(ings)))

    valid_deltas = []
    valid_amounts = []
    for d in candidates:
        # run_sim과 동일: 동시에 MULTI_CHANGE_N개 초과 성분을 MULTI_CHANGE_PCT 이상 변경 금지
        if int(np.sum(np.abs(d) >= float(MULTI_CHANGE_PCT))) > int(MULTI_CHANGE_N):
            continue
        amt = base_amt * (1.0 + d / 100.0)
        violated = False
        for i, ing in enumerate(ings):
            v = float(amt[i])
            if ing in stable_pivot.columns and len(stable_pivot) >= 15:
                mn = float(stable_pivot[ing].min())
                mx = float(stable_pivot[ing].max())
                if v < mn or v > mx:
                    violated = True
                    break
            key = normalize_ing(ing)
            if key in REDLINE_RULES:
                redline, _, _ = REDLINE_RULES[key]
                if v > float(redline):
                    violated = True
                    break
        if not violated:
            valid_deltas.append(np.array(d, dtype=float))
            valid_amounts.append(np.array(amt, dtype=float))

    if not valid_deltas:
        return None, "제약 조건을 만족하는 조합을 찾지 못했습니다."

    x_rows = []
    base_map = (
        df[df["ID"] == selected_batch_id][["name", "amount"]]
        .dropna()
        .groupby("name", as_index=False)["amount"]
        .sum()
        .set_index("name")["amount"]
        .to_dict()
    )
    for amt in valid_amounts:
        x = [base_map.get(col, 0.0) for col in feature_names]
        col_idx = {c: i for i, c in enumerate(feature_names)}
        for ing, v in zip(ings, amt):
            if ing in col_idx:
                x[col_idx[ing]] = float(v)
        x_rows.append(x)
    pred = model.predict(pd.DataFrame(x_rows, columns=feature_names))
    pred_labels = [str(p) if isinstance(p, (str, np.str_)) else label_encoder.inverse_transform([int(p)])[0] for p in pred]

    before_vec = pivot_all.loc[selected_batch_id].copy()
    best_idx = None
    best_key = None
    best_meta = None
    for i, d in enumerate(valid_deltas):
        after_type = str(pred_labels[i]).upper()
        after_vec = before_vec.copy()
        for ing, v in zip(ings, valid_amounts[i]):
            if ing in after_vec.index:
                after_vec[ing] = float(v)
        sim_mean = estimate_curve_mean_knn(after_vec, pivot_all, curve_mean)
        sim_score = float(compute_grade_score(float(sim_mean), ref_curve)) if (sim_mean is not None and len(ref_curve) > 10) else None
        grade_ok = sim_score is not None and (1.0 <= sim_score <= 3.0)
        type_ok = after_type == target_type.upper()
        both_ok = type_ok and grade_ok
        closeness = -abs((sim_score if sim_score is not None else 9.0) - 2.0)
        move = -float(np.sum(np.abs(d)))
        key = (int(both_ok), int(type_ok), int(grade_ok), closeness, move)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i
            best_meta = {"type": after_type, "score": sim_score, "type_ok": type_ok, "grade_ok": grade_ok, "both_ok": both_ok}

    if best_idx is None:
        return None, "자동 탐색 결과를 계산하지 못했습니다."

    result = {ing: float(valid_deltas[best_idx][j]) for j, ing in enumerate(ings)}
    return {"deltas": result, "meta": best_meta}, None


def classify_selected_viscosity_grade(df: pd.DataFrame, batch_df: pd.DataFrame, selected_batch_id, target_type: str):
    curve_mean_by_id = build_curve_mean_by_id(df)
    if selected_batch_id not in curve_mean_by_id.index:
        return None, None, None, None, "선택 배치의 레올로지 곡선 점도 평균을 계산하지 못했습니다."

    selected_mean = float(curve_mean_by_id.loc[selected_batch_id])
    reference, source = get_target_reference_curve(df, batch_df, target_type)
    if len(reference) < 10:
        return None, selected_mean, None, source, "기준 배치 수가 부족하여 점도 등급 계산을 보류합니다."

    q25, q75 = reference.quantile([0.25, 0.75])
    if selected_mean < q25:
        grade = "묽음"
    elif selected_mean <= q75:
        grade = "안정"
    else:
        grade = "고점도"
    score = compute_grade_score(selected_mean, reference)
    return grade, selected_mean, score, source, None


def compute_curve_out_of_band_pct(
    df: pd.DataFrame, batch_df: pd.DataFrame, selected_batch_id, selected_x: np.ndarray | None, selected_y: np.ndarray | None, target_type: str
) -> float | None:
    if selected_x is None or selected_y is None:
        return None
    ref_ids = batch_df[
        (batch_df["Stability_Passed"] == True) & (batch_df["Rheology_Type"] == target_type)
    ]["ID"].tolist()
    ref_ids = [rid for rid in ref_ids if rid != selected_batch_id][:20]
    if not ref_ids:
        return None

    ref_curves = []
    for rid in ref_ids:
        row = df[df["ID"] == rid]
        if row.empty:
            continue
        rx, ry = get_batch_curve(row.iloc[0])
        if rx is None or len(rx) <= 3:
            continue
        safe_mask = (selected_x > 0) & np.isfinite(selected_x) & np.isfinite(selected_y) & (selected_y > 0)
        sx = selected_x[safe_mask]
        if len(sx) <= 3:
            continue
        interp = np.interp(np.log10(sx), np.log10(rx), np.log10(ry))
        ref_full = np.full_like(selected_x, np.nan, dtype=float)
        ref_full[safe_mask] = interp
        ref_curves.append(ref_full)

    if not ref_curves:
        return None

    ref_arr = np.vstack(ref_curves)
    med = np.nanmedian(ref_arr, axis=0)
    std = np.nanstd(ref_arr, axis=0)
    upper = np.power(10, med + std)
    lower = np.power(10, med - std)
    out_mask = (selected_y > upper) | (selected_y < lower)
    return float(np.mean(out_mask) * 100)


def build_blend_adjustment_guide(df: pd.DataFrame, batch_df: pd.DataFrame, selected_batch_id):
    pivot = df.pivot_table(index="ID", columns="name", values="amount", aggfunc="sum").fillna(0)
    if selected_batch_id not in pivot.index:
        return pd.DataFrame(), "선택 배치의 배합 데이터가 없습니다."

    selected_type = batch_df.loc[batch_df["ID"] == selected_batch_id, "Rheology_Type"].iloc[0]
    stable_meta = batch_df[batch_df["Stability_Passed"] == True]
    target_type = TARGET_RHEOLOGY_TYPE
    ref_ids = stable_meta[stable_meta["Rheology_Type"] == target_type]["ID"].tolist()
    if not ref_ids:
        ref_ids = stable_meta["ID"].tolist()
        target_type = "STABLE_POOL"

    ref_pool = pivot[pivot.index.isin(ref_ids)]
    if ref_pool.empty:
        return pd.DataFrame(), "안정권 기준 배치가 없어 배합 가이드를 만들 수 없습니다."

    selected_vec = pivot.loc[selected_batch_id]
    selected_ing_set = set(
        df[(df["ID"] == selected_batch_id) & (pd.to_numeric(df["amount"], errors="coerce").fillna(0) > 0)]["name"].dropna().astype(str).unique().tolist()
    )
    rows = []
    for ing in pivot.columns:
        if str(ing) not in selected_ing_set:
            continue
        ref_series = ref_pool[ing]
        nonzero = ref_series[ref_series > 0]
        base_series = nonzero if len(nonzero) >= 5 else ref_series
        q10, q25, q50, q75, q90 = base_series.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        min_v = float(base_series.min())
        max_v = float(base_series.max())
        key = normalize_ing(ing)
        redline = float(REDLINE_RULES[key][0]) if key in REDLINE_RULES else np.nan
        practical_low = min_v
        practical_high = max_v if not np.isfinite(redline) else min(max_v, redline)
        if practical_high < practical_low:
            practical_high = practical_low
        cur = float(selected_vec.get(ing, 0.0))
        if cur < q25:
            rec_action = "증가"
            target = float(q50)
        else:
            rec_action = "유지"
            target = cur

        target = float(np.clip(target, practical_low, practical_high))

        delta = float(target - cur)
        rec_action = "증가" if delta > 1e-9 else "유지"

        rows.append(
            {
                "성분": ingredient_label(ing),
                "현재 투입량": cur,
                "권장 목표값": target,
                "허용 하한(q10)": float(q10),
                "허용 상한(q90)": float(q90),
                "안정권 하한(q25)": float(q25),
                "안정권 중앙(q50)": float(q50),
                "안정권 상한(q75)": float(q75),
                "데이터 최소": min_v,
                "데이터 최대": max_v,
                "실무 하한": float(practical_low),
                "실무 상한": float(practical_high),
                "레드라인": "-" if not np.isfinite(redline) else round(float(redline), 2),
                "권장 변경량": delta,
                "권장 조치": rec_action,
                "원성분": ing,
            }
        )
    out = pd.DataFrame(rows)
    out["우선순위 점수"] = out["권장 변경량"].abs()
    out = out[out["권장 조치"] == "증가"].sort_values("우선순위 점수", ascending=False).head(6)
    out["실행 액션"] = out.apply(
        lambda r: (
            f"{r['원성분']} {r['권장 조치']} {abs(float(r['권장 변경량'])):.2f} -> {float(r['권장 목표값']):.2f}"
        ) if r["권장 조치"] == "증가" else "-",
        axis=1,
    )
    msg = f"현재 타입: {selected_type} / 목표 타입: {target_type}"
    return out, msg


def build_root_cause_table(df: pd.DataFrame, batch_df: pd.DataFrame, selected_batch_id, target_type: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="ID", columns="name", values="amount", aggfunc="sum").fillna(0)
    target = df.groupby("ID")["Rheology_Type"].first().reindex(pivot.index).fillna("UNKNOWN")

    if selected_batch_id not in pivot.index:
        return pd.DataFrame([{"성분": "-", "기여도(%)": np.nan, "현재": np.nan, "기준중앙(q50)": np.nan, "편차(절대량)": "-", "해석": "선택 배치 입력 벡터를 만들 수 없습니다."}])

    if len(target.unique()) < 2:
        return pd.DataFrame(
            [{"성분": "-", "기여도(%)": np.nan, "현재": np.nan, "기준중앙(q50)": np.nan, "편차(절대량)": "-", "해석": "학습 가능한 라벨 다양성이 부족합니다."}]
        )

    le = LabelEncoder()
    y = le.fit_transform(target)

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(pivot, y)

    fi = (
        pd.DataFrame({"성분": pivot.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )

    # 목표 타입 기준(안정성 합격 + 목표 타입) 분포를 "평소 범위"로 사용
    stable_ids = batch_df[(batch_df["Stability_Passed"] == True) & (batch_df["Rheology_Type"] == target_type)]["ID"].tolist()
    ref_pool = pivot[pivot.index.isin(stable_ids)]
    if ref_pool.empty:
        ref_pool = pivot[pivot.index.isin(batch_df[batch_df["Stability_Passed"] == True]["ID"].tolist())]

    selected_vec = pivot.loc[selected_batch_id]
    med = ref_pool.median() if not ref_pool.empty else pivot.median()
    std = (ref_pool.std().replace(0, np.nan) if not ref_pool.empty else pivot.std().replace(0, np.nan))
    z = ((selected_vec - med) / std).replace([np.inf, -np.inf], np.nan).fillna(0)
    fi["selected_amount"] = fi["성분"].map(selected_vec.to_dict()).fillna(0)
    fi["z_abs"] = fi["성분"].map(z.abs().to_dict()).fillna(0)
    fi["score"] = fi["importance"] * (1.0 + fi["z_abs"])
    fi = fi.sort_values("score", ascending=False).head(6)

    rows = []
    for _, r in fi.iterrows():
        ingredient = str(r["성분"])
        amount = float(r["selected_amount"])
        ref_series = ref_pool[ingredient] if (not ref_pool.empty and ingredient in ref_pool.columns) else pivot[ingredient]
        nz = ref_series[ref_series > 0]
        base = nz if len(nz) >= 5 else ref_series
        q25, q50, q75 = base.quantile([0.25, 0.50, 0.75])
        if amount < q25:
            level_text = "기준 하한 미만(부족)"
        elif amount > q75:
            level_text = "기준 상한 초과(과다)"
        else:
            level_text = "기준 범위 내"
        diff_abs = abs(float(amount - q50))

        comment = f"{level_text}"
        low_name = ingredient.lower()
        if "menthol" in low_name:
            comment += " · 멘톨 계열 농도 변화로 점도/쿨링 체감 변동 가능"
        elif "citric" in low_name:
            comment += " · 산도 조절 성분 편차로 안정성 변동 가능"
        elif "chloride" in low_name:
            comment += " · 염 농도 변화로 점도 저하 가능"

        rows.append(
            {
                "성분": ingredient_label(ingredient),
                "기여도(%)": round(float(r["score"] * 100), 1),
                "현재": round(amount, 2),
                "기준중앙(q50)": round(float(q50), 2),
                "기준범위(q25~q75)": f"{float(q25):.2f}~{float(q75):.2f}",
                "편차(절대량)": f"{diff_abs:.2f}",
                "해석": comment,
            }
        )
    return pd.DataFrame(rows)



@st.cache_resource
def load_tabpfn_artifacts():
    model_dir = Path("models")
    model_path = model_dir / "tabpfn_model.pkl"
    le_path = model_dir / "label_encoder.pkl"
    feature_path = model_dir / "feature_names.json"

    if not (model_path.exists() and le_path.exists() and feature_path.exists()):
        return None, None, None, "models 폴더에 모델 파일 3개(tabpfn_model.pkl, label_encoder.pkl, feature_names.json)를 넣어주세요."

    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(le_path)
        with feature_path.open("r", encoding="utf-8") as f:
            feature_names = json.load(f)
        return model, label_encoder, feature_names, None
    except Exception as e:
        return None, None, None, f"모델 로드 실패: {e}"


def make_tabpfn_input(df: pd.DataFrame, batch_id, feature_names: list[str]) -> pd.DataFrame:
    selected = df[df["ID"] == batch_id][["name", "amount"]].copy()
    selected["amount"] = pd.to_numeric(selected["amount"], errors="coerce").fillna(0.0)
    by_name = selected.groupby("name")["amount"].sum().to_dict()
    return pd.DataFrame([[by_name.get(col, 0.0) for col in feature_names]], columns=feature_names)


def plotly_with_click_value(fig: go.Figure, chart_key: str, x_label: str, y_label: str, show_idle_hint: bool = True):
    state_key = f"{chart_key}_selected_point"
    selected_point = st.session_state.get(state_key)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
    )
    fig.update_layout(clickmode="event+select")

    # 이전 선택점이 있으면: 전체를 반투명 처리하고 선택점만 강조
    if selected_point is not None:
        curve_idx = int(selected_point.get("curve", -1))
        point_idx = int(selected_point.get("point", -1))
        if 0 <= curve_idx < len(fig.data):
            for tr in fig.data:
                if hasattr(tr, "opacity"):
                    tr.opacity = 0.22
            try:
                base_tr = fig.data[curve_idx]
                x_vals = list(base_tr.x)
                y_vals = list(base_tr.y)
                if 0 <= point_idx < len(x_vals) and 0 <= point_idx < len(y_vals):
                    sx = x_vals[point_idx]
                    sy = y_vals[point_idx]
                    fig.add_trace(
                        go.Scatter(
                            x=[sx],
                            y=[sy],
                            mode="markers",
                            marker=dict(size=14, color="#22d3ee", line=dict(color="#ffffff", width=1.5)),
                            name="선택 포인트",
                            showlegend=False,
                            hovertemplate=f"{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>",
                        )
                    )
            except Exception:
                pass

    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key,
            on_select="rerun",
            selection_mode=["points"],
        )
        points = []
        if event is not None:
            if isinstance(event, dict):
                points = event.get("selection", {}).get("points", [])
            elif hasattr(event, "selection"):
                sel = event.selection
                if isinstance(sel, dict):
                    points = sel.get("points", [])
                elif hasattr(sel, "points"):
                    points = sel.points
        if points:
            p = points[-1]
            x_val = p.get("x")
            y_val = p.get("y")
            curve_idx = p.get("curve_number", p.get("curveNumber", 0))
            point_idx = p.get("point_index", p.get("pointNumber", 0))
            st.session_state[state_key] = {"curve": curve_idx, "point": point_idx}
            st.caption(f"선택 포인트: {x_label}={x_val:.4g}, {y_label}={y_val:.4g}")
        else:
            if show_idle_hint:
                st.caption("그래프 점을 클릭하면 해당 값을 표시합니다.")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
        if show_idle_hint:
            st.caption("그래프 점을 클릭하면 해당 값을 표시합니다.")


def render_rheology_type_distribution(selected_first: pd.Series, selected_batch_id):
    current_type = str(selected_first.get("Rheology_Type", "UNKNOWN"))
    predicted_type = "-"
    model, label_encoder, feature_names, model_msg = load_tabpfn_artifacts()
    if model_msg:
        st.info("모델 파일이 없어 판정 타입은 표시하지 않습니다.")
    else:
        try:
            x_input = make_tabpfn_input(df, selected_batch_id, feature_names)
            pred = model.predict(x_input)[0]
            if isinstance(pred, (str, np.str_)):
                predicted_type = str(pred)
            else:
                predicted_type = label_encoder.inverse_transform([int(pred)])[0]
        except Exception as e:
            st.warning(f"TabPFN 예측 실패: {e}")

    target_type = TARGET_RHEOLOGY_TYPE
    judge_type = predicted_type if predicted_type != "-" else current_type
    is_match = str(judge_type).upper() == str(target_type).upper()
    result_text = "목표 타입 일치" if is_match else "목표 타입 불일치"
    result_color = "#16a34a" if is_match else "#dc2626"

    type_cols = st.columns(3)
    type_values = [
        ("현재 타입", current_type),
        ("목표 타입", target_type),
        ("판단 결과", result_text),
    ]
    for idx, (title, value) in enumerate(type_values):
        with type_cols[idx]:
            if title == "판단 결과":
                st.markdown(
                    f"<div class='panel'><div class='kpi-title'>{title}</div><div class='kpi-value' style='color:{result_color};'>{value}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='panel'><div class='kpi-title'>{title}</div><div class='kpi-value'>{value}</div></div>",
                    unsafe_allow_html=True,
                )
    # 모델 판정 타입은 화면 단순화를 위해 표시하지 않음

df = load_data()

batch_df = df.groupby("ID", as_index=False).agg(
    Stability_Passed=("Stability_Passed", "first"),
    Rheology_Type=("Rheology_Type", "first"),
)
stable_ids = batch_df[batch_df["Stability_Passed"] == True]["ID"].dropna().tolist()
stable_df = df[df["ID"].isin(stable_ids)].copy()

all_ids = sorted(df["ID"].dropna().unique().tolist())
stable_options = sorted(stable_ids)
options = stable_options[::-1] if stable_options else (all_ids[::-1] if all_ids else [])
if options and "selected_batch_id" not in st.session_state:
    st.session_state["selected_batch_id"] = options[0]
if options and st.session_state.get("selected_batch_id") not in options:
    st.session_state["selected_batch_id"] = options[0]

# 고정 실무 기준(필요 시 코드에서만 수정)
TEMP_MIN, TEMP_MAX = 25.0, 30.0
HUM_MIN, HUM_MAX = 40.0, 60.0
STD_PROCESS_TEMP = 25.0
STD_PROCESS_HUM = 50.0
TARGET_RHEOLOGY_TYPE = "SHEAR-THIN"
Q_LOW, Q_HIGH = 0.10, 0.90
CITRIC_ABS_MAX = 0.10
SALT_REL_MAX = 0.15
SURF_REL_MAX = 0.05
MULTI_CHANGE_PCT = 15
MULTI_CHANGE_N = 1
SIM_DELTA_MIN = 0.0
SIM_DELTA_MAX = 30.0

header_title_col, header_time_col = st.columns([5.2, 1.2], gap="small")
with header_title_col:
    st.markdown(
        """
        <div class='hero'>
          <div class='hero-title'>Shampoo Quality Control Command Dashboard</div>
          <div class='hero-sub'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_time_col:
    now_top = datetime.now()
    st.markdown(
        f"""
        <div style='text-align:right; margin-top:8px; font-weight:700; color:#0f172a; line-height:1.35;'>
          <div style='font-size:1.02rem;'>{now_top.strftime('%Y-%m-%d')}</div>
          <div style='font-size:1.02rem;'>{now_top.strftime('%H:%M:%S')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

process_temp = STD_PROCESS_TEMP
process_hum = STD_PROCESS_HUM

current_id = st.session_state.get("selected_batch_id", options[0] if options else None)

# 배치 ID가 많아 구간(100 단위) -> ID 선택 순서로 분리
id_ints = sorted({int(x) for x in options}) if options else []
range_buckets = []
if id_ints:
    start = (min(id_ints) // 100) * 100
    end = (max(id_ints) // 100) * 100
    for s in range(start, end + 1, 100):
        e = s + 99
        bucket_ids = [i for i in id_ints if s <= i <= e]
        if bucket_ids:
            range_buckets.append((f"{s}~{e}", bucket_ids))

sel_range_col, sel_id_col, _ = st.columns([1.0, 1.3, 3.7])
if range_buckets:
    current_int = int(st.session_state.get("selected_batch_id", id_ints[0]))
    range_labels = [r[0] for r in range_buckets]
    saved_range = st.session_state.get("selected_id_range")
    default_range_idx = range_labels.index(saved_range) if saved_range in range_labels else 0
    with sel_range_col:
        selected_range_label = st.selectbox(
            "ID 구간",
            range_labels,
            index=default_range_idx,
            key="selected_id_range",
        )
    range_ids = dict(range_buckets)[selected_range_label]
    if current_int not in range_ids:
        current_int = range_ids[0]
    with sel_id_col:
        selected_batch_id = st.selectbox(
            "배치 ID",
            options=range_ids,
            format_func=lambda x: f"ID {x}",
            index=range_ids.index(current_int),
            key="selected_batch_id_in_range",
        )
    st.session_state["selected_batch_id"] = int(selected_batch_id)
else:
    selected_batch_id = st.selectbox("배치 ID", options, key="selected_batch_id")
selected_batch_rows = df[df["ID"] == selected_batch_id]
selected_first = selected_batch_rows.iloc[0] if not selected_batch_rows.empty else pd.Series(dtype=object)

selected_curve_x, selected_curve_y = get_batch_curve(selected_first)
selected_grade_global, selected_mean_global, selected_score_global, selected_source_global, selected_grade_msg_global = classify_selected_viscosity_grade(
    df, batch_df, selected_batch_id, TARGET_RHEOLOGY_TYPE
)
sim_state_key = f"sim_state_{selected_batch_id}"
sim_state = st.session_state.get(sim_state_key, {})
sim_eval_key = f"sim_eval_{selected_batch_id}"
sim_eval = st.session_state.get(sim_eval_key, {})
sim_estimated_score_global = sim_state.get("sim_score")
sim_pred_type_global = sim_state.get("pred_type")
sim_curve_scale_global = float(sim_state.get("curve_scale", 1.0)) if sim_state else 1.0
sim_active_global = bool(sim_state)
curve_out_pct_global = compute_curve_out_of_band_pct(
    df,
    batch_df,
    selected_batch_id,
    selected_curve_x,
    selected_curve_y,
    TARGET_RHEOLOGY_TYPE,
)

temp_ok = TEMP_MIN <= process_temp <= TEMP_MAX
hum_ok = HUM_MIN <= process_hum <= HUM_MAX
if not temp_ok or not hum_ok:
    st.error(
        "진행 불가: 공정 환경 기준을 벗어났습니다. "
        f"온도 {TEMP_MIN:.1f}~{TEMP_MAX:.1f}°C, 습도 {HUM_MIN:.0f}~{HUM_MAX:.0f}%RH를 만족해야 분석을 진행합니다."
    )
    st.write(f"- 현재 온도: {process_temp:.1f}°C ({'정상' if temp_ok else '이탈'})")
    st.write(f"- 현재 습도: {process_hum:.0f}%RH ({'정상' if hum_ok else '이탈'})")
    st.stop()

total_batches = int(batch_df["ID"].nunique())
stable_count = int(batch_df["Stability_Passed"].sum()) if "Stability_Passed" in batch_df else 0
usage_count = int(selected_batch_rows["name"].nunique()) if not selected_batch_rows.empty else 0
total_material_count = int(df["name"].dropna().nunique()) if "name" in df.columns else 0
current_type_card = str(selected_first.get("Rheology_Type", "UNKNOWN"))
display_type = current_type_card
if sim_active_global and sim_pred_type_global:
    display_type = f"{current_type_card} -> {sim_pred_type_global}"
type_match_card = (
    str(sim_pred_type_global).upper() == TARGET_RHEOLOGY_TYPE
    if sim_active_global and sim_pred_type_global
    else current_type_card.upper() == TARGET_RHEOLOGY_TYPE
)
result_text = "목표 타입 일치" if type_match_card else "목표 타입 불일치"
result_color = "#16a34a" if type_match_card else "#dc2626"
viscosity_grade_text = selected_grade_global if selected_grade_global is not None else "계산불가"
if sim_active_global and sim_estimated_score_global is not None and selected_grade_global is not None:
    viscosity_grade_text = f"{selected_grade_global} -> {grade_from_score(float(sim_estimated_score_global))}"
viscosity_grade_text = viscosity_grade_text.replace("묽음", "저점도")

kpi_row = [
    ("판단 결과", result_text),
    ("ID 번호", f"{selected_batch_id}"),
    ("현재 타입", display_type),
    ("목표 타입", TARGET_RHEOLOGY_TYPE),
    ("점도 등급", viscosity_grade_text),
    ("사용된 원료 수 / 전체 원료 수", f"{usage_count} / {total_material_count}"),
    ("상 안정성 합격 배치 수 / 전체 배치 수", f"{stable_count} / {total_batches}"),
    ("온도 / 습도", f"{process_temp:.1f}°C / {process_hum:.0f}%RH"),
]
row_cols = st.columns(8)
for idx, (col, (title, value)) in enumerate(zip(row_cols, kpi_row)):
    with col:
        if idx == 0:
            st.markdown(
                f"<div class='panel'><div class='kpi-title'>{title}</div><div class='kpi-value' style='color:{result_color};'>{value}</div></div>",
                unsafe_allow_html=True,
            )
        elif idx == 4:
            parts = [str(p).strip() for p in str(value).split("->")]
            colored_parts = []
            for p in parts:
                color = "#16a34a" if p == "안정" else "#dc2626"
                colored_parts.append(f"<span style='color:{color};'>{p}</span>")
            value_html = " <span style='color:#64748b;'>-></span> ".join(colored_parts)
            st.markdown(
                f"<div class='panel'><div class='kpi-title'>{title}</div><div class='kpi-value'>{value_html}</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='panel'><div class='kpi-title'>{title}</div><div class='kpi-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
is_stable_pass = bool(selected_first.get("Stability_Passed", False))
if type_match_card and viscosity_grade_text == "안정" and is_stable_pass:
    st.success("출하 허가 가능합니다: 목표 타입 일치 + 점도 등급 안정 + 상 안정성 합격")
    st.stop()

root_df = build_root_cause_table(df, batch_df, selected_batch_id, TARGET_RHEOLOGY_TYPE)
blend_df, _ = build_blend_adjustment_guide(df, batch_df, selected_batch_id)

with st.expander("목표 불일치 원인 파악", expanded=True):
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("<div class='plain-title'>[목표 타입 기준 레올로지 편차]</div>", unsafe_allow_html=True)
        fig_cmp = go.Figure()
        selected_x, selected_y = selected_curve_x, selected_curve_y
        ref_ids = batch_df[
            (batch_df["Stability_Passed"] == True) & (batch_df["Rheology_Type"] == TARGET_RHEOLOGY_TYPE)
        ]["ID"].tolist()
        ref_ids = [rid for rid in ref_ids if rid != selected_batch_id][:20]
        show_compare_lines = st.checkbox("참고용 비교선(ID) 함께 보기", value=False, key=f"show_cmp_ids_exp_{selected_batch_id}")
        ref_curves = []
        if selected_x is not None and len(ref_ids) > 0:
            for rid in ref_ids:
                row = df[df["ID"] == rid].iloc[0]
                rx, ry = get_batch_curve(row)
                if rx is not None and len(rx) > 3:
                    safe_mask = (selected_x > 0) & np.isfinite(selected_x) & np.isfinite(selected_y) & (selected_y > 0)
                    sx = selected_x[safe_mask]
                    if len(sx) > 3:
                        interp = np.interp(np.log10(sx), np.log10(rx), np.log10(ry))
                        ref_full = np.full_like(selected_x, np.nan, dtype=float)
                        ref_full[safe_mask] = interp
                        ref_curves.append(ref_full)
        if selected_x is not None and ref_curves:
            ref_arr = np.vstack(ref_curves)
            med = np.nanmedian(ref_arr, axis=0)
            std = np.nanstd(ref_arr, axis=0)
            upper = np.power(10, med + std)
            lower = np.power(10, med - std)
            fig_cmp.add_trace(go.Scatter(x=selected_x, y=upper, mode="lines", line=dict(width=0), showlegend=False))
            fig_cmp.add_trace(
                go.Scatter(
                    x=selected_x,
                    y=lower,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(122, 156, 140, 0.20)",
                    line=dict(width=0),
                    name="정상 범위",
                )
            )
        if selected_x is not None:
            fig_cmp.add_trace(
                go.Scatter(
                    x=selected_x,
                    y=selected_y,
                    mode="lines",
                    name=f"기준 Batch {selected_batch_id}",
                    line=dict(color="#2563eb", width=2.5),
                )
            )
        if show_compare_lines:
            compare_ids = sorted([rid for rid in ref_ids if rid != selected_batch_id], reverse=True)[:3]
            compare_palette = ["#b45309", "#7c3aed", "#0f766e", "#be123c", "#2563eb"]
            for idx, cid in enumerate(compare_ids):
                row = df[df["ID"] == cid].iloc[0]
                x, y = get_batch_curve(row)
                if x is not None:
                    fig_cmp.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            name=f"ID {cid}",
                            line=dict(width=1.4, color=compare_palette[idx % len(compare_palette)]),
                            opacity=0.75,
                        )
                    )
        if fig_cmp.data:
            fig_cmp.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_type="log",
                yaxis_type="log",
                xaxis_title="Log 전단속도",
                yaxis_title="Log 점도",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(148,163,184,0.45)", borderwidth=1, font=dict(size=12, color="#0f172a")),
                font=dict(color="#0f172a", size=15),
            )
            fig_cmp.update_xaxes(
                showgrid=False,
                zeroline=False,
                minor=dict(showgrid=False),
                showline=True,
                linewidth=2.2,
                linecolor="#334155",
                mirror=False,
                ticks="outside",
                ticklen=4,
                tickwidth=1,
                tickcolor="#64748b",
                tickfont=dict(color="#1f2937", size=13),
                title_font=dict(color="#111827", size=16),
            )
            fig_cmp.update_yaxes(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.16)",
                zeroline=False,
                minor=dict(showgrid=False),
                showline=True,
                linewidth=2.2,
                linecolor="#334155",
                mirror=False,
                ticks="outside",
                ticklen=4,
                tickwidth=1,
                tickcolor="#64748b",
                tickfont=dict(color="#1f2937", size=13),
                title_font=dict(color="#111827", size=16),
            )
            plotly_with_click_value(fig_cmp, f"curve_compare_exp_{selected_batch_id}", "Log 전단속도", "Log 점도", show_idle_hint=False)
    with c2:
        st.markdown("### 현재 배치 원료 모니터링")
        if not selected_batch_rows.empty:
            top_ing = selected_batch_rows[["name", "amount"]].dropna().sort_values("amount", ascending=False).head(10).copy()
            if not top_ing.empty:
                top_ing["성분표기"] = top_ing["name"].apply(ingredient_label)
                top_ing["기능"] = top_ing["name"].apply(ingredient_category)
                top_ing = top_ing.sort_values("amount", ascending=True)
                fig_i = px.bar(top_ing, x="amount", y="성분표기", orientation="h", color="기능", height=300, hover_data={"name": True, "기능": True, "amount": ":.2f"})
                fig_i.update_layout(
                    margin=dict(l=10, r=10, t=60, b=10),
                    yaxis_title="성분",
                    xaxis_title="투입량",
                    showlegend=False,
                    font=dict(color="#0f172a", size=15),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                )
                fig_i.update_xaxes(tickfont=dict(color="#1f2937", size=13), title_font=dict(color="#111827", size=16))
                fig_i.update_yaxes(tickfont=dict(color="#1f2937", size=13), title_font=dict(color="#111827", size=16))
                st.plotly_chart(fig_i, use_container_width=True)
        st.markdown("<div class='plain-title' style='margin-top:-12px;'>타입 불일치 원인 분석</div>", unsafe_allow_html=True)
        root_view = root_df.copy()
        if not root_view.empty and "기준중앙(q50)" in root_view.columns:
            root_view["기준값(q50)"] = pd.to_numeric(root_view["기준중앙(q50)"], errors="coerce").round(2)
            root_view["현재값"] = pd.to_numeric(root_view["현재"], errors="coerce").round(2)
            root_view["기여도"] = pd.to_numeric(root_view["기여도(%)"], errors="coerce").round(1)
            root_view = root_view[["성분", "기여도", "현재값", "기준값(q50)", "편차(절대량)", "해석"]]
        if "현재값" in root_view.columns:
            root_view = root_view[pd.to_numeric(root_view["현재값"], errors="coerce").fillna(0) > 0]
        if root_view.empty:
            st.info("현재 배치에 실제 투입된 원료가 없어 표시할 항목이 없습니다.")
        else:
            root_styler = (
                root_view.style
                .format({"기여도": "{:.1f}", "현재값": "{:.2f}", "기준값(q50)": "{:.2f}"})
                .bar(subset=["기여도"], color="#94a3b8", vmin=0)
                .set_table_styles(
                    [
                        {"selector": "thead th", "props": [("background-color", "#f1f5f9"), ("color", "#0f172a"), ("font-weight", "700")]},
                        {"selector": "tbody tr:nth-child(odd) td", "props": [("background-color", "#ffffff")]},
                        {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#f8fafc")]},
                        {"selector": "td, th", "props": [("border-color", "#e2e8f0")]},
                    ]
                )
                .set_properties(**{"background-color": "#ffffff", "color": "#0f172a"})
            )
            st.table(root_styler)

with st.expander("점도 등급 안정화 판정", expanded=True):
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("### 점도 등급 안정화 판정")
        current_grade, current_mean, selected_score_value, score_source, score_msg = classify_selected_viscosity_grade(df, batch_df, selected_batch_id, TARGET_RHEOLOGY_TYPE)
        if score_msg:
            st.info(score_msg)
        else:
            fig_grade = go.Figure()
            fig_grade.add_vrect(x0=0, x1=1, fillcolor="rgba(180,180,180,0.25)", line_width=0)
            fig_grade.add_vrect(x0=1, x1=3, fillcolor="rgba(120,200,120,0.2)", line_width=0)
            fig_grade.add_vrect(x0=3, x1=4, fillcolor="rgba(255,120,120,0.2)", line_width=0)
            fig_grade.add_annotation(x=0.5, y=0.95, xref="x", yref="paper", text="저점도 구간", showarrow=False, font=dict(color="#6b7280", size=12))
            fig_grade.add_annotation(x=2.0, y=0.95, xref="x", yref="paper", text="안정 구간", showarrow=False, font=dict(color="#166534", size=12))
            fig_grade.add_annotation(x=3.5, y=0.95, xref="x", yref="paper", text="고점도 구간", showarrow=False, font=dict(color="#b91c1c", size=12))
            fig_grade.add_trace(go.Scatter(x=[float(selected_score_value)], y=[0], mode="markers+text", text=[f"현재 ID {selected_batch_id}"], textposition="top center", marker=dict(size=14, color="#2563eb")))
            if sim_estimated_score_global is not None:
                fig_grade.add_trace(go.Scatter(x=[float(sim_estimated_score_global)], y=[0], mode="markers+text", text=["시뮬레이션"], textposition="bottom center", marker=dict(size=14, color="#16a34a", symbol="diamond")))
            fig_grade.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="점도 등급 점수 (0~4)",
                yaxis_title="",
                xaxis=dict(range=[0, 4], dtick=0.5, showgrid=False, zeroline=False),
                yaxis=dict(range=[-0.4, 0.4], showticklabels=False, zeroline=False, showgrid=False),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(color="#0f172a", size=15),
                showlegend=False,
            )
            fig_grade.update_xaxes(
                tickfont=dict(color="#1f2937", size=13),
                title_font=dict(color="#111827", size=16),
                showline=True,
                linewidth=1.4,
                linecolor="#64748b",
            )
            fig_grade.update_yaxes(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.16)",
                showline=True,
                linewidth=1.4,
                linecolor="#64748b",
            )
            st.plotly_chart(fig_grade, use_container_width=True)
    with c2:
        grade, grade_msg = selected_grade_global, selected_grade_msg_global
        type_now = str(selected_first.get("Rheology_Type", "UNKNOWN"))
        st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='plain-title' style='font-size:1.62rem;'>안정권 진입 배합 가이드</div>", unsafe_allow_html=True)
        if blend_df.empty:
            st.info("권장 조정 성분이 없습니다.")
        else:
            view = blend_df[["성분", "권장 조치", "현재 투입량", "권장 목표값", "권장 변경량", "안정권 하한(q25)", "안정권 상한(q75)", "실무 하한", "실무 상한", "레드라인", "실행 액션"]].copy()
            num_cols = ["현재 투입량", "권장 목표값", "권장 변경량", "안정권 하한(q25)", "안정권 상한(q75)", "실무 하한", "실무 상한"]
            for c in num_cols:
                view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
            view["레드라인"] = pd.to_numeric(view["레드라인"], errors="coerce").round(2)
            view = view[pd.to_numeric(view["현재 투입량"], errors="coerce").fillna(0) > 0]
            view["목표범위(q25~q75)"] = view.apply(lambda r: f"{r['안정권 하한(q25)']:.2f} ~ {r['안정권 상한(q75)']:.2f}", axis=1)
            view["실무범위"] = view.apply(lambda r: f"{r['실무 하한']:.2f} ~ {r['실무 상한']:.2f}", axis=1)
            view = view[["성분", "권장 조치", "현재 투입량", "권장 목표값", "목표범위(q25~q75)", "실무범위", "레드라인", "권장 변경량", "실행 액션"]]
            guide_styler = (
                view.style
                .format({"현재 투입량": "{:.2f}", "권장 목표값": "{:.2f}", "권장 변경량": "{:.2f}", "레드라인": "{:.2f}"})
                .set_table_styles(
                    [
                        {"selector": "thead th", "props": [("background-color", "#f1f5f9"), ("color", "#0f172a"), ("font-weight", "700")]},
                        {"selector": "tbody tr:nth-child(odd) td", "props": [("background-color", "#ffffff")]},
                        {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#f8fafc")]},
                        {"selector": "td, th", "props": [("border-color", "#e2e8f0")]},
                    ]
                )
                .set_properties(**{"background-color": "#ffffff", "color": "#0f172a"})
            )
            st.table(guide_styler)
        st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)
        if not grade_msg:
            current_type_color = "#dc2626"
            target_type_color = "#16a34a"
            grade_color = "#16a34a" if str(grade) == "안정" else "#dc2626"
            type_html = f"타입 상태: 현재 <span style='color:{current_type_color}; font-weight:900;'>{type_now}</span> / 목표 <span style='color:{target_type_color}; font-weight:900;'>{TARGET_RHEOLOGY_TYPE}</span>"
            grade_html = f"점도 상태: <span style='color:{grade_color}; font-weight:900;'>{grade}</span>"
            st.markdown(f"<div class='status-summary-box'><div class='status-summary-title'>현재 상태 결론</div><div class='status-line'>{type_html}</div><div class='status-line'>{grade_html}</div></div>", unsafe_allow_html=True)

# legacy selector block disabled
section_mode = ""

lower_left, lower_right = st.columns([1, 1], gap="large")

with lower_left:
    if section_mode == "목표 타입 불일치 원인 파악":
        st.markdown("<div id='section-cause'></div>", unsafe_allow_html=True)
        st.markdown("### 목표 타입 불일치 원인 파악")
        st.markdown("<div class='plain-title'>[목표 타입 기준 레올로지 편차]</div>", unsafe_allow_html=True)
        fig_cmp = go.Figure()
        selected_x, selected_y = selected_curve_x, selected_curve_y
        ref_ids = batch_df[
            (batch_df["Stability_Passed"] == True) & (batch_df["Rheology_Type"] == TARGET_RHEOLOGY_TYPE)
        ]["ID"].tolist()
        ref_ids = [rid for rid in ref_ids if rid != selected_batch_id][:20]
        show_compare_lines = st.checkbox("참고용 비교선(ID) 함께 보기", value=False, key=f"show_cmp_ids_{selected_batch_id}")
        ref_curves = []
        if selected_x is not None and len(ref_ids) > 0:
            for rid in ref_ids:
                row = df[df["ID"] == rid].iloc[0]
                rx, ry = get_batch_curve(row)
                if rx is not None and len(rx) > 3:
                    safe_mask = (selected_x > 0) & np.isfinite(selected_x) & np.isfinite(selected_y) & (selected_y > 0)
                    sx = selected_x[safe_mask]
                    if len(sx) > 3:
                        interp = np.interp(np.log10(sx), np.log10(rx), np.log10(ry))
                        ref_full = np.full_like(selected_x, np.nan, dtype=float)
                        ref_full[safe_mask] = interp
                        ref_curves.append(ref_full)
        if selected_x is not None and ref_curves:
            ref_arr = np.vstack(ref_curves)
            med = np.nanmedian(ref_arr, axis=0)
            std = np.nanstd(ref_arr, axis=0)
            upper = np.power(10, med + std)
            lower = np.power(10, med - std)
            fig_cmp.add_trace(go.Scatter(x=selected_x, y=upper, mode="lines", line=dict(width=0), showlegend=False))
            fig_cmp.add_trace(go.Scatter(x=selected_x, y=lower, mode="lines", fill="tonexty", fillcolor="rgba(122, 156, 140, 0.32)", line=dict(width=0), name="정상 범위"))
        if selected_x is not None:
            fig_cmp.add_trace(
                go.Scatter(
                    x=selected_x,
                    y=selected_y,
                    mode="lines+markers",
                    name=f"기준 Batch {selected_batch_id}",
                    line=dict(color="#2563eb", width=3),
                    opacity=0.35 if sim_active_global else 1.0,
                )
            )
            if sim_active_global and sim_curve_scale_global > 0:
                sim_y = selected_y * sim_curve_scale_global
                fig_cmp.add_trace(
                    go.Scatter(
                        x=selected_x,
                        y=sim_y,
                        mode="lines+markers",
                        name="시뮬레이션 적용 곡선",
                        line=dict(color="#16a34a", width=3, dash="dot"),
                    )
                )
        if show_compare_lines:
            compare_ids = sorted([rid for rid in ref_ids if rid != selected_batch_id], reverse=True)[:3]
            for cid in compare_ids:
                row = df[df["ID"] == cid].iloc[0]
                x, y = get_batch_curve(row)
                if x is not None:
                    fig_cmp.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"ID {cid}", line=dict(width=1)))
        if fig_cmp.data:
            fig_cmp.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), xaxis_type="log", yaxis_type="log", xaxis_title="Log 전단속도", yaxis_title="Log 점도")
            plotly_with_click_value(fig_cmp, f"curve_compare_{selected_batch_id}", "Log 전단속도", "Log 점도", show_idle_hint=False)
        else:
            st.info("비교 가능한 레올로지 데이터가 없습니다.")
    elif section_mode == "점도 등급 안정화 판정":
        st.markdown("### 점도 등급 안정화 판정")
        current_grade, current_mean, selected_score_value, score_source, score_msg = classify_selected_viscosity_grade(
            df, batch_df, selected_batch_id, TARGET_RHEOLOGY_TYPE
        )
        reference, _ = get_target_reference_curve(df, batch_df, TARGET_RHEOLOGY_TYPE)
        if score_msg:
            st.info(score_msg)
        else:
            fig_grade = go.Figure()
            fig_grade.add_vrect(x0=0, x1=1, fillcolor="rgba(180,180,180,0.25)", line_width=0)
            fig_grade.add_vrect(x0=1, x1=3, fillcolor="rgba(120,200,120,0.2)", line_width=0)
            fig_grade.add_vrect(x0=3, x1=4, fillcolor="rgba(255,120,120,0.2)", line_width=0)
            fig_grade.add_annotation(x=0.5, y=0.95, xref="x", yref="paper", text="저점도 구간", showarrow=False, font=dict(color="#6b7280", size=12))
            fig_grade.add_annotation(x=2.0, y=0.95, xref="x", yref="paper", text="안정 구간", showarrow=False, font=dict(color="#166534", size=12))
            fig_grade.add_annotation(x=3.5, y=0.95, xref="x", yref="paper", text="고점도 구간", showarrow=False, font=dict(color="#b91c1c", size=12))
            fig_grade.add_trace(
                go.Scatter(
                    x=[float(selected_score_value)],
                    y=[0],
                    mode="markers+text",
                    name="현재",
                    text=[f"현재 ID {selected_batch_id}"],
                    textposition="top center",
                    marker=dict(size=14, color="#2563eb"),
                )
            )
            if sim_estimated_score_global is not None:
                fig_grade.add_trace(
                    go.Scatter(
                        x=[float(sim_estimated_score_global)],
                        y=[0],
                        mode="markers+text",
                        name="시뮬레이션",
                        text=["시뮬레이션"],
                        textposition="bottom center",
                        marker=dict(size=14, color="#16a34a", symbol="diamond"),
                    )
                )
                fig_grade.add_annotation(
                    x=float(sim_estimated_score_global),
                    y=0.25,
                    yref="y",
                    text="적용 후",
                    showarrow=False,
                    font=dict(color="#15803d", size=11),
                )
            fig_grade.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="점도 등급 점수 (0~4)",
                yaxis_title="",
                xaxis=dict(range=[0, 4], dtick=0.5),
                yaxis=dict(range=[-0.4, 0.4], showticklabels=False, zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig_grade, use_container_width=True)
            if sim_estimated_score_global is not None:
                sim_state = grade_from_score(float(sim_estimated_score_global))
                if sim_state == "안정":
                    st.success(f"시뮬레이션 적용 후: {sim_state} (안정 구간 진입)")
                else:
                    st.warning(f"시뮬레이션 적용 후: {sim_state}")
    

with lower_right:
    if section_mode == "목표 타입 불일치 원인 파악":
        st.markdown("### 현재 배치 원료 모니터링")
        if not selected_batch_rows.empty:
            top_ing = (
                selected_batch_rows[["name", "amount"]]
                .dropna()
                .sort_values("amount", ascending=False)
                .head(10)
            )
            if not top_ing.empty:
                top_ing = top_ing.copy()
                top_ing["성분표기"] = top_ing["name"].apply(ingredient_label)
                top_ing["기능"] = top_ing["name"].apply(ingredient_category)
                top_ing = top_ing.sort_values("amount", ascending=True)
                fig_i = px.bar(
                    top_ing,
                    x="amount",
                    y="성분표기",
                    orientation="h",
                    color="기능",
                    height=300,
                    hover_data={"name": True, "기능": True, "amount": ":.2f"},
                )
                fig_i.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title="성분", xaxis_title="투입량", showlegend=False)
                st.plotly_chart(fig_i, use_container_width=True)

        st.markdown("<div id='section-diagnosis'></div>", unsafe_allow_html=True)
        st.markdown("<div class='plain-title' style='margin-top:-12px;'>타입 불일치 원인 분석</div>", unsafe_allow_html=True)
        root_view = root_df.copy()
        if not root_view.empty and "기준중앙(q50)" in root_view.columns:
            root_view["기준값(q50)"] = pd.to_numeric(root_view["기준중앙(q50)"], errors="coerce").round(2)
            root_view["현재값"] = pd.to_numeric(root_view["현재"], errors="coerce").round(2)
            root_view["기여도"] = pd.to_numeric(root_view["기여도(%)"], errors="coerce").round(1)
            root_view = root_view[["성분", "기여도", "현재값", "기준값(q50)", "편차(절대량)", "해석"]]
        if "현재값" in root_view.columns:
            root_view = root_view[pd.to_numeric(root_view["현재값"], errors="coerce").fillna(0) > 0]
        if root_view.empty:
            st.info("현재 배치에 실제 투입된 원료가 없어 표시할 항목이 없습니다.")
        else:
            root_styler = (
                root_view.style
                .format({"기여도": "{:.1f}", "현재값": "{:.2f}", "기준값(q50)": "{:.2f}"})
                .set_table_styles(
                    [
                        {"selector": "thead th", "props": [("background-color", "#f1f5f9"), ("color", "#0f172a"), ("font-weight", "700")]},
                        {"selector": "tbody tr:nth-child(odd) td", "props": [("background-color", "#ffffff")]},
                        {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#f8fafc")]},
                        {"selector": "td, th", "props": [("border-color", "#e2e8f0")]},
                    ]
                )
                .set_properties(**{"background-color": "#ffffff", "color": "#0f172a"})
            )
            st.dataframe(root_styler, use_container_width=True, hide_index=True)
    elif section_mode == "점도 등급 안정화 판정":
        grade, grade_msg = selected_grade_global, selected_grade_msg_global
        type_now = str(selected_first.get("Rheology_Type", "UNKNOWN"))
        st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='plain-title' style='font-size:1.62rem;'>안정권 진입 배합 가이드</div>", unsafe_allow_html=True)
        if blend_df.empty:
            st.info("권장 조정 성분이 없습니다.")
        else:
            view = blend_df[["성분", "권장 조치", "현재 투입량", "권장 목표값", "권장 변경량", "안정권 하한(q25)", "안정권 상한(q75)", "실무 하한", "실무 상한", "레드라인", "실행 액션"]].copy()
            num_cols = ["현재 투입량", "권장 목표값", "권장 변경량", "안정권 하한(q25)", "안정권 상한(q75)", "실무 하한", "실무 상한"]
            for c in num_cols:
                view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
            view["레드라인"] = pd.to_numeric(view["레드라인"], errors="coerce").round(2)
            view = view[pd.to_numeric(view["현재 투입량"], errors="coerce").fillna(0) > 0]
            view["목표범위(q25~q75)"] = view.apply(lambda r: f"{r['안정권 하한(q25)']:.2f} ~ {r['안정권 상한(q75)']:.2f}", axis=1)
            view["실무범위"] = view.apply(lambda r: f"{r['실무 하한']:.2f} ~ {r['실무 상한']:.2f}", axis=1)
            view = view[["성분", "권장 조치", "현재 투입량", "권장 목표값", "목표범위(q25~q75)", "실무범위", "레드라인", "권장 변경량", "실행 액션"]]
            if view.empty:
                st.info("현재 배치에 실제 투입된 원료 기준으로는 권장 조정 항목이 없습니다.")
            else:
                guide_styler = (
                    view.style
                    .format({"현재 투입량": "{:.2f}", "권장 목표값": "{:.2f}", "권장 변경량": "{:.2f}", "레드라인": "{:.2f}"})
                    .set_table_styles(
                        [
                            {"selector": "thead th", "props": [("background-color", "#f1f5f9"), ("color", "#0f172a"), ("font-weight", "700")]},
                            {"selector": "tbody tr:nth-child(odd) td", "props": [("background-color", "#ffffff")]},
                            {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#f8fafc")]},
                            {"selector": "td, th", "props": [("border-color", "#e2e8f0")]},
                        ]
                    )
                    .set_properties(**{"background-color": "#ffffff", "color": "#0f172a"})
                )
                st.dataframe(guide_styler, use_container_width=True, hide_index=True)

        st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)
        if grade_msg:
            st.info("점도 상태 계산이 불완전해 결론 확정이 어렵습니다. 데이터 확인이 필요합니다.")
        else:
            current_type_color = "#dc2626"
            target_type_color = "#16a34a"
            grade_color = "#16a34a" if str(grade) == "안정" else "#dc2626"
            type_html = (
                "타입 상태: 현재 "
                f"<span style='color:{current_type_color}; font-weight:900;'>{type_now}</span> / 목표 "
                f"<span style='color:{target_type_color}; font-weight:900;'>{TARGET_RHEOLOGY_TYPE}</span>"
            )
            grade_html = (
                "점도 상태: "
                f"<span style='color:{grade_color}; font-weight:900;'>{grade}</span>"
            )
            st.markdown(
                f"""
                <div class='status-summary-box'>
                  <div class='status-summary-title'>현재 상태 결론</div>
                  <div class='status-line' style='margin-top:2px;'>{type_html}</div>
                  <div class='status-line'>{grade_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

action_exp = st.expander("조치 실행", expanded=True)
action_exp.markdown("### 조치 실행")
action_left, action_right = action_exp.columns([1, 1], gap="large")

with action_left:
    sim_pool = (
        selected_batch_rows[["name", "amount"]]
        .dropna()
        .groupby("name", as_index=False)["amount"]
        .sum()
    )
    # 현재 배치에 실제로 투입된(>0) 원료만 시뮬레이터 대상으로 사용
    sim_pool["amount"] = pd.to_numeric(sim_pool["amount"], errors="coerce").fillna(0.0)
    sim_pool = sim_pool[sim_pool["amount"] > 0].copy()
    if sim_pool.empty:
        st.info("시뮬레이션할 성분 데이터가 없습니다.")
    else:
        top_names = sim_pool.sort_values("amount", ascending=False).head(8)["name"].tolist()
        sim_names = list(dict.fromkeys(top_names))
        sim_base = sim_pool[sim_pool["name"].isin(sim_names)].copy()

        btn_cols = st.columns(3)
        with btn_cols[0]:
            apply_guide = st.button("가이드 적용", use_container_width=True, key=f"apply_guide_{selected_batch_id}")
        with btn_cols[1]:
            run_sim = st.button("시뮬레이터 시작", use_container_width=True, key=f"run_sim_{selected_batch_id}")
        with btn_cols[2]:
            if st.button("시뮬레이션 초기화", use_container_width=True, key=f"clear_sim_{selected_batch_id}"):
                st.session_state.pop(sim_state_key, None)
                st.rerun()

        guide_map = {}
        if not blend_df.empty and "원성분" in blend_df.columns and "권장 목표값" in blend_df.columns:
            guide_map = {
                str(r["원성분"]): float(r["권장 목표값"])
                for _, r in blend_df.iterrows()
                if pd.notna(r.get("권장 목표값", np.nan))
            }

        if apply_guide:
            plan, err = find_auto_simulation_plan(
                df,
                batch_df,
                selected_batch_id,
                sim_base,
                TARGET_RHEOLOGY_TYPE,
                SIM_DELTA_MIN,
                SIM_DELTA_MAX,
                guide_map=guide_map,
                n_random=2500,
            )
            if plan is None:
                st.session_state[f"auto_plan_note_{selected_batch_id}"] = f"가이드 적용 실패: {err}"
            else:
                base_map_for_guide = sim_base.set_index("name")["amount"].astype(float).to_dict()
                for ing, delta_pct in plan["deltas"].items():
                    base_amt = float(base_map_for_guide.get(ing, 0.0))
                    # 자동탐색 결과(%)를 슬라이더 입력 단위(절대량)로 변환
                    delta_amt = base_amt * (float(delta_pct) / 100.0)
                    key = f"sim_delta_{selected_batch_id}_{ing}"
                    max_amt = base_amt * (SIM_DELTA_MAX / 100.0)
                    st.session_state[key] = float(np.clip(delta_amt, 0.0, max_amt))
                meta = plan["meta"]
                if meta["both_ok"]:
                    note = "가이드 적용: 목표 타입 일치 + 점도 안정 동시 달성 예상"
                elif meta["type_ok"]:
                    note = "가이드 적용: 목표 타입 일치 우선 달성 예상 (점도 안정은 미달성)"
                elif meta["grade_ok"]:
                    note = "가이드 적용: 점도 안정 우선 달성 예상 (타입 일치는 미달성)"
                else:
                    note = "가이드 적용: 제약 내 최적 근사값 적용 (타입/점도 동시 달성 조합 없음)"
                st.session_state[f"auto_plan_note_{selected_batch_id}"] = note
            st.rerun()

        sim_cols = st.columns(2)
        adjust_rows = []
        for idx, (_, row) in enumerate(sim_base.iterrows()):
            col = sim_cols[idx % 2]
            ing = str(row["name"])
            base_amt = float(row["amount"])
            s_key = f"sim_delta_{selected_batch_id}_{ing}"
            max_delta_amt = float(base_amt * (SIM_DELTA_MAX / 100.0))
            if s_key not in st.session_state:
                st.session_state[s_key] = 0.0
            # 과거 상태(%)가 남아있을 수 있어 범위 보정
            st.session_state[s_key] = float(np.clip(float(st.session_state[s_key]), 0.0, max_delta_amt))
            with col:
                delta_amt = st.slider(
                    f"{ingredient_label(ing)}",
                    min_value=0.0,
                    max_value=max_delta_amt,
                    step=0.01,
                    key=s_key,
                )
            delta_pct = (delta_amt / base_amt * 100.0) if base_amt > 0 else 0.0
            sim_amt = base_amt + delta_amt
            adjust_rows.append((ing, ingredient_label(ing), base_amt, delta_amt, delta_pct, sim_amt))

        sim_df = pd.DataFrame(adjust_rows, columns=["원성분", "성분", "현재", "변경량", "변경률(%)", "시뮬레이션"])
        sim_view = sim_df[["성분", "현재", "변경량", "시뮬레이션"]].copy()
        sim_styler = (
            sim_view.style
            .format({"현재": "{:.2f}", "변경량": "{:.2f}", "시뮬레이션": "{:.2f}"})
            .set_table_styles(
                [
                    {"selector": "thead th", "props": [("background-color", "#f1f5f9"), ("color", "#0f172a"), ("font-weight", "700")]},
                    {"selector": "tbody tr:nth-child(odd) td", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#f8fafc")]},
                    {"selector": "td, th", "props": [("border-color", "#e2e8f0")]},
                ]
            )
            .set_properties(**{"background-color": "#ffffff", "color": "#0f172a"})
        )
        st.table(sim_styler)
        auto_note = st.session_state.get(f"auto_plan_note_{selected_batch_id}")
        if auto_note:
            st.caption(auto_note)

        if sim_active_global:
            sim_type = str(sim_pred_type_global) if sim_pred_type_global else str(current_type_card)
            goal1 = str(sim_type).upper() == TARGET_RHEOLOGY_TYPE
            goal2 = (grade_from_score(float(sim_estimated_score_global)) == "안정") if sim_estimated_score_global is not None else False
            g1_txt = "달성" if goal1 else "미달성"
            g2_txt = "달성" if goal2 else "미달성"
            g1_color = "#16a34a" if goal1 else "#dc2626"
            g2_color = "#16a34a" if goal2 else "#dc2626"
            st.markdown(
                f"""
                <div class='panel'>
                  <div class='kpi-title'>시뮬레이터 결과 요약</div>
                  <div style='display:flex; gap:24px; flex-wrap:wrap; justify-content:center; align-items:center; text-align:center;'>
                    <div><b>목표 1 타입 일치:</b> <span style='color:{g1_color}; font-weight:800;'>{g1_txt}</span></div>
                    <div><b>목표 2 점도 안정:</b> <span style='color:{g2_color}; font-weight:800;'>{g2_txt}</span></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if sim_eval:
                if bool(sim_eval.get("release_ok", False)):
                    st.success("판정: 출하 가능합니다. (목표 타입 일치 + 점도 안정 + 실무 제약 통과)")
                else:
                    st.warning("판정: 출하 대기/보류 (실무 제약 위반 또는 목표 미충족)")
        if run_sim:
            violations = []
            baseline_warnings = []
            near_warnings = []
            constraint_rows = []
            sim_map = {r["원성분"]: float(r["시뮬레이션"]) for _, r in sim_df.iterrows()}
            base_map = {r["원성분"]: float(r["현재"]) for _, r in sim_df.iterrows()}
            changed_set = {r["원성분"] for _, r in sim_df.iterrows() if abs(float(r["변경량"])) >= 0.01}

            pivot_all = df.pivot_table(index="ID", columns="name", values="amount", aggfunc="first").fillna(0)
            stable_ids_local = batch_df[batch_df["Stability_Passed"] == True]["ID"].tolist()
            stable_pivot = pivot_all[pivot_all.index.isin(stable_ids_local)]

            for ing in sim_map.keys():
                status = "통과"
                reason = []
                if ing in stable_pivot.columns and len(stable_pivot) >= 15:
                    min_v = float(stable_pivot[ing].min())
                    max_v = float(stable_pivot[ing].max())
                    if sim_map[ing] < min_v or sim_map[ing] > max_v:
                        msg = f"{ingredient_label(ing)}: 허용 범위({min_v:.2f}~{max_v:.2f}) 밖입니다."
                        if ing in changed_set:
                            violations.append(msg)
                        else:
                            baseline_warnings.append(msg)
                        status = "위반"
                        reason.append("범위")
                    else:
                        span = max(max_v - min_v, 1e-9)
                        edge_ratio = min((sim_map[ing] - min_v) / span, (max_v - sim_map[ing]) / span)
                        if edge_ratio < 0.08:
                            near_warnings.append(f"{ingredient_label(ing)}: 허용범위 경계 근접")
                            if status != "위반":
                                status = "경계"
                            reason.append("범위근접")

                key = normalize_ing(ing)
                if key in REDLINE_RULES:
                    redline, risk_level, note = REDLINE_RULES[key]
                    if sim_map[ing] > redline:
                        msg = f"{ingredient_label(ing)}: 레드라인 {redline:.2f}% 초과 ({risk_level}) · {note}"
                        if ing in changed_set:
                            violations.append(msg)
                        else:
                            baseline_warnings.append(msg)
                        status = "위반"
                        reason.append("레드라인")
                    elif sim_map[ing] >= 0.95 * float(redline):
                        near_warnings.append(f"{ingredient_label(ing)}: 레드라인 95% 이상 근접")
                        if status != "위반":
                            status = "경계"
                        reason.append("레드라인근접")

                if ing in changed_set:
                    constraint_rows.append(
                        {
                            "성분": ingredient_label(ing),
                            "현재": round(float(base_map.get(ing, 0.0)), 2),
                            "시뮬레이션": round(float(sim_map.get(ing, 0.0)), 2),
                            "상태": status,
                            "근거": ",".join(reason) if reason else "-",
                        }
                    )

            if sum(abs(float(v)) >= float(MULTI_CHANGE_PCT) for v in sim_df["변경률(%)"]) > int(MULTI_CHANGE_N):
                violations.append(
                    f"동시에 {int(MULTI_CHANGE_N)}개 초과 성분을 ±{int(MULTI_CHANGE_PCT)}% 이상 변경할 수 없습니다."
                )

            if violations:
                st.error("시뮬레이션 차단: 실무 제약 위반")
                for v in violations:
                    st.write(f"- {v}")
                st.session_state[sim_eval_key] = {
                    "release_ok": False,
                    "goal_type_ok": False,
                    "goal_viscosity_ok": False,
                    "constraint_ok": False,
                    "status": "출하 대기",
                    "reason": "실무 제약 위반",
                }
            else:
                pred_label = None
                sim_model, sim_le, sim_features, sim_msg = load_tabpfn_artifacts()
                if sim_msg:
                    st.info("모델 파일이 없거나 로드 실패로 타입 예측은 건너뜁니다.")
                else:
                    try:
                        x_base = make_tabpfn_input(df, selected_batch_id, sim_features)
                        for _, r in sim_df.iterrows():
                            ing = r["원성분"]
                            if ing in x_base.columns:
                                x_base.at[0, ing] = float(r["시뮬레이션"])
                        pred = sim_model.predict(x_base)[0]
                        pred_label = str(pred) if isinstance(pred, (str, np.str_)) else sim_le.inverse_transform([int(pred)])[0]
                    except Exception:
                        pred_label = None

                stable_meta = batch_df[batch_df["Stability_Passed"] == True]
                ref_ids = stable_meta[stable_meta["Rheology_Type"] == TARGET_RHEOLOGY_TYPE]["ID"].tolist()
                if not ref_ids:
                    ref_ids = stable_meta["ID"].tolist()

                sim_score = None
                curve_scale = 1.0
                if selected_batch_id in pivot_all.index and ref_ids:
                    ref_vec = pivot_all[pivot_all.index.isin(ref_ids)].median()
                    before = pivot_all.loc[selected_batch_id].copy()
                    after = before.copy()
                    for _, r in sim_df.iterrows():
                        if r["원성분"] in after.index:
                            after[r["원성분"]] = float(r["시뮬레이션"])
                    base_dist = float((before - ref_vec).abs().sum())
                    sim_dist = float((after - ref_vec).abs().sum())
                    ref_curve, _ = get_target_reference_curve(df, batch_df, TARGET_RHEOLOGY_TYPE)
                    current_curve_mean = parse_rheology_mean(selected_first.get("Rheology_Data"))
                    sim_curve_mean = estimate_curve_mean_knn(after, pivot_all, build_curve_mean_by_id(df))
                    if sim_curve_mean is None and current_curve_mean is not None and len(ref_curve) > 10 and base_dist > 1e-9:
                        current_score = compute_grade_score(float(current_curve_mean), ref_curve)
                        move_ratio = np.clip((base_dist - sim_dist) / base_dist, -1.0, 1.0)
                        sim_score = float(np.clip(current_score + (2.0 - current_score) * move_ratio, 0, 4))
                    elif sim_curve_mean is not None and len(ref_curve) > 10:
                        sim_score = float(compute_grade_score(float(sim_curve_mean), ref_curve))
                        if current_curve_mean is not None and current_curve_mean > 1e-9:
                            curve_scale = max(float(sim_curve_mean) / float(current_curve_mean), 0.2)

                st.session_state[sim_state_key] = {
                    "sim_map": sim_map,
                    "sim_score": sim_score,
                    "pred_type": pred_label,
                    "curve_scale": curve_scale,
                    "changed_count": int(len(changed_set)),
                }
                goal_type_ok = str(pred_label).upper() == TARGET_RHEOLOGY_TYPE if pred_label is not None else False
                goal_viscosity_ok = (sim_score is not None) and (1.0 <= float(sim_score) <= 3.0)
                release_ok = bool(goal_type_ok and goal_viscosity_ok)
                st.session_state[sim_eval_key] = {
                    "release_ok": release_ok,
                    "goal_type_ok": bool(goal_type_ok),
                    "goal_viscosity_ok": bool(goal_viscosity_ok),
                    "constraint_ok": True,
                    "status": "출하 가능" if release_ok else "출하 대기",
                    "reason": "목표 2개 충족" if release_ok else "목표 미충족",
                }
                st.success("시뮬레이션 적용 완료: 상단/중간 그래프에 반영되었습니다.")
                if baseline_warnings:
                    st.warning("참고: 변경하지 않은 기존 성분 중 기준 이탈 항목이 있습니다.")
                if near_warnings:
                    st.warning("실무 경계 경고")
                    for w in sorted(set(near_warnings)):
                        st.write(f"- {w}")
                if constraint_rows:
                    st.dataframe(pd.DataFrame(constraint_rows), use_container_width=True, hide_index=True, height=180)
                st.rerun()

with action_right:
    st.markdown("<div class='checklist-title'>체크리스트</div>", unsafe_allow_html=True)
    release_mode = bool(sim_eval.get("release_ok", False))
    if sim_eval:
        if release_mode:
            st.markdown("**출하 전 최종 체크리스트**")
            st.checkbox("최종 시험성적서(CoA) 승인", key=f"rel_chk_1_{selected_batch_id}")
            st.checkbox("배치기록(BMR) 이탈 없음 확인", key=f"rel_chk_2_{selected_batch_id}")
            st.checkbox("출하 승인자 검토/서명 완료", key=f"rel_chk_3_{selected_batch_id}")
    if not release_mode:
        st.markdown(
            "<div class='priority-card priority-red'><div class='priority-title'>즉시 조치</div></div>",
            unsafe_allow_html=True,
        )
        st.checkbox("현재 상태 결론에서 타입/점도 상태 확인", key=f"red_chk_1_{selected_batch_id}")
        st.checkbox("타입 불일치 원인 분석 표 상위 3개 성분 확인", key=f"red_chk_2_{selected_batch_id}")
        st.checkbox("안정권 진입 배합 가이드의 권장 조치 방향 확인", key=f"red_chk_3_{selected_batch_id}")

        st.markdown(
            "<div class='priority-card priority-yellow'><div class='priority-title'>우선 확인</div></div>",
            unsafe_allow_html=True,
        )
        st.checkbox("가이드 적용 버튼 실행", key=f"yellow_chk_1_{selected_batch_id}")
        st.checkbox("시뮬레이터 시작 후 목표 1/2 달성 여부 확인", key=f"yellow_chk_2_{selected_batch_id}")
        st.checkbox("점도 안정 판정 그래프에서 점 위치 이동 확인", key=f"yellow_chk_3_{selected_batch_id}")

        st.markdown(
            "<div class='priority-card priority-green'><div class='priority-title'>유지 관리</div></div>",
            unsafe_allow_html=True,
        )
        st.checkbox("출하 가능/대기 판정 메시지 확인", key=f"green_chk_1_{selected_batch_id}")
        st.checkbox("적용한 조정값(변경%) 표 확인 및 기록", key=f"green_chk_2_{selected_batch_id}")
        st.checkbox("미충족 시 조정값 수정 후 시뮬레이터 재실행", key=f"green_chk_3_{selected_batch_id}")

with st.expander("원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)
