import os
import uuid
import datetime as dt

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from streamlit_autorefresh import st_autorefresh  # untuk auto refresh
from dotenv import load_dotenv
load_dotenv()


# ============ CONFIG APP ============

st.set_page_config(
    page_title="Car Insurance & UX Analytics",
    layout="wide",
)

PRIMARY_COLOR = "#0F4C81"
SECONDARY_COLOR = "#1E88E5"
ACCENT_COLOR = "#F39C12"
SUCCESS_COLOR = "#27AE60"
DANGER_COLOR = "#E74C3C"
BG_COLOR = "#F7F9FB"

COLOR_PALETTE = [
    "#0F4C81", "#1E88E5", "#191B1D", "#64B5F6",
    "#F39C12", "#E67E22", "#27AE60", "#2ECC71",
    "#9B59B6", "#E74C3C"
]

st.markdown(
    f"""
    <style>
    .main {{ background-color: {BG_COLOR}; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 1.5rem; }}
    h1, h2, h3, h4 {{ color: {PRIMARY_COLOR}; }}
    .stMetric {{ 
        background-color: white; 
        border-radius: 12px; 
        padding: 15px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 24px;
        font-weight: bold;
        color: {PRIMARY_COLOR};
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 14px;
        color: #666;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============ DB CONNECTION ============

POSTGRES_USER = st.secrets["POSTGRES_USER"]
POSTGRES_PASSWORD = st.secrets["POSTGRES_PASSWORD"]
POSTGRES_DB = st.secrets["POSTGRES_DB"]
POSTGRES_HOST = st.secrets["POSTGRES_HOST_OUT"]
POSTGRES_PORT = st.secrets["POSTGRES_PORT_OUT"]

DB_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:"
    f"{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_engine(DB_URL)

# ============ SESSION & LOGGING SETUP ============

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "last_event_ts" not in st.session_state:
    st.session_state["last_event_ts"] = dt.datetime.utcnow()

if "last_page" not in st.session_state:
    st.session_state["last_page"] = None

# state untuk toggle Top Segmen Risiko
if "show_risk_seg" not in st.session_state:
    st.session_state["show_risk_seg"] = False

# state mode slider waktu (auto/manual) di page 2
if "time_range_manual" not in st.session_state:
    st.session_state["time_range_manual"] = False

SESSION_ID = st.session_state["session_id"]
USER_ID = "anonymous"  # ganti kalau ada mekanisme login


def log_event(
    event_type,
    page_name,
    page_url=None,
    element_id=None,
    dwell_time_sec=None,
    is_bounce=None,
    error_code=None,
    funnel_stage=None,
    usability_score=None,
):
    """Insert 1 baris event ke realtime.user_events"""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO realtime.user_events (
                    session_id, user_id, page_url, page_name,
                    event_type, element_id, dwell_time_sec,
                    is_bounce, error_code, funnel_stage,
                    usability_score, created_at
                )
                VALUES (
                    :session_id, :user_id, :page_url, :page_name,
                    :event_type, :element_id, :dwell_time_sec,
                    :is_bounce, :error_code, :funnel_stage,
                    :usability_score, now()
                )
            """),
            {
                "session_id": SESSION_ID,
                "user_id": USER_ID,
                "page_url": page_url,
                "page_name": page_name,
                "event_type": event_type,
                "element_id": element_id,
                "dwell_time_sec": dwell_time_sec,
                "is_bounce": is_bounce,
                "error_code": error_code,
                "funnel_stage": funnel_stage,
                "usability_score": usability_score,
            },
        )


def track_filter_change(session_key, current_value, element_id, page_name, funnel_stage):
    """
    Mencatat event 'click' ketika nilai filter/elemen berubah.
    """
    prev_key = f"{session_key}_prev"

    # Inisialisasi pertama kali tanpa mencatat click
    if prev_key not in st.session_state:
        st.session_state[prev_key] = current_value
        return

    if st.session_state[prev_key] != current_value:
        log_event(
            event_type="click",
            page_name=page_name,
            element_id=element_id,
            funnel_stage=funnel_stage,
        )
        st.session_state[prev_key] = current_value


# ============ DATA LOAD FUNCTIONS ============

@st.cache_data(ttl=300)
def load_dw_data():
    fact = pd.read_sql("SELECT * FROM dw.fact_claim", engine)
    dim_cust = pd.read_sql("SELECT * FROM dw.dim_customer", engine)
    dim_veh = pd.read_sql("SELECT * FROM dw.dim_vehicle", engine)
    dim_loc = pd.read_sql("SELECT * FROM dw.dim_location", engine)
    dim_beh = pd.read_sql("SELECT * FROM dw.dim_behavior", engine)

    df = (
        fact
        .merge(dim_cust, on="customer_id")
        .merge(dim_veh, on="vehicle_id")
        .merge(dim_loc, left_on="location_id", right_on="location_key")
        .merge(dim_beh, on="behavior_id")
    )
    return df


def load_realtime_behavior():
    """JANGAN di-cache supaya benar-benar realtime."""
    ses = pd.read_sql("SELECT * FROM realtime.session_metrics", engine)
    ev = pd.read_sql("SELECT * FROM realtime.user_events", engine)

    for col in ["start_time", "end_time"]:
        if col in ses.columns:
            ses[col] = pd.to_datetime(ses[col])
    if "created_at" in ev.columns:
        ev["created_at"] = pd.to_datetime(ev["created_at"])

    return ses, ev

# ============ CHART HELPER FUNCTIONS ============


def create_donut_chart(df, values_col, names_col, title):
    fig = px.pie(
        df,
        values=values_col,
        names=names_col,
        hole=0.55,
        color_discrete_sequence=COLOR_PALETTE
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        textfont_size=11,
        hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,}<br>Persentase: %{percent}<extra></extra>",
        pull=[0.05] * len(df),
    )
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color=PRIMARY_COLOR),
            x=0.5,
            xanchor="center",
        ),
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{df[values_col].sum():,}</b>",
                x=0.5,
                y=0.5,
                font_size=18,
                font_color=PRIMARY_COLOR,
                showarrow=False,
            )
        ],
        transition={"duration": 800, "easing": "cubic-in-out"},
    )
    return fig


def create_vertical_bar(df, x_col, y_col, title, color=PRIMARY_COLOR):
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color_discrete_sequence=[color]
    )
    fig.update_traces(
        texttemplate="%{y:,.0f}",
        textposition="outside",
        textfont_size=10,
        hovertemplate="<b>%{x}</b><br>Jumlah: %{y:,}<extra></extra>",
        marker=dict(line=dict(width=0)),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PRIMARY_COLOR)),
        xaxis_title="",
        yaxis_title="Jumlah Nasabah",
        height=320,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E0E0E0"),
        transition={"duration": 600, "easing": "elastic-out"},
    )
    return fig


def create_heatmap(df, x_col, y_col, z_col, title):
    pivot_df = df.pivot_table(
        index=y_col, columns=x_col, values=z_col, aggfunc="mean"
    )

    fig = px.imshow(
        pivot_df,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        text_auto=".2f",
    )
    fig.update_traces(
        textfont=dict(size=10),
        hovertemplate="%{x} x %{y}<br>Rata-rata Outcome: %{z:.3f}<extra></extra>",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PRIMARY_COLOR)),
        height=380,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="",
        coloraxis_colorbar=dict(
            title="Outcome",
            thickness=15,
            len=0.7,
        ),
        transition={"duration": 600, "easing": "cubic-in-out"},
    )
    return fig


def create_scatter_map(df, lat_col, lon_col, size_col, state_col, title):
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        size=size_col,
        color=size_col,
        color_continuous_scale=[
            [0.0, "#FFED6F"],
            [0.15, "#FFC837"],
            [0.3, "#FF8C00"],
            [0.45, "#FF4500"],
            [0.6, "#DC143C"],
            [0.75, "#B22222"],
            [0.9, "#8B0000"],
            [1.0, "#4B0000"],
        ],
        size_max=35,
        zoom=3,
        mapbox_style="carto-positron",
        hover_name=state_col,
        hover_data={
            lat_col: False,
            lon_col: False,
            size_col: ":,",
            state_col: False,
        },
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Jumlah Klaim: %{customdata[2]:,}<extra></extra>"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PRIMARY_COLOR)),
        height=450,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="Jumlah<br>Klaim",
            thickness=15,
            len=0.6,
        ),
        transition={"duration": 1000, "easing": "cubic-in-out"},
    )
    return fig


def create_impact_chart(df, x_col, title, x_title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df["claims"],
            name="Jumlah Klaim",
            marker_color=PRIMARY_COLOR,
            text=df["claims"],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Jumlah Klaim: %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df["claim_rate"] * 100,
            name="Tingkat Klaim",
            mode="lines+markers",
            marker=dict(color=DANGER_COLOR, size=8),
            line=dict(color=DANGER_COLOR, width=2),
            hovertemplate="<b>%{x}</b><br>Tingkat Klaim: %{y:.1f}%<extra></extra>",
            connectgaps=True,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PRIMARY_COLOR)),
        xaxis=dict(title=dict(text=x_title)),
        height=400,
        margin=dict(l=20, r=20, t=70, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        transition={"duration": 700, "easing": "cubic-in-out"},
    )

    fig.update_yaxes(
        title=dict(text="Jumlah Klaim", font=dict(color=PRIMARY_COLOR)),
        showgrid=True,
        gridcolor="#E0E0E0",
        secondary_y=False,
    )

    fig.update_yaxes(
        title=dict(text="Tingkat Klaim (%)", font=dict(color=DANGER_COLOR)),
        showgrid=False,
        secondary_y=True,
    )

    return fig


def create_funnel_chart(df, x_col, y_col, title):
    fig = px.funnel(
        df,
        x=x_col,
        y=y_col,
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig.update_traces(
        texttemplate="%{value:,}",
        textposition="inside",
        textfont_size=11,
        hovertemplate="<b>%{y}</b><br>Jumlah: %{x:,}<extra></extra>",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PRIMARY_COLOR)),
        height=380,
        margin=dict(l=20, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 800, "easing": "cubic-in-out"},
    )
    return fig

with st.sidebar:
    # Logo di kiri atas sidebar
    st.image("Logo_PastiAman.png", use_container_width=True)

    st.markdown("---")

    st.title("Navigasi")
    page = st.radio(
        "Pilih dashboard:",
        ["Asuransi Mobil", "User Behavior"]
    )
    st.markdown("---")

# ============ LOG PAGE VIEW (REALTIME) ============

now_ts = dt.datetime.utcnow()
last_ts = st.session_state["last_event_ts"]
dwell = (now_ts - last_ts).total_seconds()
st.session_state["last_event_ts"] = now_ts

current_page = page

log_event(
    event_type="page_view",
    page_name=current_page,
    dwell_time_sec=dwell if dwell > 0 else None,
    funnel_stage="landing" if current_page == "Asuransi Mobil" else "behavior_dashboard",
)

st.session_state["last_page"] = current_page

# ============ PAGE 1: ASURANSI MOBIL ============

if page == "Asuransi Mobil":
    df = load_dw_data()

    st.title("Dashboard PastiAman Autocare")
    st.caption("Analisis klaim asuransi mobil dengan profil risiko, distribusi geografis, dan segmentasi pelanggan.")

    # ---- FILTERS ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Data")

    years = sorted(df["vehicle_year"].dropna().unique())
    year_sel = st.sidebar.multiselect(
        "Tahun Kendaraan",
        years,
        default=years,
        key="filter_vehicle_year",
    )

    all_states = sorted(df["state"].dropna().unique())
    state_sel = st.sidebar.multiselect(
        "Negara Bagian",
        all_states,
        default=all_states[:10] if len(all_states) > 10 else all_states,
        key="filter_state",
    )

    genders = sorted(df["gender"].dropna().unique())
    gender_sel = st.sidebar.multiselect(
        "Jenis Kelamin",
        genders,
        default=genders,
        key="filter_gender",
    )

    vehicle_types = sorted(df["vehicle_type"].dropna().unique())
    vehicle_sel = st.sidebar.multiselect(
        "Tipe Kendaraan",
        vehicle_types,
        default=vehicle_types,
        key="filter_vehicle_type",
    )

    # ---- LOG FILTER CLICKS (STAGE 2: segment_filtering) ----
    track_filter_change(
        session_key="filter_vehicle_year",
        current_value=year_sel,
        element_id="filter_vehicle_year",
        page_name="Asuransi Mobil",
        funnel_stage="segment_filtering",
    )
    track_filter_change(
        session_key="filter_state",
        current_value=state_sel,
        element_id="filter_state",
        page_name="Asuransi Mobil",
        funnel_stage="segment_filtering",
    )
    track_filter_change(
        session_key="filter_gender",
        current_value=gender_sel,
        element_id="filter_gender",
        page_name="Asuransi Mobil",
        funnel_stage="segment_filtering",
    )
    track_filter_change(
        session_key="filter_vehicle_type",
        current_value=vehicle_sel,
        element_id="filter_vehicle_type",
        page_name="Asuransi Mobil",
        funnel_stage="segment_filtering",
    )

    filtered = df.copy()
    if year_sel:
        filtered = filtered[filtered["vehicle_year"].isin(year_sel)]
    if state_sel:
        filtered = filtered[filtered["state"].isin(state_sel)]
    if gender_sel:
        filtered = filtered[filtered["gender"].isin(gender_sel)]
    if vehicle_sel:
        filtered = filtered[filtered["vehicle_type"].isin(vehicle_sel)]

    # ---- TOP KPI METRICS ----
    st.subheader("Indikator Kinerja Utama")

    jumlah_nasabah = int(filtered["customer_id"].nunique()) if not filtered.empty else 0
    jumlah_claim = int(filtered[filtered["outcome"] == 1]["outcome"].sum()) if not filtered.empty else 0
    belum_claim = int(filtered[filtered["outcome"] == 0]["outcome"].count()) if not filtered.empty else 0
    tingkat_claim = (jumlah_claim / jumlah_nasabah * 100) if jumlah_nasabah > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Jumlah Nasabah", f"{jumlah_nasabah:,}")
    with kpi2:
        st.metric("Jumlah Klaim", f"{jumlah_claim:,}")
    with kpi3:
        st.metric("Belum Klaim", f"{belum_claim:,}")
    with kpi4:
        st.metric("Tingkat Klaim", f"{tingkat_claim:.1f}%")

    st.markdown("---")

    # ---- CUSTOMER PROFILE ANALYSIS ----
    st.subheader("Profil Pelanggan")

    ccol1, ccol2, ccol3 = st.columns(3)

    # Donut 1: gender
    with ccol1:
        if not filtered.empty:
            gender_dist = (
                filtered.groupby("gender")
                .agg(customers=("customer_id", "nunique"))
                .reset_index()
            )
            if not gender_dist.empty:
                fig_gender = create_donut_chart(
                    gender_dist, "customers", "gender", "Distribusi Jenis Kelamin"
                )
                st.plotly_chart(fig_gender, use_container_width=True)
            else:
                st.info("Tidak ada data jenis kelamin untuk saat ini.")
        else:
            st.info("Tidak ada data untuk saat ini.")

    # Donut 2: education
    with ccol2:
        if not filtered.empty:
            edu_dist = (
                filtered.groupby("education")
                .agg(customers=("customer_id", "nunique"))
                .reset_index()
                .sort_values("customers", ascending=False)
            )
            if not edu_dist.empty:
                edu_label_map = {
                    "University": "Univ.",
                    "High School": "High Sch",
                    "High Schc": "High Sch",  # jaga-jaga typo
                }
                edu_dist["education_display"] = edu_dist["education"].map(
                    edu_label_map
                ).fillna(edu_dist["education"])
                fig_edu = create_donut_chart(
                    edu_dist,
                    "customers",
                    "education_display",
                    "Distribusi Pendidikan",
                )
                st.plotly_chart(fig_edu, use_container_width=True)
            else:
                st.info("Tidak ada data pendidikan untuk saat ini.")
        else:
            st.info("")

    # Donut 3: married
    with ccol3:
        if not filtered.empty:
            married_dist = (
                filtered.groupby("married")
                .agg(customers=("customer_id", "nunique"))
                .reset_index()
            )
            if not married_dist.empty:
                married_dist["married_label"] = married_dist["married"].map(
                    {True: "Menikah", False: "Blm Menikah"}
                )
                fig_married = create_donut_chart(
                    married_dist, "customers", "married_label", "Status Pernikahan"
                )
                st.plotly_chart(fig_married, use_container_width=True)
            else:
                st.info("Tidak ada data status pernikahan untuk saat ini.")
        else:
            st.info("")

    ccol4, ccol5, ccol6 = st.columns(3)

    with ccol4:
        if not filtered.empty:
            accidents_dist = (
                filtered.groupby("past_accidents")
                .agg(customers=("customer_id", "nunique"))
                .reset_index()
                .sort_values("past_accidents")
            )
            if not accidents_dist.empty:
                fig_accidents = create_vertical_bar(
                    accidents_dist,
                    "past_accidents",
                    "customers",
                    "Distribusi Kecelakaan Sebelumnya",
                    color=DANGER_COLOR,
                )
                st.plotly_chart(fig_accidents, use_container_width=True)
            else:
                st.info("Tidak ada data kecelakaan sebelumnya.")
        else:
            st.info("")

    with ccol5:
        if not filtered.empty:
            violations_dist = (
                filtered.groupby("speeding_violations")
                .agg(customers=("customer_id", "nunique"))
                .reset_index()
                .sort_values("speeding_violations")
            )
            if not violations_dist.empty:
                fig_violations = create_vertical_bar(
                    violations_dist,
                    "speeding_violations",
                    "customers",
                    "Distribusi Pelanggaran Kecepatan",
                    color=ACCENT_COLOR,
                )
                st.plotly_chart(fig_violations, use_container_width=True)
            else:
                st.info("Tidak ada data pelanggaran kecepatan.")
        else:
            st.info("")

    with ccol6:
        if not filtered.empty:
            duis_dist = (
                filtered.groupby("duis")
                .agg(customers=("customer_id", "nunique"))
                .reset_index()
                .sort_values("duis")
            )
            if not duis_dist.empty:
                fig_duis = create_vertical_bar(
                    duis_dist,
                    "duis",
                    "customers",
                    "Distribusi DUI (Driving Under Influence)",
                    color=SECONDARY_COLOR,
                )
                st.plotly_chart(fig_duis, use_container_width=True)
            else:
                st.info("Tidak ada data DUI.")
        else:
            st.info("")

    # ---- AGE FUNNEL ----
    st.subheader("Distribusi Klaim per Kelompok Usia")

    if not filtered.empty:
        age_funnel = (
            filtered.groupby("age")
            .agg(claims=("outcome", "sum"))
            .reset_index()
            .sort_values("claims", ascending=False)
        )
        if not age_funnel.empty:
            fig_funnel = create_funnel_chart(
                age_funnel,
                "claims",
                "age",
                "Funnel Klaim Berdasarkan Kelompok Usia",
            )
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.info("Tidak ada klaim pada saat ini.")
    else:
        st.info("Tidak ada data untuk saat ini.")

    st.markdown("---")

    # ---- GEOGRAPHIC ANALYSIS ----
    st.subheader("Negara Bagian dengan Risiko Tertinggi")

    if not filtered.empty:
        map_df = (
            filtered.groupby(["latitude", "longitude", "state"], as_index=False)
            .agg(claims=("outcome", "sum"))
            .dropna(subset=["latitude", "longitude"])
        )

        if not map_df.empty:
            fig_map = create_scatter_map(
                map_df,
                "latitude",
                "longitude",
                "claims",
                "state",
                "Peta Distribusi Risiko per Negara Bagian",
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Tidak ada data lokasi yang lengkap untuk ditampilkan di peta.")
    else:
        st.info("Tidak ada data untuk saat ini.")

    st.markdown("---")

    # ---- VEHICLE ANALYSIS ----
    st.subheader("Analisis Kendaraan")

    vcol1, vcol2 = st.columns(2)

    with vcol1:
        if not filtered.empty:
            by_vehicle = (
                filtered.groupby("vehicle_type")
                .agg(claims=("outcome", "sum"))
                .reset_index()
            )
            if not by_vehicle.empty:
                fig_vehicle_donut = create_donut_chart(
                    by_vehicle,
                    "claims",
                    "vehicle_type",
                    "Klaim per Tipe Kendaraan",
                )
                st.plotly_chart(fig_vehicle_donut, use_container_width=True)
            else:
                st.info("Tidak ada data klaim per tipe kendaraan.")
        else:
            st.info("")

    with vcol2:
        if not filtered.empty:
            by_year = (
                filtered.groupby("vehicle_year")
                .agg(claims=("outcome", "sum"))
                .reset_index()
                .sort_values("vehicle_year")
            )

            if not by_year.empty:
                fig_year = px.bar(
                    by_year,
                    x="vehicle_year",
                    y="claims",
                    color_discrete_sequence=[PRIMARY_COLOR],
                )
                fig_year.update_traces(
                    texttemplate="%{y:,.0f}",
                    textposition="outside",
                    textfont_size=10,
                    hovertemplate="<b>Tahun %{x}</b><br>Jumlah Klaim: %{y:,}<extra></extra>",
                )
                fig_year.update_layout(
                    title=dict(
                        text="Klaim per Tahun Kendaraan",
                        font=dict(size=15, color=PRIMARY_COLOR),
                    ),
                    xaxis=dict(title=dict(text="Tahun Kendaraan")),
                    yaxis=dict(
                        title=dict(
                            text="Jumlah Klaim",
                            font=dict(color=PRIMARY_COLOR),
                        ),
                        showgrid=True,
                        gridcolor="#E0E0E0",
                    ),
                    height=320,
                    margin=dict(l=20, r=20, t=50, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_showgrid=False,
                    transition={"duration": 700, "easing": "cubic-in-out"},
                )
                st.plotly_chart(fig_year, use_container_width=True)
            else:
                st.info("Tidak ada data klaim per tahun kendaraan.")
        else:
            st.info("")

    st.markdown("---")

    # ---- RISK FACTOR ANALYSIS ----
    st.subheader("Analisis Faktor Risiko")

    if not filtered.empty:
        # Heatmap risiko
        heatmap_data = (
            filtered.groupby(["vehicle_type", "age"])
            .agg(avg_outcome=("outcome", "mean"))
            .reset_index()
        )
        if not heatmap_data.empty:
            fig_heatmap = create_heatmap(
                heatmap_data,
                "age",
                "vehicle_type",
                "avg_outcome",
                "Peta Panas Risiko: Tipe Kendaraan vs Kelompok Usia",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Tidak ada data untuk peta panas risiko.")

        # BUTTON TOGGLE untuk Top Segmen Risiko
        if st.button(
            "Lihat Top Segmen Risiko Tertinggi Berdasarkan Tipe Kendaraan vs Kelompok Usia",
            key="btn_risk_focus",
        ):
            st.session_state["show_risk_seg"] = not st.session_state["show_risk_seg"]
            if st.session_state["show_risk_seg"]:
                log_event(
                    event_type="click",
                    page_name="Asuransi Mobil",
                    element_id="btn_risk_focus",
                    funnel_stage="risk_exploration",
                )

        if st.session_state["show_risk_seg"]:

            risk_seg = (
                filtered.groupby(["vehicle_type", "age"])
                .agg(
                    customers=("outcome", "size"),
                    claims=("outcome", "sum"),
                )
                .reset_index()
            )

            if not risk_seg.empty:
                risk_seg = risk_seg[risk_seg["customers"] > 0].copy()
                risk_seg["claim_rate"] = (
                    risk_seg["claims"] / risk_seg["customers"] * 100
                ).round(2)

                risk_seg = (
                    risk_seg.sort_values("claim_rate", ascending=False).head(10)
                )

                if not risk_seg.empty:
                    risk_seg = risk_seg.reset_index(drop=True)
                    risk_seg.index = risk_seg.index + 1
                    risk_seg.index.name = "Rank"

                    tampil = risk_seg.rename(
                        columns={
                            "vehicle_type": "Tipe Kendaraan",
                            "age": "Usia",
                            "customers": "Jumlah Nasabah",
                            "claims": "Jumlah Klaim",
                            "claim_rate": "Tingkat Klaim (%)",
                        }
                    )

                    st.markdown("### Top Segmen Risiko Tertinggi")
                    st.dataframe(
                        tampil,
                        use_container_width=True,
                        height=350,
                    )
                else:
                    st.info("Tidak ada segmen risiko yang dapat ditampilkan.")
            else:
                st.info("Tidak ada segmen risiko yang dapat dihitung.")
    else:
        st.info("Tidak ada data saat ini.")

    st.markdown("---")

    # ---- BEHAVIOR ANALYSIS ----
    st.subheader("Analisis Perilaku Mengemudi")

    if not filtered.empty:
        dui_data = (
            filtered.groupby("duis")
            .agg(
                claims=("outcome", "sum"),
                claim_rate=("outcome", "mean"),
            )
            .reset_index()
        )
        if not dui_data.empty:
            fig_dui = create_impact_chart(
                dui_data,
                "duis",
                "Dampak DUI terhadap Klaim",
                "Jumlah DUI",
            )
            st.plotly_chart(fig_dui, use_container_width=True)

        bcol1, bcol2 = st.columns(2)

        with bcol1:
            accidents_data = (
                filtered.groupby("past_accidents")
                .agg(
                    claims=("outcome", "sum"),
                    claim_rate=("outcome", "mean"),
                )
                .reset_index()
            )
            if not accidents_data.empty:
                fig_accidents_impact = create_impact_chart(
                    accidents_data,
                    "past_accidents",
                    "Dampak Kecelakaan Masa Lalu terhadap Klaim",
                    "Jumlah Kecelakaan Masa Lalu",
                )
                st.plotly_chart(fig_accidents_impact, use_container_width=True)
            else:
                st.info("Tidak ada data kecelakaan masa lalu.")

        with bcol2:
            violations_data = (
                filtered.groupby("speeding_violations")
                .agg(
                    claims=("outcome", "sum"),
                    claim_rate=("outcome", "mean"),
                )
                .reset_index()
            )
            if not violations_data.empty:
                fig_violations_impact = create_impact_chart(
                    violations_data,
                    "speeding_violations",
                    "Dampak Pelanggaran Lalu Lintas terhadap Klaim",
                    "Jumlah Pelanggaran",
                )
                st.plotly_chart(fig_violations_impact, use_container_width=True)
            else:
                st.info("Tidak ada data pelanggaran kecepatan.")
    else:
        st.info("Tidak ada data untuk saat ini.")

    st.markdown("---")

    # ---- DATA TABLE ----
    with st.expander("Lihat Data", expanded=False):
        if not filtered.empty:
            display_cols = [
                "claim_id",
                "c",
                "gender",
                "age",
                "education",
                "income",
                "vehicle_type",
                "vehicle_year",
                "state",
                "credit_score",
                "past_accidents",
                "speeding_violations",
                "duis",
                "outcome",
            ]
            available_cols = [col for col in display_cols if col in filtered.columns]

            if available_cols:
                # ====== URUTKAN DULU DATA DW SESUAI KUNCI ======
                # prioritas: customer_id lalu claim_id (kalau ada)
                sort_cols = [c for c in ["customer_id", "claim_id"] if c in filtered.columns]

                if sort_cols:
                    data_sorted = filtered.sort_values(sort_cols)
                else:
                    data_sorted = filtered.copy()

                # 1) TAMPILKAN HANYA 100 BARIS PERTAMA DI LAYAR
                sample_df = data_sorted[available_cols].head(100).reset_index(drop=True)

                # Jadikan index 1, 2, 3, ... dan beri nama "No"
                sample_df.index = sample_df.index + 1
                sample_df.index.name = "No"

                st.dataframe(
                    sample_df,
                    use_container_width=True,
                    height=400,
                )

                # 2) DATA UNTUK DI-DOWNLOAD = SELURUH HASIL FILTER + URUTAN YANG SAMA
                download_df = data_sorted[available_cols].copy()
                csv = download_df.to_csv(index=False).encode("utf-8")

                if st.download_button(
                    "Download Data (CSV)",
                    data=csv,
                    file_name="claims_filtered_dw.csv",
                    mime="text/csv",
                    key="btn_download_raw_data",
                ):
                    log_event(
                        event_type="click",
                        page_name="Asuransi Mobil",
                        element_id="btn_download_raw_data",
                        funnel_stage="raw_data",
                    )
            else:
                st.info("Tidak ada kolom yang tersedia untuk ditampilkan.")
        else:
            st.info("Tidak ada data untuk saat ini.")


# ============ PAGE 2: USER BEHAVIOR ============

else:
    ses_df, ev_df = load_realtime_behavior()

    st.title("Dashboard Perilaku Pengguna dan UX (Realtime)")

    # --------- AUTO REFRESH SETIAP 5 DETIK ---------
    st_autorefresh(interval=5000, key="auto_refresh_user_behavior")
    # -----------------------------------------------

    if ses_df.empty and ev_df.empty:
        st.warning(
            "Belum ada data pada realtime.session_metrics maupun realtime.user_events. "
            "Coba gunakan dashboard beberapa kali lalu buka halaman ini lagi."
        )
    else:
        st.sidebar.subheader("Filter Waktu Realtime")

        # ========== DEFAULT: ALL HISTORY (SES + EVENT) ==========        
        times_min = []
        times_max = []

        if not ses_df.empty:
            times_min.append(ses_df["start_time"].min())
            times_max.append(ses_df["start_time"].max())
        if not ev_df.empty:
            times_min.append(ev_df["created_at"].min())
            times_max.append(ev_df["created_at"].max())

        min_time = min(times_min)
        max_time = max(times_max)

        default_min = min_time.to_pydatetime()
        default_max = max_time.to_pydatetime()
        default_range = (default_min, default_max)

        slider_key = "filter_time_range"

        # Kalau MASIH AUTO, paksa slider pakai default_range terbaru
        if not st.session_state["time_range_manual"]:
            if slider_key in st.session_state:
                del st.session_state[slider_key]

        # Slider rentang waktu
        time_range = st.sidebar.slider(
            "Rentang waktu",
            min_value=default_min,
            max_value=default_max,
            value=default_range,
            format="YYYY-MM-DD HH:mm",
            key=slider_key,
        )

        # Auto/manual:
        if time_range != default_range:
            st.session_state["time_range_manual"] = True
        else:
            st.session_state["time_range_manual"] = False

        # Filter sesi & event berdasarkan slider
        if not ses_df.empty:
            ses_f = ses_df[
                (ses_df["start_time"] >= time_range[0])
                & (ses_df["start_time"] <= time_range[1])
            ]
        else:
            ses_f = ses_df.copy()

        if not ev_df.empty:
            ev_f = ev_df[
                (ev_df["created_at"] >= time_range[0])
                & (ev_df["created_at"] <= time_range[1])
            ]
        else:
            ev_f = ev_df.copy()

        st.caption(
            f"Menampilkan data dari {time_range[0].strftime('%Y-%m-%d %H:%M:%S')} "
            f"sampai {time_range[1].strftime('%Y-%m-%d %H:%M:%S')} (UTC)"
        )

        # ---- USER BEHAVIOR METRICS ----
        st.subheader("Metrik Perilaku Pengguna")

        total_sessions = len(ses_f) if not ses_f.empty else 0

        if total_sessions > 0:
            # Sesi yang benar-benar berinteraksi (bukan bounce)
            engaged_ses = ses_f[ses_f["bounced"] == False]

            # Dwell time = rata-rata lama interaksi (detik),
            # dari buka dashboard sampai interaksi terakhir
            if not engaged_ses.empty and "dwell_time_engaged_sec" in engaged_ses.columns:
                avg_dwell = engaged_ses["dwell_time_engaged_sec"].mean()
            else:
                avg_dwell = 0.0

            # Bounce rate = proporsi sesi yang tidak punya interaksi
            bounce_rate = ses_f["bounced"].mean()
        else:
            avg_dwell = 0.0
            bounce_rate = 0.0

        if not ev_f.empty:
            total_events = len(ev_f)
            total_clicks = int(ev_f[ev_f["event_type"] != "page_view"].shape[0])

            error_events = ev_f[ev_f["event_type"] == "error"]
            error_rate = len(error_events) / total_events if total_events else 0.0
        else:
            total_events = 0
            total_clicks = 0
            error_events = ev_f
            error_rate = 0.0

        # hanya dua metric: Dwell Time & Bounce Rate
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Dwell Time (detik, sesi yang interaksi)", round(avg_dwell, 1))
        with m2:
            st.metric("Bounce Rate", f"{bounce_rate*100:.1f}%")

        # ---- USABILITY SCORE (0–100) ----
        st.subheader("Skor Kegunaan (0–100)")

        if total_sessions == 0 or total_events == 0:
            st.info(
                "Belum cukup data untuk menghitung skor kegunaan "
                "pada rentang waktu ini."
            )
        else:
            # Normalisasi ke 0–1 (1 = terbaik)
            # Asumsi: 60 detik dwell time atau lebih dianggap sudah optimal
            norm_dwell = min(avg_dwell / 60.0, 1.0)

            # Bounce rate & error rate sudah dalam 0–1, dibalik agar 1 = terbaik
            norm_bounce = 1.0 - max(0.0, min(bounce_rate, 1.0))
            norm_error = 1.0 - max(0.0, min(error_rate, 1.0))

            # Rata-rata tiga komponen
            raw_score = (norm_dwell + norm_bounce + norm_error) / 3.0

            # Skala 0–100
            usability_score = raw_score * 100.0

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Usability Score (0–100)", f"{usability_score:.1f}")
            with c2:
                st.write(
                    """
                    Skor ini dihitung dari kombinasi:
                    - Dwell Time (rata-rata waktu interaksi)
                    - Bounce Rate (semakin rendah semakin baik)
                    - Error Rate (semakin rendah semakin baik)
                    """
                )

        # ---- ERROR RATE ----
        st.subheader("Error Rate dan Interaksi Gagal")

        ec1, ec2 = st.columns(2)
        with ec1:
            st.metric("Error Rate", f"{error_rate*100:.2f}%")
        with ec2:
            if not error_events.empty and "error_code" in error_events.columns:
                by_error = (
                    error_events["error_code"]
                    .value_counts()
                    .head(5)
                    .reset_index()
                )
                by_error.columns = ["error_code", "count"]
                fig_error = px.bar(
                    by_error,
                    x="error_code",
                    y="count",
                    color_discrete_sequence=[DANGER_COLOR],
                )
                fig_error.update_layout(
                    title=dict(
                        text="Kode Error Teratas",
                        font=dict(size=15, color=PRIMARY_COLOR),
                    ),
                    height=250,
                    paper_bgcolor="rgba(0,0,0,0)",
                    transition={"duration": 600, "easing": "cubic-in-out"},
                )
                st.plotly_chart(fig_error, use_container_width=True)

        # ---- FUNNEL ANALYSIS: per section analisis dashboard asuransi ----
        st.subheader("Analisis Funnel")

        funnel_order = [
            "landing",            # Stage 1: masuk ke dashboard
            "segment_filtering",  # Stage 2: pakai filter
            "risk_exploration",   # Stage 3: fokus risiko tinggi
            "raw_data",           # Stage 4: download data
        ]

        if "funnel_stage" in ev_f.columns:
            mask = (
                ev_f["funnel_stage"].isin(funnel_order)
                & (ev_f["page_name"] == "Asuransi Mobil")
            )
            funnel_counts = (
                ev_f[mask]
                .groupby("funnel_stage")["session_id"]
                .nunique()
                .reset_index()
                .rename(columns={"session_id": "sessions"})
            )

            label_map = {
                "landing": "Landing",
                "segment_filtering": "Filter",
                "risk_exploration": "Eksplorasi Detail Risiko",
                "raw_data": "Download Data",
            }
            funnel_counts["Alur Funnel"] = funnel_counts["funnel_stage"].map(
                label_map
            ).fillna(funnel_counts["funnel_stage"])

            funnel_counts = funnel_counts.sort_values(
                "sessions", ascending=False
            )

            fig_funnel_rt = create_funnel_chart(
                funnel_counts,
                "sessions",
                "Alur Funnel",
                "Analisis funnel alur pengguna dashboard asuransi",
            )
            st.plotly_chart(fig_funnel_rt, use_container_width=True)
        else:
            st.info("Kolom funnel_stage belum tersedia pada tabel event.")

        # ---- UI/UX PERFORMANCE TREND ----
        st.subheader("Tren Performa UI/UX")

        if not ses_f.empty:
            perf_trend = (
                ses_f.assign(date=lambda d: d["start_time"].dt.floor("5min"))
                .groupby("date")
                .agg(
                    dwell_time_avg=("dwell_time_engaged_sec", "mean"),
                    bounce_rate=("bounced", "mean"),
                    errors=("total_errors", "mean"),
                )
                .reset_index()
            )
        else:
            perf_trend = pd.DataFrame()

        if not perf_trend.empty:
            fig_perf = go.Figure()
            fig_perf.add_trace(
                go.Scatter(
                    x=perf_trend["date"],
                    y=perf_trend["dwell_time_avg"],
                    name="Waktu Interaksi (detik)",
                    mode="lines+markers",
                    line=dict(color=PRIMARY_COLOR, width=2),
                    marker=dict(size=6),
                )
            )
            fig_perf.add_trace(
                go.Scatter(
                    x=perf_trend["date"],
                    y=perf_trend["bounce_rate"] * 100,
                    name="Bounce Rate %",
                    mode="lines+markers",
                    line=dict(color=ACCENT_COLOR, width=2),
                    marker=dict(size=6),
                )
            )
            fig_perf.add_trace(
                go.Scatter(
                    x=perf_trend["date"],
                    y=perf_trend["errors"],
                    name="Error",
                    mode="lines+markers",
                    line=dict(color=DANGER_COLOR, width=2),
                    marker=dict(size=6),
                )
            )
            fig_perf.update_layout(
                title=dict(
                    text="Performa UI/UX Seiring Waktu",
                    font=dict(size=15, color=PRIMARY_COLOR),
                ),
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#E0E0E0"),
                transition={"duration": 800, "easing": "cubic-in-out"},
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Belum ada data sesi pada rentang waktu ini.")

