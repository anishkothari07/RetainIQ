"""
RetainIQ - UI Utilities and Design System
Provides custom CSS, styling helpers, Plotly charting templates,
KPI card components, and badge renderers for a premium B2B SaaS look and feel.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math
from typing import Optional, Dict, Any, List

# Primary Color Palette
PRIMARY = "#4F46E5"        # Electric Indigo
PRIMARY_DARK = "#3730A3"   # Deep Indigo
PRIMARY_LIGHT = "#EEF2FF"  # Indigo Tint
TEXT_PRIMARY = "#0F172A"   # Slate 900
TEXT_SECONDARY = "#475569" # Slate 600
TEXT_MUTED = "#94A3B8"     # Slate 400
BG_WORKSPACE = "#F8FAFC"   # Slate 50
CARD_BG = "#FFFFFF"        # Pure White
BORDER_COLOR = "#E2E8F0"   # Slate 200

# Semantic Colors
RISK_HIGH = "#EF4444"      # Coral Red
RISK_MEDIUM = "#F59E0B"    # Amber
RISK_LOW = "#10B981"       # Emerald

# Segment Colors
SEGMENT_COLORS = {
    "Champions": "#10B981",          # Emerald
    "Loyal Customers": "#3B82F6",    # Blue
    "Potential Customers": "#8B5CF6",# Purple
    "Recent Customers": "#06B6D4",   # Cyan
    "At Risk": "#F59E0B",            # Amber
    "Lost Customers": "#EF4444",     # Rose
}

def inject_custom_css():
    """Injects high-end enterprise B2B SaaS CSS into the Streamlit app."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Global Typography and Reset */
    html, body, [class*="css"], [class*="st-"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {TEXT_PRIMARY};
    }}

    /* Preserve Material Icons font so Streamlit UI icons and sidebar toggles do not render as raw text */
    span[data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] *,
    button[data-testid="stSidebarCollapseButton"] span,
    button[data-testid="stBaseButton-header"] span,
    .material-symbols-rounded,
    [class*="material-symbols"] {{
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
    }}

    /* Main Workspace Background */
    .stApp {{
        background-color: {BG_WORKSPACE};
    }}

    /* Top Header Bar */
    header[data-testid="stHeader"] {{
        background-color: rgba(248, 250, 252, 0.85);
        backdrop-filter: blur(8px);
    }}

    /* Sidebar Styling - Dark Navy Charcoal Enterprise Theme */
    [data-testid="stSidebar"] {{
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }}

    [data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {{
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    [data-testid="stSidebarNav"] {{
        padding-top: 1rem;
    }}

    [data-testid="stSidebarNav"] a {{
        background-color: transparent !important;
        border-radius: 8px !important;
        margin: 2px 8px !important;
        padding: 8px 12px !important;
        transition: all 0.2s ease;
    }}

    [data-testid="stSidebarNav"] a:hover {{
        background-color: #1E293B !important;
    }}

    [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background-color: #1E293B !important;
        border-left: 3px solid #6366F1 !important;
    }}

    [data-testid="stSidebarNav"] a[aria-current="page"] span {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    /* Cards and Containers */
    .retainiq-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.03), 0 1px 2px -1px rgba(15, 23, 42, 0.03);
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .retainiq-card:hover {{
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
    }}

    /* Streamlit Native Bordered Containers (Cards) */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.03), 0 1px 2px -1px rgba(15, 23, 42, 0.03) !important;
        transition: box-shadow 0.15s ease !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {{
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04) !important;
    }}

    /* Global Inline Code and Monospace Elements - High Contrast Light Theme */
    code,
    pre code,
    .stMarkdown code,
    [data-testid="stMarkdownContainer"] code,
    p code,
    li code,
    span code,
    div code {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
        background-color: #EEF2FF !important; /* Soft indigo tint */
        color: #3730A3 !important;            /* High contrast deep indigo */
        padding: 0.18rem 0.45rem !important;
        border-radius: 6px !important;
        font-size: 0.86em !important;
        border: 1px solid #C7D2FE !important;
        font-weight: 600 !important;
        display: inline-block !important;
        line-height: 1.25 !important;
    }}

    /* Sidebar Code Overrides */
    [data-testid="stSidebar"] code,
    [data-testid="stSidebar"] pre code,
    [data-testid="stSidebar"] .stMarkdown code,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code {{
        background-color: #1E293B !important;
        color: #A5B4FC !important;
        border: 1px solid #334155 !important;
    }}

    /* KPI Metrics Card */
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        position: relative;
        overflow: hidden;
    }}

    .metric-card-accent-indigo {{
        border-top: 3px solid #4F46E5;
    }}
    .metric-card-accent-emerald {{
        border-top: 3px solid #10B981;
    }}
    .metric-card-accent-amber {{
        border-top: 3px solid #F59E0B;
    }}
    .metric-card-accent-rose {{
        border-top: 3px solid #EF4444;
    }}

    .metric-label {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {TEXT_SECONDARY};
        margin-bottom: 0.35rem;
    }}

    .metric-value {{
        font-size: 1.75rem;
        font-weight: 800;
        color: {TEXT_PRIMARY};
        line-height: 1.2;
        letter-spacing: -0.02em;
    }}

    .metric-subtext {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-top: 0.35rem;
        font-weight: 500;
    }}

    /* Notice and Warning Banners */
    .retainiq-alert {{
        border-radius: 10px;
        padding: 0.85rem 1.15rem;
        margin: 0.75rem 0 1.25rem 0;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        line-height: 1.45;
    }}

    .alert-info {{
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
    }}

    .alert-warning {{
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
    }}

    .alert-historical {{
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 4px solid #64748B;
        color: #334155;
    }}

    /* Status Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}

    .badge-high {{
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }}

    .badge-medium {{
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }}

    .badge-low {{
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {PRIMARY};
        color: #FFFFFF !important;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        box-shadow: 0 1px 2px rgba(79, 70, 229, 0.2);
        transition: all 0.15s ease;
    }}

    .stButton > button:hover {{
        background-color: {PRIMARY_DARK};
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.3);
        transform: translateY(-1px);
    }}

    .stDownloadButton > button {{
        background-color: #FFFFFF;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }}

    .stDownloadButton > button:hover {{
        background-color: #F1F5F9;
        border-color: #CBD5E1;
    }}

    /* Streamlit Tables & DataFrames */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        border-bottom: 1px solid {BORDER_COLOR};
    }}

    .stTabs [data-baseweb="tab"] {{
        padding: 0.6rem 1.25rem;
        font-weight: 600;
        font-size: 0.88rem;
        color: {TEXT_SECONDARY};
        border-radius: 6px 6px 0 0;
    }}

    .stTabs [aria-selected="true"] {{
        color: {PRIMARY} !important;
        border-bottom: 2px solid {PRIMARY} !important;
    }}

    /* Hide standard Streamlit footer and hamburger menu */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar_brand():
    """Renders the RetainIQ SaaS Brand Identity in the Streamlit Sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid #1E293B; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1rem; color: #FFFFFF;">
                        R
                    </div>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1;">
                            Retain<span style="color: #818CF8;">IQ</span>
                        </div>
                        <div style="font-size: 0.68rem; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em; text-transform: uppercase; margin-top: 2px;">
                            Enterprise Intelligence
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_header(title: str, subtitle: str, badge: Optional[str] = None):
    """Renders a clean, executive-level header for dashboard pages."""
    badge_html = f'<span style="background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 9999px; margin-left: 0.75rem; vertical-align: middle;">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem; padding-bottom: 0.75rem; border-bottom: 1px solid {BORDER_COLOR};">
            <h1 style="font-size: 1.85rem; font-weight: 800; color: {TEXT_PRIMARY}; margin: 0; letter-spacing: -0.03em; display: flex; align-items: center;">
                {title} {badge_html}
            </h1>
            <p style="font-size: 0.92rem; color: {TEXT_SECONDARY}; margin: 0.35rem 0 0 0; font-weight: 500;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    accent: str = "indigo"
):
    """Renders an enterprise KPI Card."""
    accent_class = f"metric-card-accent-{accent}"
    st.markdown(
        f"""
        <div class="metric-card {accent_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert(text: str, alert_type: str = "info", icon: str = "ℹ️"):
    """Renders a custom enterprise alert box."""
    css_class = f"alert-{alert_type}"
    st.markdown(
        f"""
        <div class="retainiq-alert {css_class}">
            <span style="font-size: 1.1rem;">{icon}</span>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_category: str) -> str:
    """Returns HTML for a risk category pill."""
    cat_lower = str(risk_category).lower()
    if cat_lower == "high":
        return '<span class="badge badge-high">High Risk</span>'
    elif cat_lower == "medium":
        return '<span class="badge badge-medium">Medium Risk</span>'
    else:
        return '<span class="badge badge-low">Low Risk</span>'


def render_segment_badge(segment: str) -> str:
    """Returns HTML for an RFM segment pill with tailored colors."""
    color = SEGMENT_COLORS.get(segment, "#64748B")
    return f'<span style="background-color: {color}1A; color: {color}; border: 1px solid {color}4D; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">{segment}</span>'


def get_plotly_theme(**kwargs) -> Dict[str, Any]:
    """Returns a tailored Plotly layout configuration for clean B2B SaaS charts."""
    theme = dict(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(size=11, color=TEXT_SECONDARY),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            showline=False,
            tickfont=dict(size=11, color=TEXT_SECONDARY),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=TEXT_SECONDARY),
        ),
        hoverlabel=dict(
            bgcolor=TEXT_PRIMARY,
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color="#FFFFFF",
        ),
    )
    for k, v in kwargs.items():
        if k in theme and isinstance(theme[k], dict) and isinstance(v, dict):
            theme[k] = {**theme[k], **v}
        else:
            theme[k] = v
    return theme


def format_currency(amount: float) -> str:
    """Formats a number as British Pounds currency string."""
    if amount >= 1_000_000:
        return f"£{amount / 1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"£{amount / 1_000:.1f}k"
    else:
        return f"£{amount:,.2f}"


def format_number(val: int | float) -> str:
    """Formats an integer or float with thousands separators."""
    return f"{val:,.0f}" if isinstance(val, (int, float)) and val == int(val) else f"{val:,.2f}"


def format_probability_pct(prob: Optional[float]) -> str:
    """
    Formats a churn probability as a percentage string with at least 2 decimal places.
    Prevents rounding small non-zero values to 0.00% and clearly distinguishes missing values.
    """
    if prob is None or pd.isna(prob):
        return "N/A"
    try:
        p = float(prob)
        if p == 0.0:
            return "0.00%"
        pct = p * 100.0
        if 0.0 < pct < 0.01:
            return "<0.01%"
        return f"{pct:.2f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_probability_decimal(prob: Optional[float], decimals: int = 4) -> str:
    """
    Formats decimal probability with at least 4 decimal places, handling edge cases.
    """
    if prob is None or pd.isna(prob):
        return "N/A"
    try:
        p = float(prob)
        if p == 0.0:
            return "0.0000"
        cutoff = 10 ** (-decimals)
        if 0.0 < p < cutoff:
            return f"<{cutoff:.{decimals}f}"
        return f"{p:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"
