"""Apple Human Interface Guidelines theme for Streamlit.

Injects a comprehensive CSS override that brings Apple HIG design
principles to the Streamlit app: SF Pro typography, 8pt grid spacing,
semantic colors, rounded corners, subtle shadows, and dark mode support.

Usage:
    from display.theme import inject_apple_hig_css
    inject_apple_hig_css()   # call once in app.py after set_page_config
"""

from __future__ import annotations

import streamlit as st

_APPLE_HIG_CSS = """
<style>
/* ============================================================
   Apple HIG Theme for Streamlit
   ============================================================ */

/* ── CSS Variables ── */

:root {
    /* Backgrounds */
    --hig-bg-primary: #F2F2F7;
    --hig-bg-secondary: #FFFFFF;
    --hig-bg-tertiary: #F9F9FB;

    /* Accent */
    --hig-accent: #007AFF;
    --hig-accent-hover: #0066D6;
    --hig-accent-subtle: rgba(0, 122, 255, 0.08);

    /* Semantic */
    --hig-red: #FF3B30;
    --hig-orange: #FF9500;
    --hig-green: #34C759;
    --hig-purple: #AF52DE;
    --hig-teal: #5AC8FA;
    --hig-indigo: #5856D6;

    /* Text */
    --hig-text-primary: #1C1C1E;
    --hig-text-secondary: #3C3C43;
    --hig-text-tertiary: #8E8E93;

    /* Separators & Fills */
    --hig-separator: rgba(60, 60, 67, 0.12);
    --hig-fill-tertiary: rgba(120, 120, 128, 0.12);

    /* Shadows */
    --hig-shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
    --hig-shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04);

    /* Radii */
    --hig-radius-sm: 8px;
    --hig-radius-md: 12px;
    --hig-radius-lg: 16px;
    --hig-radius-full: 9999px;

    /* Transitions */
    --hig-ease: cubic-bezier(0.25, 0.1, 0.25, 1.0);
}

/* ── Dark Mode ── */

@media (prefers-color-scheme: dark) {
    :root {
        --hig-bg-primary: #0A1628;
        --hig-bg-secondary: #132240;
        --hig-bg-tertiary: #1A2D52;
        --hig-accent: #0A84FF;
        --hig-accent-hover: #409CFF;
        --hig-accent-subtle: rgba(10, 132, 255, 0.12);
        --hig-red: #FF453A;
        --hig-orange: #FF9F0A;
        --hig-green: #30D158;
        --hig-purple: #BF5AF2;
        --hig-teal: #64D2FF;
        --hig-indigo: #5E5CE6;
        --hig-text-primary: #FFFFFF;
        --hig-text-secondary: rgba(255, 255, 255, 0.85);
        --hig-text-tertiary: rgba(255, 255, 255, 0.65);
        --hig-separator: rgba(84, 84, 88, 0.65);
        --hig-fill-tertiary: rgba(120, 120, 128, 0.24);
        --hig-shadow-sm: 0 1px 4px rgba(0, 0, 0, 0.3);
        --hig-shadow-md: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
}

/* ── Reduced Motion ── */

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* ── Global Typography ── */

html, body, [class*="st-"]:not([class*="material"]):not([data-testid="stIconMaterial"]) {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    font-size: 1.0625rem !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Preserve Material Symbols font for icons */
.material-symbols-rounded,
[data-testid="stIconMaterial"],
span[class*="material"] {
    font-family: "Material Symbols Rounded" !important;
}

/* ── Main container background ── */

.stApp {
    background-color: var(--hig-bg-primary) !important;
}

/* Remove the default Streamlit header line */
header[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
}

/* ── Sidebar ── */

section[data-testid="stSidebar"] {
    background-color: var(--hig-bg-secondary) !important;
    border-right: 1px solid var(--hig-separator) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

/* ── Headings ── */

h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--hig-text-primary) !important;
    font-size: 2rem !important;
    line-height: 1.2 !important;
}

h2 {
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    color: var(--hig-text-primary) !important;
    font-size: 1.5rem !important;
    line-height: 1.25 !important;
}

h3 {
    font-weight: 600 !important;
    color: var(--hig-text-primary) !important;
    font-size: 1.25rem !important;
    line-height: 1.3 !important;
}

h4 {
    font-weight: 600 !important;
    color: var(--hig-text-primary) !important;
    font-size: 1.1rem !important;
}

/* Subtle caption styling */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--hig-text-tertiary) !important;
    font-size: 0.8125rem !important;
}

/* ── Buttons ── */

/* Primary buttons */
button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: var(--hig-accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--hig-radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.9375rem !important;
    min-height: 44px !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 150ms var(--hig-ease) !important;
    letter-spacing: -0.01em !important;
}

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background-color: var(--hig-accent-hover) !important;
    transform: scale(1.01);
}

button[kind="primary"]:active,
.stButton > button[kind="primary"]:active {
    transform: scale(0.98);
}

/* Secondary buttons */
button[kind="secondary"],
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background-color: var(--hig-fill-tertiary) !important;
    color: var(--hig-accent) !important;
    border: none !important;
    border-radius: var(--hig-radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.9375rem !important;
    min-height: 44px !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 150ms var(--hig-ease) !important;
}

button[kind="secondary"]:hover,
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    background-color: var(--hig-accent-subtle) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: var(--hig-fill-tertiary) !important;
    color: var(--hig-accent) !important;
    border: none !important;
    border-radius: var(--hig-radius-md) !important;
    font-weight: 600 !important;
    min-height: 44px !important;
    transition: all 150ms var(--hig-ease) !important;
}

.stDownloadButton > button:hover {
    background-color: var(--hig-accent-subtle) !important;
}

/* ── Inputs ── */

/* Text inputs, number inputs, date inputs */
input[type="text"],
input[type="password"],
input[type="number"],
input[type="email"],
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border: 1px solid rgba(60, 60, 67, 0.18) !important;
    border-radius: var(--hig-radius-sm) !important;
    font-size: 1rem !important;
    min-height: 44px !important;
    padding: 0.5rem 0.875rem !important;
    transition: border-color 150ms var(--hig-ease),
                box-shadow 150ms var(--hig-ease) !important;
}

input[type="text"]:focus,
input[type="password"]:focus,
input[type="number"]:focus,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--hig-accent) !important;
    box-shadow: 0 0 0 3px var(--hig-accent-subtle) !important;
}

/* Text areas */
.stTextArea textarea {
    border: 1px solid rgba(60, 60, 67, 0.18) !important;
    border-radius: var(--hig-radius-sm) !important;
    font-size: 1rem !important;
    padding: 0.75rem !important;
    line-height: 1.5 !important;
    transition: border-color 150ms var(--hig-ease),
                box-shadow 150ms var(--hig-ease) !important;
}

.stTextArea textarea:focus {
    border-color: var(--hig-accent) !important;
    box-shadow: 0 0 0 3px var(--hig-accent-subtle) !important;
}

/* Select boxes */
.stSelectbox > div > div {
    border-radius: var(--hig-radius-sm) !important;
    min-height: 44px !important;
}

/* ── Expanders ── */

details[data-testid="stExpander"] {
    border: none !important;
    border-radius: var(--hig-radius-md) !important;
    background-color: var(--hig-bg-secondary) !important;
    box-shadow: var(--hig-shadow-sm) !important;
    margin-bottom: 0.75rem !important;
    overflow: hidden;
}

details[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.875rem 1rem !important;
    border-radius: var(--hig-radius-md) !important;
    transition: background-color 150ms var(--hig-ease) !important;
}

details[data-testid="stExpander"] summary:hover {
    background-color: var(--hig-fill-tertiary) !important;
}

details[data-testid="stExpander"] > div {
    padding: 0 1rem 1rem !important;
}


/* ── Tabs ── */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    border-bottom: 1px solid var(--hig-separator) !important;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.9375rem !important;
    border-radius: var(--hig-radius-sm) var(--hig-radius-sm) 0 0 !important;
    padding: 0.625rem 1.25rem !important;
    min-height: 44px !important;
    color: var(--hig-text-secondary) !important;
    transition: all 150ms var(--hig-ease) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: var(--hig-fill-tertiary) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--hig-accent) !important;
}

/* ── Metrics ── */

[data-testid="stMetric"] {
    background: var(--hig-bg-secondary) !important;
    border-radius: var(--hig-radius-md) !important;
    padding: 1rem 1.25rem !important;
    box-shadow: var(--hig-shadow-sm) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.8125rem !important;
    font-weight: 600 !important;
    color: var(--hig-text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: var(--hig-text-primary) !important;
    letter-spacing: -0.02em !important;
}

/* ── Progress bar ── */

.stProgress > div > div {
    border-radius: var(--hig-radius-full) !important;
    height: 8px !important;
}

.stProgress > div > div > div {
    background-color: var(--hig-accent) !important;
    border-radius: var(--hig-radius-full) !important;
}

/* ── Alert banners (info, success, warning, error) ── */

.stAlert {
    border-radius: var(--hig-radius-sm) !important;
    border-left-width: 4px !important;
    font-size: 0.9375rem !important;
}

/* Success alerts */
[data-testid="stAlert"][data-baseweb*="positive"],
div[role="alert"].st-success {
    border-left-color: var(--hig-green) !important;
}

/* Warning alerts */
[data-testid="stAlert"][data-baseweb*="warning"] {
    border-left-color: var(--hig-orange) !important;
}

/* Error alerts */
[data-testid="stAlert"][data-baseweb*="negative"] {
    border-left-color: var(--hig-red) !important;
}

/* Info alerts */
[data-testid="stAlert"][data-baseweb*="info"] {
    border-left-color: var(--hig-accent) !important;
}

/* ── DataFrames ── */

.stDataFrame {
    border-radius: var(--hig-radius-md) !important;
    overflow: hidden;
    box-shadow: var(--hig-shadow-sm) !important;
}

/* ── Dividers ── */

hr {
    border: none !important;
    border-top: 1px solid var(--hig-separator) !important;
    margin: 1.5rem 0 !important;
}

/* ── Radio buttons ── */

.stRadio > div {
    gap: 0.5rem !important;
}

.stRadio [role="radiogroup"] label {
    font-size: 0.9375rem !important;
    min-height: 36px !important;
    display: flex !important;
    align-items: center !important;
    padding: 0.25rem 0.75rem !important;
    border-radius: var(--hig-radius-sm) !important;
    transition: background-color 150ms var(--hig-ease) !important;
}

.stRadio [role="radiogroup"] label:hover {
    background-color: var(--hig-fill-tertiary) !important;
}

/* ── Sliders ── */

.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--hig-accent) !important;
    width: 28px !important;
    height: 28px !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
}

.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    font-weight: 700 !important;
    color: var(--hig-accent) !important;
}

/* ── Select slider (used for PDSQI-9 scores) ── */

[data-testid="stThumbValue"] {
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* ── Blockquotes (used for claim text) ── */

blockquote {
    border-left: 3px solid var(--hig-accent) !important;
    padding: 0.5rem 1rem !important;
    margin: 0.75rem 0 !important;
    background: var(--hig-accent-subtle) !important;
    border-radius: 0 var(--hig-radius-sm) var(--hig-radius-sm) 0 !important;
    color: var(--hig-text-primary) !important;
    font-size: 0.9375rem !important;
    line-height: 1.55 !important;
}

/* ── Container (used for scrollable review form) ── */

[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--hig-radius-lg) !important;
    border: 1px solid var(--hig-separator) !important;
}

/* ── Form submit button ── */

.stFormSubmitButton > button {
    background-color: var(--hig-accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--hig-radius-md) !important;
    font-weight: 600 !important;
    min-height: 44px !important;
    transition: all 150ms var(--hig-ease) !important;
}

.stFormSubmitButton > button:hover {
    background-color: var(--hig-accent-hover) !important;
}

/* ── Balloons / confetti — keep default ── */

/* ── Scrollbar styling ── */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.15);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.25);
}

@media (prefers-color-scheme: dark) {
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
    }
}

/* ── Monospace elements (code, trace data) ── */

code, pre, .stCode {
    font-family: "SF Mono", ui-monospace, "Fira Code", "Cascadia Code",
                 "Consolas", monospace !important;
    border-radius: var(--hig-radius-sm) !important;
}

/* ── Links ── */

a {
    color: var(--hig-accent) !important;
    text-decoration: none !important;
    transition: opacity 150ms var(--hig-ease) !important;
}

a:hover {
    opacity: 0.8;
}

/* ── Focus visible outlines (accessibility) ── */

*:focus-visible {
    outline: 3px solid var(--hig-accent) !important;
    outline-offset: 2px !important;
}

/* ── Citation superscript markers ── */
/* Non-chromatic signal: decision_critical = bold blue; unverified = bold amber.
   Color is reinforcement, not the sole channel (tier also encoded in weight
   and the ⚠ glyph in the tooltip). See frontend/src/App.css for JSX twin. */

.icp-cite {
    position: relative;
    font-weight: 600;
    font-size: 0.85em;
    vertical-align: super;
    line-height: 0;
    margin-left: 0.1em;
    cursor: help;
    text-decoration: none;
    outline: none;   /* hide default focus ring; we highlight via tooltip */
}

/* Instant hover/focus tooltip — native ``title=`` has a ~1.5s browser delay
   that's too slow for a clinical review workflow. attr(title) reuses the
   existing attribute so the renderer code stays the tooltip's single source
   of truth. Focus state (from tabindex=0 on the <sup>) makes the tooltip
   "stick" when clicked so reviewers can read longer tooltips without having
   to keep the mouse perfectly still. */
.icp-cite::after {
    content: attr(title);
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(28, 28, 30, 0.96);
    color: #fff;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    line-height: 1.4;
    white-space: nowrap;
    max-width: 360px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.1s ease-in;
    z-index: 9999;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

.icp-cite:hover::after,
.icp-cite:focus::after {
    opacity: 1;
}

.icp-cite--decision_critical {
    color: #1e40af;   /* deep blue — "anchored, worth verifying" */
}

.icp-cite--unverified {
    color: #b45309;   /* amber — provenance check failed */
}

@media (prefers-color-scheme: dark) {
    .icp-cite--decision_critical { color: #93c5fd; }
    .icp-cite--unverified        { color: #fbbf24; }
}

</style>
"""


def inject_apple_hig_css() -> None:
    """Inject Apple HIG CSS overrides into the current Streamlit page."""
    st.markdown(_APPLE_HIG_CSS, unsafe_allow_html=True)
