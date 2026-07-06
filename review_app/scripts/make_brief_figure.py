"""Render a case's brief + source panel to a standalone, print-quality HTML file.

This is the manuscript-figure replacement for screenshotting the live review
app. It reuses the REAL renderers (``display.note_renderer.render_note`` and
``display.source_renderer.render_source``) by shimming Streamlit so their HTML
emissions are captured into a buffer instead of being painted to a browser.
The captured content is wrapped in a self-contained print stylesheet (no
Streamlit-coupled selectors), so the text stays fully vector and the font size
is controllable for print.

Workflow:
    1. Run this script -> writes an .html file.
    2. Open the .html in Chrome.
    3. File -> Print -> Destination "Save as PDF" -> Save.
       The PDF has vector text (sharp at any zoom, any DPI). Drop it straight
       into the manuscript, or convert to SVG/EPS as the journal requires.

PHI: source bundles are unredacted by design. For a PUBLISHED figure you MUST
use ``--redact`` (Philter) or a synthetic/consented case. The script refuses to
omit redaction silently for a non-demo case unless you pass ``--no-redact``.

Usage (from the review_app/ directory so imports resolve):
    cd review_app
    python scripts/make_brief_figure.py --hosp-id <HOSP_ID> --redact \\
        -o ~/Desktop/fig2_brief.html

    # brief only (no source side panel):
    python scripts/make_brief_figure.py --hosp-id <HOSP_ID> --redact --no-source

    # bigger fonts for a small column figure:
    python scripts/make_brief_figure.py --hosp-id <HOSP_ID> --redact --font-scale 1.15
"""

from __future__ import annotations

import argparse
import html as _html
import os
import re
import sys
from pathlib import Path

# Make the review_app package root importable when run from anywhere.
_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))


# ---------------------------------------------------------------------------
# Streamlit shim: capture emitted HTML into a tree of containers.
# ---------------------------------------------------------------------------


class _Container:
    """A node that collects child HTML fragments."""

    def __init__(self, kind: str, **attrs):
        self.kind = kind
        self.attrs = attrs
        self.children: list[str] = []

    def add(self, html_str: str) -> None:
        self.children.append(html_str)

    def inner(self) -> str:
        return "".join(self.children)

    def render(self) -> str:
        inner = self.inner()
        if self.kind == "root":
            return inner
        if self.kind == "expander":
            title = _md_inline(self.attrs.get("label", ""))
            return (
                '<section class="fig-card">'
                f'<div class="fig-card-header">{title}</div>'
                f'<div class="fig-card-body">{inner}</div>'
                "</section>"
            )
        if self.kind == "column":
            return f'<div class="fig-col">{inner}</div>'
        if self.kind == "columns":
            return f'<div class="fig-cols">{inner}</div>'
        return inner


class _Ctx:
    """Context manager that makes ``container`` the current emit target."""

    def __init__(self, st: "_FakeStreamlit", container: _Container):
        self._st = st
        self._container = container

    def __enter__(self):
        self._st._stack.append(self._container)
        return self

    def __exit__(self, *exc):
        self._st._stack.pop()
        return False


class _FakeStreamlit:
    """Minimal Streamlit API surface used by the two renderers."""

    def __init__(self):
        self.root = _Container("root")
        self._stack: list[_Container] = [self.root]
        self.session_state: dict = {}

    # -- emit helpers -------------------------------------------------------

    @property
    def _cur(self) -> _Container:
        return self._stack[-1]

    def _emit(self, html_str: str) -> None:
        self._cur.add(html_str)

    # -- text / markdown ----------------------------------------------------

    def markdown(self, body: str = "", unsafe_allow_html: bool = False, **kw) -> None:
        self._emit(f'<div class="fig-md">{_md_block(body)}</div>')

    def write(self, body: str = "", **kw) -> None:
        self.markdown(str(body))

    def caption(self, body: str = "", **kw) -> None:
        self._emit(f'<div class="fig-caption">{_md_inline(str(body))}</div>')

    def code(self, body: str = "", language=None, **kw) -> None:
        self._emit(f'<pre class="fig-code">{_html.escape(str(body))}</pre>')

    def divider(self, **kw) -> None:
        self._emit('<hr class="fig-divider">')

    # -- callouts -----------------------------------------------------------

    def _callout(self, body: str, kind: str) -> None:
        self._emit(
            f'<div class="fig-callout fig-callout-{kind}">{_md_block(str(body))}</div>'
        )

    def warning(self, body: str = "", **kw) -> None:
        self._callout(body, "warning")

    def error(self, body: str = "", **kw) -> None:
        self._callout(body, "error")

    def info(self, body: str = "", **kw) -> None:
        self._callout(body, "info")

    def success(self, body: str = "", **kw) -> None:
        self._callout(body, "info")

    # -- dataframe ----------------------------------------------------------

    def dataframe(self, df, **kw) -> None:
        try:
            table = df.to_html(
                index=not kw.get("hide_index", False),
                border=0,
                classes="fig-table",
                escape=True,
            )
        except Exception:
            table = f"<pre>{_html.escape(str(df))}</pre>"
        self._emit(table)

    def table(self, df, **kw) -> None:
        self.dataframe(df, **kw)

    # -- layout containers --------------------------------------------------

    def expander(self, label: str = "", expanded: bool = False, **kw) -> _Ctx:
        node = _Container("expander", label=label)
        self._cur.add(_Deferred(node))
        return _Ctx(self, node)

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        wrap = _Container("columns")
        self._cur.add(_Deferred(wrap))
        cols = []
        for _ in range(n):
            col = _Container("column")
            wrap.children.append(_Deferred(col))
            cols.append(_Ctx(self, col))
        return cols

    def container(self, **kw) -> _Ctx:
        node = _Container("root")  # transparent passthrough
        self._cur.add(_Deferred(node))
        return _Ctx(self, node)

    # -- no-ops for anything else the renderers might touch -----------------

    def __getattr__(self, name):
        def _noop(*a, **k):
            return _Ctx(self, _Container("root"))

        return _noop


class _Deferred:
    """Placeholder so nested containers render after their children fill in."""

    def __init__(self, container: _Container):
        self.container = container

    def __str__(self) -> str:  # rendered lazily during final flatten
        return self.container.render()


def _flatten(container: _Container) -> str:
    """Recursively render a container tree to an HTML string."""
    parts = []
    for child in container.children:
        if isinstance(child, _Deferred):
            parts.append(_flatten(child.container))
        else:
            parts.append(str(child))
    container.children = parts
    return container.render()


# ---------------------------------------------------------------------------
# Tiny markdown: only what the renderers actually emit.
# ---------------------------------------------------------------------------

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_ICODE = re.compile(r"`([^`]+?)`")


def _md_inline(text: str) -> str:
    """Bold + inline-code. Pass other HTML through (renderers emit trusted HTML)."""
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ICODE.sub(r"<code>\1</code>", text)
    return text


def _md_block(text: str) -> str:
    """Block-ish: handle bullet lines and Streamlit's two-space line breaks."""
    text = _md_inline(text)
    # Streamlit treats "  \n" as a hard break; renderers also use bare "\n".
    text = text.replace("  \n", "<br>").replace("\n", "<br>")
    return text


# ---------------------------------------------------------------------------
# Page assembly.
# ---------------------------------------------------------------------------


def _page_css(font_scale: float) -> str:
    base = round(13.0 * font_scale, 2)
    return f"""
<style>
  :root {{
    --fig-text: #1c1c1e;
    --fig-text-2: #3c3c43;
    --fig-text-3: #6b6b70;
    --fig-accent: #0a5bd0;
    --fig-sep: #d8d8de;
    --fig-card-bg: #ffffff;
    --fig-page-bg: #f2f2f7;
    --fig-warn: #b25e00;
    --fig-err: #c0271d;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    background: var(--fig-page-bg);
    color: var(--fig-text);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Helvetica Neue", Arial, sans-serif;
    font-size: {base}px;
    line-height: 1.42;
    -webkit-font-smoothing: antialiased;
  }}
  .fig-page {{ display: flex; gap: 14px; padding: 16px; align-items: flex-start; }}
  .fig-panel {{ background: transparent; }}
  .fig-panel.brief  {{ flex: 0 0 56%; }}
  .fig-panel.source {{ flex: 0 0 calc(44% - 14px); }}
  .fig-panel.full   {{ flex: 1 1 100%; }}
  .fig-panel-title {{
    font-size: {round(base * 1.08, 2)}px; font-weight: 700;
    color: var(--fig-text-3); text-transform: uppercase; letter-spacing: .04em;
    margin: 0 2px 8px;
  }}
  .fig-card {{
    background: var(--fig-card-bg);
    border: 1px solid var(--fig-sep);
    border-radius: 12px;
    margin: 0 0 10px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }}
  .fig-card-header {{
    padding: 8px 12px;
    background: #fafafc;
    border-bottom: 1px solid var(--fig-sep);
    font-weight: 650;
    color: var(--fig-text);
  }}
  .fig-card-body {{ padding: 9px 13px; color: var(--fig-text-2); }}
  .fig-md {{ margin: 0 0 2px; }}
  .fig-md:last-child {{ margin-bottom: 0; }}
  .fig-caption {{ color: var(--fig-text-3); font-size: .86em; margin: 2px 0; }}
  .fig-divider {{ border: none; border-top: 1px solid var(--fig-sep); margin: 8px 0; }}
  code {{
    font-family: "SF Mono", ui-monospace, "Cascadia Code", Menlo, monospace;
    font-size: .9em; background: rgba(0,0,0,.045);
    padding: .06em .3em; border-radius: 5px;
  }}
  .fig-code {{
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: .86em; white-space: pre-wrap; background: #f6f6f8;
    border: 1px solid var(--fig-sep); border-radius: 8px; padding: 8px 10px; margin: 4px 0;
  }}
  sup {{ color: var(--fig-accent); font-weight: 600; font-size: .72em; }}
  .fig-cols {{ display: flex; gap: 12px; }}
  .fig-col {{ flex: 1; min-width: 0; }}
  .fig-callout {{
    border-radius: 8px; padding: 7px 11px; margin: 5px 0;
    border-left: 3px solid var(--fig-text-3); background: #f6f6f8; font-size: .94em;
  }}
  .fig-callout-warning {{ border-left-color: var(--fig-warn); background: #fff7ec; color: var(--fig-warn); }}
  .fig-callout-error   {{ border-left-color: var(--fig-err);  background: #fdeeed; color: var(--fig-err); }}
  .fig-callout-info    {{ border-left-color: var(--fig-accent); background: #eef4fd; }}
  table.fig-table {{
    border-collapse: collapse; width: 100%; font-size: .86em; margin: 4px 0;
  }}
  table.fig-table th, table.fig-table td {{
    border: 1px solid var(--fig-sep); padding: 3px 7px; text-align: left;
  }}
  table.fig-table th {{ background: #fafafc; font-weight: 650; }}
  @media print {{
    body {{ background: #fff; }}
    .fig-page {{ padding: 0; }}
    .fig-card {{ box-shadow: none; break-inside: avoid; }}
    @page {{ margin: 10mm; }}
  }}
</style>
"""


def _capture(render_fn, *args) -> str:
    """Run a renderer under the Streamlit shim and return captured HTML."""
    fake = _FakeStreamlit()
    sys.modules["streamlit"] = fake
    # Import renderers AFTER the shim is in place so their ``import streamlit``
    # binds to the fake. They are re-imported fresh each call is unnecessary;
    # module-level ``import streamlit as st`` resolves the name at use time.
    render_fn(fake, *args)
    return _flatten(fake.root)


def render_brief_html(fake: _FakeStreamlit, output: dict) -> None:
    from display.note_renderer import render_note  # noqa: WPS433

    # render_note uses the module-global ``st``; rebind it to our fake.
    import display.note_renderer as nr

    nr.st = fake
    render_note(output)


def render_source_html(fake: _FakeStreamlit, source: dict) -> None:
    from display.source_renderer import render_source  # noqa: WPS433
    import display.source_renderer as sr

    sr.st = fake
    render_source(source)


def build_document(
    output: dict,
    source: dict | None,
    *,
    font_scale: float,
    title: str,
) -> str:
    brief_html = _capture(render_brief_html, output)
    panels = []
    if source is not None:
        source_html = _capture(render_source_html, source)
        panels.append(
            f'<div class="fig-panel source"><div class="fig-panel-title">Source data</div>{source_html}</div>'
        )
        panels.append(
            f'<div class="fig-panel brief"><div class="fig-panel-title">Generated brief</div>{brief_html}</div>'
        )
        # Source on the left (wider), brief on the right — matches the app.
        panels = [panels[0], panels[1]]
    else:
        panels.append(
            f'<div class="fig-panel full"><div class="fig-panel-title">Generated brief</div>{brief_html}</div>'
        )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(title)}</title>"
        f"{_page_css(font_scale)}</head><body>"
        f'<div class="fig-page">{"".join(panels)}</div>'
        "</body></html>"
    )


def _load_env() -> None:
    """Load review_app/.env so Blob access works (mirrors the app)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(_APP_ROOT / ".env")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hosp-id", help="Load output+source from Blob via load_case().")
    src.add_argument("--output-json", help="Path to a local *.brief.json / output.json.")
    ap.add_argument("--source-json", help="Path to a local source_bundle.json (with --output-json).")
    ap.add_argument("--no-source", action="store_true", help="Brief only; omit the source panel.")
    ap.add_argument("--redact", action="store_true", help="Run Philter redaction on the case (PHI).")
    ap.add_argument(
        "--no-redact",
        action="store_true",
        help="Explicitly acknowledge an UNREDACTED render (e.g. synthetic/consented case).",
    )
    ap.add_argument("--font-scale", type=float, default=1.0, help="Multiply base font size (default 1.0).")
    ap.add_argument("-o", "--out", default="brief_figure.html", help="Output HTML path.")
    args = ap.parse_args()

    import json

    if args.hosp_id:
        _load_env()
        from storage.case_loader import load_case

        case = load_case(args.hosp_id)
        output, source = case.output, case.source
        label = args.hosp_id
    else:
        output = json.loads(Path(args.output_json).read_text())
        source = (
            json.loads(Path(args.source_json).read_text())
            if args.source_json and not args.no_source
            else None
        )
        label = Path(args.output_json).stem

    if args.no_source:
        source = None

    if not args.redact and not args.no_redact:
        ap.error(
            "Refusing to render without a redaction decision. Pass --redact for a "
            "publishable figure, or --no-redact to acknowledge a synthetic/consented case."
        )

    if args.redact:
        from redaction.philter_runner import redact_case_payload

        red_source, output = redact_case_payload(source or {}, output)
        source = None if args.no_source else red_source

    doc = build_document(output, source, font_scale=args.font_scale, title=f"ICU-PAUSE brief — {label}")
    out_path = Path(os.path.expanduser(args.out))
    out_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {out_path}")
    print("Open it in Chrome, then File -> Print -> Save as PDF (vector text).")
    if args.no_redact:
        print("WARNING: rendered WITHOUT redaction — confirm the case is synthetic/consented before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
