"""
Interface primitives: one type scale, one accent, one way to show a figure.

Everything visual in this application is built from the handful of pieces in
this module, so that a change to the look happens in one place rather than in
forty call sites. It implements milestone U0 of `docs/synthbio-ui-plan.pdf`.

Three rules this module exists to enforce.

A figure always carries its own caveat. The project's honesty work depends on
a reader seeing what a number is not, at the moment they see the number. Stated
as a paragraph between sections, the same sentence gets skipped, which is the
worst outcome: the number lands and the warning does not. So `figure` takes a
note and puts it directly under the value.

Numbers are formatted once. The same screen used to show 0.6, 63.33% and 44.0,
three different conventions in one row, because each call site formatted its
own. `number` and `percent` are the only way values reach the screen.

Status colour is never the only signal. Each state carries a glyph as well as a
colour, so the state survives a colour blind reader and a greyscale print. That
was already the instinct behind VERDICT_STATES in the old interface and it is
kept here.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
import streamlit as st

# The accent is the one colour the interface spends on attention. It matches
# primaryColor in .streamlit/config.toml, which is the theme the application
# already ships with.
ACCENT = "#4C8DFF"
MUTED = "#9AA6B8"
SURFACE = "#1A1F2B"
BORDER = "#2A3242"

# Status colours, each paired with a glyph that carries the same meaning.
STATE_STYLES = {
    "good": {"colour": "#3FB68B", "glyph": "●"},
    "caution": {"colour": "#E0A33E", "glyph": "◆"},
    "bad": {"colour": "#E2614A", "glyph": "▲"},
    "neutral": {"colour": MUTED, "glyph": "○"},
}

_CSS = f"""
<style>
/* One type scale: verdict, section, body, caption. Everything else inherits. */
.sb-verdict {{ font-size: 1.65rem; font-weight: 650; line-height: 1.25; margin: 0 0 .2rem 0; }}
.sb-lede {{ font-size: 1.02rem; line-height: 1.55; color: #D5DCE8; margin: 0 0 .2rem 0; }}
.sb-section {{
    font-size: .78rem; font-weight: 650; letter-spacing: .09em; text-transform: uppercase;
    color: {MUTED}; border-top: 1px solid {BORDER}; padding-top: .7rem; margin: 1.6rem 0 .7rem 0;
}}
.sb-note {{ font-size: .80rem; line-height: 1.45; color: {MUTED}; margin: .15rem 0 0 0; }}

/* The figure card. A heading plus a caption rather than st.metric, which clips
   a long label to one line and so hid the qualification it carried. */
.sb-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: .8rem .9rem; height: 100%;
}}
.sb-card-value {{ font-size: 1.45rem; font-weight: 640; line-height: 1.2; }}
.sb-card-label {{ font-size: .80rem; color: #C3CCDA; margin-top: .15rem; }}
.sb-card-note {{ font-size: .73rem; color: {MUTED}; margin-top: .35rem; line-height: 1.4; }}

/* The verdict band. The left rule carries the state colour; the glyph carries
   it again for anyone who cannot see the colour. */
.sb-band {{
    border-left: 4px solid var(--sb-state); background: {SURFACE};
    border-radius: 0 10px 10px 0; padding: 1rem 1.1rem; margin: .2rem 0 .6rem 0;
}}
.sb-band .sb-glyph {{ color: var(--sb-state); font-size: 1.1rem; margin-right: .45rem; }}

/* Chips: the agent state and the gate badge. */
.sb-chip {{
    display: inline-block; font-size: .70rem; font-weight: 600; letter-spacing: .03em;
    padding: .13rem .5rem; border-radius: 999px; border: 1px solid var(--sb-chip);
    color: var(--sb-chip); margin-right: .35rem; white-space: nowrap;
}}
.sb-agent {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: .7rem .8rem; height: 100%;
}}
.sb-agent-name {{ font-size: .88rem; font-weight: 640; margin-bottom: .1rem; }}
.sb-agent-q {{ font-size: .72rem; color: {MUTED}; line-height: 1.4; margin-bottom: .45rem; }}
.sb-agent-answer {{ font-size: .95rem; font-weight: 600; margin-bottom: .3rem; }}
.sb-dim {{ opacity: .55; }}

/* The codon diff, at text resolution. */
.sb-codon {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; }}
.sb-codon-changed {{ color: {ACCENT}; font-weight: 650; }}
</style>
"""


def inject_css() -> None:
    """Install the style sheet once per session."""
    if not st.session_state.get("_sb_css"):
        st.html(_CSS)
        st.session_state["_sb_css"] = True


# --- numbers ----------------------------------------------------------------


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def number(value: Any, places: int = 2, suffix: str = "") -> str:
    """A number formatted the one way this application formats numbers."""
    if _missing(value):
        return "N/A"
    return f"{float(value):,.{places}f}{suffix}"


def percent(value: Any, places: int = 1, already_scaled: bool = True) -> str:
    """A percentage. `already_scaled` is False for a probability in 0 to 1."""
    if _missing(value):
        return "N/A"
    scaled = float(value) if already_scaled else float(value) * 100.0
    return f"{scaled:,.{places}f}%"


def signed(value: Any, places: int = 3) -> str:
    if _missing(value):
        return "N/A"
    return f"{float(value):+,.{places}f}"


# --- layout -----------------------------------------------------------------


def section(title: str, note: str = "") -> None:
    """A section heading. A thin rule above, rather than a full width divider."""
    st.html(f'<div class="sb-section">{title}</div>')
    if note:
        st.html(f'<div class="sb-note">{note}</div>')


def figure(container, value: str, label: str, note: str = "") -> None:
    """One figure, its label, and the caveat that belongs to it."""
    container.html(
        f'<div class="sb-card">'
        f'<div class="sb-card-value">{value}</div>'
        f'<div class="sb-card-label">{label}</div>'
        + (f'<div class="sb-card-note">{note}</div>' if note else "")
        + "</div>"
    )


def figures(items: Sequence[dict]) -> None:
    """A row of figures, evenly spaced."""
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        figure(column, item["value"], item["label"], item.get("note", ""))


def band(state: str, headline: str, lede: str = "") -> None:
    """
    The verdict band at the top of the screen.

    One statement, one supporting sentence, and a colour that means something
    with a glyph that means the same thing.
    """
    style = STATE_STYLES.get(state, STATE_STYLES["neutral"])
    st.html(
        f'<div class="sb-band" style="--sb-state:{style["colour"]}">'
        f'<div class="sb-verdict"><span class="sb-glyph">{style["glyph"]}</span>{headline}</div>'
        + (f'<div class="sb-lede">{lede}</div>' if lede else "")
        + "</div>"
    )


def chip(text: str, state: str = "neutral") -> str:
    """An inline state chip. Returns markup, for embedding in a larger block."""
    colour = STATE_STYLES.get(state, STATE_STYLES["neutral"])["colour"]
    return f'<span class="sb-chip" style="--sb-chip:{colour}">{text}</span>'


def note(text: str) -> None:
    st.html(f'<div class="sb-note">{text}</div>')


def notes(lines: Iterable[str]) -> None:
    for line in lines:
        if line:
            note(line)
