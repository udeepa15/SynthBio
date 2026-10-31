"""
Showing how the agents work, which the old interface only told you.

Milestone U4 of `docs/synthbio-ui-plan.pdf`. Two pieces: a strip of four
compact cards at depth two, and a flow diagram that describes this construct
rather than the architecture in the abstract.

The point of both is one idea. Three of four agents abstaining is normal and
correct, and when a reader can see that three declined and read one line each
on why, abstention stops looking like a failure and starts looking like the
design. The old interface rendered four identical collapsed rows, which carried
none of that.

Nothing here invents content. The name, the question, the applied flag, the
abstention reason, the gate result and the attribution all come straight off
`AgentOpinion`.
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from app import ui
from src.agents.base import AGENT_ORDER, GATE_BLOCKED, GATE_FAILED, GATE_PASSED

# What each agent is called on screen. The internal names are lower case keys.
AGENT_LABELS = {
    "expression": "Expression",
    "solubility": "Solubility",
    "variant": "Codon variant",
    "structure": "Structure",
}

# One short phrase per agent, shorter than the full question, for the card.
AGENT_SHORT = {
    "expression": "Will it express, and in which tier",
    "solubility": "Soluble, or inclusion bodies",
    "variant": "Which synonymous version is better",
    "structure": "Does the 5' end block initiation",
}

GATE_CHIP_STATE = {
    GATE_PASSED: "good",
    GATE_FAILED: "bad",
    GATE_BLOCKED: "caution",
}


def _gate_chip(gate) -> str:
    state = GATE_CHIP_STATE.get(gate.status, "neutral")
    label = f"gate {gate.status}"
    if gate.measured_on:
        label += f" {gate.measured_on:%b %Y}"
    return ui.chip(label, state)


def _answer_line(opinion) -> str:
    """The agent's answer in one phrase, in its own units."""
    if not opinion.applies:
        return "No opinion"
    if opinion.proba and set(opinion.proba) == {0, 1}:
        return f"{ui.percent(opinion.proba[1], 0, already_scaled=False)} lowest band"
    if opinion.score is not None:
        return ui.number(opinion.score, 2)
    return "Answered"


def render_strip(verdict) -> None:
    """
    One card per agent: what it answers, whether it did, and its gate.

    A card is dimmed when the agent abstained, so the shape of the row tells
    the reader at a glance how much of the system spoke.
    """
    if verdict is None:
        return

    columns = st.columns(len(AGENT_ORDER))
    for column, name in zip(columns, AGENT_ORDER):
        opinion = verdict.opinions.get(name)
        if opinion is None:
            continue
        applied = opinion.applies
        state_chip = ui.chip("applied", "good") if applied else ui.chip("abstained", "neutral")
        dim = "" if applied else " sb-dim"
        column.html(
            f'<div class="sb-agent{dim}">'
            f'<div class="sb-agent-name">{AGENT_LABELS.get(name, name.title())}</div>'
            f'<div class="sb-agent-q">{AGENT_SHORT.get(name, opinion.question)}</div>'
            f'<div class="sb-agent-answer">{_answer_line(opinion)}</div>'
            f'<div>{state_chip}{_gate_chip(opinion.gate)}</div>'
            "</div>"
        )


def render_detail(verdict) -> None:
    """Depth three: every agent, its reason, its gate line and its attribution."""
    if verdict is None:
        return
    for name in AGENT_ORDER:
        opinion = verdict.opinions.get(name)
        if opinion is None:
            continue
        mark = "Applied" if opinion.applies else "Abstained"
        with st.expander(f"{AGENT_LABELS.get(name, name.title())}: {mark}", expanded=False):
            st.caption(opinion.question)
            if opinion.applies:
                st.markdown(f"**{_answer_line(opinion)}**")
                top = opinion.top_attribution(4)
                if top:
                    st.caption(
                        "What moved this answer: "
                        + ", ".join(f"{feature} {ui.signed(value)}" for feature, value in top)
                    )
            else:
                st.caption(opinion.abstain_reason or "No reason recorded.")
            st.caption(opinion.gate.as_line())
            for line in opinion.domain.notes:
                st.caption(line)
            for line in opinion.provenance.notes:
                st.caption(line)


# --- the flow diagram -------------------------------------------------------

_LANE_Y = {"expression": 28, "solubility": 70, "variant": 112, "structure": 154}


def _lane(name: str, label: str, applied: bool) -> str:
    y = _LANE_Y[name]
    colour = "#4C8DFF" if applied else "#3A4356"
    text = "#D5DCE8" if applied else "#7A879B"
    fill = "#1A1F2B"
    return (
        f'<rect x="150" y="{y}" width="150" height="30" rx="7" fill="{fill}" '
        f'stroke="{colour}" stroke-width="1.6"/>'
        f'<text x="225" y="{y + 19}" text-anchor="middle" font-size="12" fill="{text}" '
        f'font-family="sans-serif">{label}</text>'
        f'<path d="M110 100 C 130 100, 130 {y + 15}, 150 {y + 15}" fill="none" '
        f'stroke="{colour}" stroke-width="1.4" opacity="{"0.9" if applied else "0.35"}"/>'
        f'<path d="M300 {y + 15} C 320 {y + 15}, 320 100, 340 100" fill="none" '
        f'stroke="{colour}" stroke-width="1.4" opacity="{"0.9" if applied else "0.35"}"/>'
    )


def render_diagram(verdict) -> None:
    """
    The architecture, drawn for this construct.

    Lanes are in full colour when the agent applied and greyed when it
    abstained, so the picture is about the sequence on screen rather than a
    generic diagram of the design.
    """
    if verdict is None:
        return

    lanes = "".join(
        _lane(name, AGENT_LABELS.get(name, name.title()), bool(
            verdict.opinions.get(name) is not None and verdict.opinions[name].applies
        ))
        for name in AGENT_ORDER
    )
    applying = verdict.agreement.n_applying
    state = verdict.state_label

    svg = f"""
<svg viewBox="0 0 470 200" width="100%" role="img" aria-label="Four agents feed one critic">
  <rect x="10" y="84" width="100" height="32" rx="7" fill="#1A1F2B" stroke="#2A3242"/>
  <text x="60" y="104" text-anchor="middle" font-size="12" fill="#D5DCE8"
        font-family="sans-serif">Construct</text>
  {lanes}
  <rect x="340" y="80" width="120" height="40" rx="7" fill="#1A1F2B" stroke="#4C8DFF"
        stroke-width="1.6"/>
  <text x="400" y="97" text-anchor="middle" font-size="12" fill="#D5DCE8"
        font-family="sans-serif">Critic</text>
  <text x="400" y="112" text-anchor="middle" font-size="10" fill="#9AA6B8"
        font-family="sans-serif">{applying} of 4 applied</text>
</svg>
"""
    st.html(svg)
    ui.note(
        f"Each agent answers one question and may decline. {applying} of 4 applied to this "
        f"construct, and the critic returned: {state}. An agent speaks only when it holds an "
        "artifact whose recorded gate is a pass, so an abstention is a result rather than an "
        "error."
    )


def render_how_it_works(verdict) -> None:
    """The diagram in an expander, open the first time and remembered after."""
    if verdict is None:
        return
    seen = st.session_state.get("_sb_how_seen", False)
    with st.expander("How this works", expanded=not seen):
        render_diagram(verdict)
    st.session_state["_sb_how_seen"] = True
