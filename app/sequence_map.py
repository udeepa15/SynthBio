"""
The sequence map: one cell per codon, so a construct can be looked at.

This is milestone U2 of `docs/synthbio-ui-plan.pdf` and the centrepiece of the
interface work. Everything it draws was already computed; none of it was
visible. GC3, rare codon membership and local folding are per codon quantities
that the application reduced to three averages, so a person whose construct was
flagged could not see where in their gene the problem was.

Two design decisions worth stating.

It wraps rather than compresses. A 931 codon construct squeezed into the width
of a browser gives each codon less than a pixel, which is a texture rather than
a map. The track breaks into rows the way a sequence viewer does, so one cell
is always wide enough to point at.

It uses Altair, which ships with Streamlit. The deployed application runs on
free hosting alongside xgboost, shap, biopython and ViennaRNA, and the multi
agent plan is explicit that another heavy dependency may not fit. A rect mark
over a dataframe needs nothing new.

The colour scales deliberately avoid the status palette. A GC rich codon is not
a warning, and if it were drawn in the same red as a high risk verdict a reader
would read it as one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from src.features import RARE_CODONS

# Codons per row. Thirty keeps a cell comfortably clickable at the usual
# content width and makes the row boundaries land on round numbers, which
# helps when a user is counting to a position.
CODONS_PER_ROW = 30

# A cell is a fixed size rather than a share of the container. Stretching 20
# codons across a wide screen gives cells the shape of bricks and makes a short
# construct look longer than a long one, which is the opposite of what a map is
# for.
CELL = 22

# The channels a reader can colour the track by. Each is a quantity the
# application already computes per codon, or can compute here without a model.
CHANNELS = {
    "GC3, wobble position": "gc3",
    "Rare codons": "rare",
    "GC content": "gc",
}

CHANNEL_HELP = {
    "gc3": (
        "G or C at the third base of each codon. GC3 is the strongest single predictor of yield "
        "on Welch's synonymous variants, at +0.64 on the polymerase series and +0.39 on the scFv "
        "series."
    ),
    "rare": (
        "Codons read slowly in E. coli. A moderate share is associated with higher yield rather "
        "than lower, since pausing gives the nascent chain time to fold, so this is shown as a "
        "fact about the sequence rather than as a fault."
    ),
    "gc": "G or C at any position of the codon, which is what the whole construct GC average is of.",
}

# Regions worth marking above the track.
INITIATION_CODONS = 16  # the first 48 nt, the window Agent D reads
HEAD_CODONS = 30  # the window the rare codon head measure covers


def codon_frame(sequence: str, conserved: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """
    One row per codon, with everything the track needs to draw and to label.

    `conserved` is a set of zero based codon indices the redesign was not
    allowed to change, which the redesign tab passes so that locked regions are
    visible rather than described.
    """
    from src.redesign import split_codons, translate_codons

    clean = sequence.upper().replace(" ", "").replace("\n", "")
    codons = split_codons(clean)
    locked = set(conserved or ())

    rows: List[Dict[str, object]] = []
    for index, codon in enumerate(codons):
        third = codon[2] if len(codon) > 2 else ""
        rows.append({
            "index": index,
            "position": index + 1,
            "codon": codon,
            "amino_acid": translate_codons([codon]) if len(codon) == 3 else "",
            "gc3": 1.0 if third in ("G", "C") else 0.0,
            "gc": sum(1.0 for base in codon if base in ("G", "C")) / max(len(codon), 1),
            "rare": 1.0 if codon in RARE_CODONS else 0.0,
            "row": index // CODONS_PER_ROW,
            "column": index % CODONS_PER_ROW,
            "locked": index in locked,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["region"] = frame["index"].apply(_region_of)
    # Explicit yes and no columns rather than encoding a float as nominal,
    # which leaves Altair to decide how to spell 1.0 and quietly drops a scale
    # whose domain spells it differently.
    frame["gc3_label"] = frame["gc3"].map({1.0: "yes", 0.0: "no"})
    frame["rare_label"] = frame["rare"].map({1.0: "yes", 0.0: "no"})
    return frame


def _region_of(index: int) -> str:
    if index < INITIATION_CODONS:
        return "Initiation window, first 48 nt"
    if index < HEAD_CODONS:
        return "Head, first 30 codons"
    return "Body"


def _channel_encoding(channel: str):
    """The colour encoding for one channel, kept away from the status palette."""
    import altair as alt

    # GC3 and rare codon membership are per codon yes or no questions, so they
    # get two colours rather than a continuous ramp. Drawing a binary quantity
    # on a gradient invites a reader to see degrees that are not there.
    if channel == "rare":
        return alt.Color(
            "rare_label:N",
            scale=alt.Scale(domain=["no", "yes"], range=["#2A3242", "#B98CE0"]),
            legend=alt.Legend(title="Rare codon"),
        )
    if channel == "gc3":
        return alt.Color(
            "gc3_label:N",
            scale=alt.Scale(domain=["no", "yes"], range=["#2A3242", "#2E9E8F"]),
            legend=alt.Legend(title="G or C at the wobble position"),
        )
    return alt.Color(
        "gc:Q",
        scale=alt.Scale(scheme="teals", domain=[0, 1]),
        legend=alt.Legend(title="GC in the codon", format=".2f"),
    )


def render_track(
    sequence: str,
    channel: str = "gc3",
    conserved: Optional[Sequence[int]] = None,
    height_per_row: int = 30,
    key: Optional[str] = None,
) -> None:
    """Draw one construct as a wrapped per codon track."""
    import altair as alt

    frame = codon_frame(sequence, conserved)
    if frame.empty:
        st.caption("Nothing to draw: the sequence holds no complete codon.")
        return

    rows = int(frame["row"].max()) + 1
    base = alt.Chart(frame).encode(
        x=alt.X("column:O", axis=None),
        y=alt.Y("row:O", axis=None),
        tooltip=[
            alt.Tooltip("position:Q", title="Codon"),
            alt.Tooltip("codon:N", title="Triplet"),
            alt.Tooltip("amino_acid:N", title="Amino acid"),
            alt.Tooltip("region:N", title="Region"),
            alt.Tooltip(
                "gc3_label:N" if channel == "gc3"
                else "rare_label:N" if channel == "rare"
                else "gc:Q",
                title=CHANNELS_INVERSE.get(channel, channel),
            ),
        ],
    )

    cells = base.mark_rect(stroke="#0E1117", strokeWidth=1, cornerRadius=2).encode(
        color=_channel_encoding(channel)
    )

    layers = [cells]

    # Locked codons get a visible outline rather than a different fill, so the
    # channel colour underneath stays readable.
    if frame["locked"].any():
        layers.append(
            alt.Chart(frame[frame["locked"]]).mark_rect(
                fill=None, stroke="#E0A33E", strokeWidth=1.6, cornerRadius=2
            ).encode(x=alt.X("column:O", axis=None), y=alt.Y("row:O", axis=None))
        )

    # The initiation window, marked on the first row only, where it lives.
    initiation = frame[frame["index"] < INITIATION_CODONS]
    if not initiation.empty:
        layers.append(
            alt.Chart(initiation).mark_rect(
                fill=None, stroke="#4C8DFF", strokeWidth=1.4, cornerRadius=2
            ).encode(x=alt.X("column:O", axis=None), y=alt.Y("row:O", axis=None))
        )

    # Explicit pixel width and height rather than a step size. A layered chart
    # with a step width leaves Vega to reconcile the step against the layers
    # and the row band collapses to a few pixels, which turns the map into a
    # rule. Numbers are unambiguous.
    columns = int(frame["column"].max()) + 1
    # Width in pixels, height as a step per row. A pixel height on a layered
    # chart with a one band ordinal axis collapses the band to a few pixels and
    # draws the map as a rule; a step says how tall one row of codons is and
    # lets Vega size the chart from it.
    chart = alt.layer(*layers).properties(
        width=CELL * max(columns, 1),
        height=alt.Step(height_per_row),
        padding={"left": 4, "right": 4, "top": 2, "bottom": 2},
    ).configure_view(strokeWidth=0).configure_legend(
        orient="bottom", direction="horizontal", titleColor="#9AA6B8", labelColor="#9AA6B8"
    )
    # Streamlit's newer signature: "content" means take the size the chart
    # declares. Passing use_container_width instead let the width stretch and
    # the height collapse, which turned the map into a single rule.
    st.altair_chart(chart, width="content", height="content", key=key)


CHANNELS_INVERSE = {v: k for k, v in CHANNELS.items()}


def render_legend(sequence: str, conserved: Optional[Sequence[int]] = None) -> None:
    """One line naming the markers that are drawn over the track."""
    parts = [
        "Blue outline: the initiation window, the first 48 nt, which is the region the structure "
        "agent reads."
    ]
    if conserved:
        parts.append("Amber outline: codons you locked, which the search could not change.")
    codons = len(sequence.replace(" ", "")) // 3
    if codons > CODONS_PER_ROW:
        parts.append(f"The track wraps every {CODONS_PER_ROW} codons, reading left to right.")
    st.html('<div class="sb-note">' + " ".join(parts) + "</div>")


def channel_picker(key: str, default: str = "gc3") -> str:
    """The segmented control that chooses what the track is coloured by."""
    labels = list(CHANNELS)
    default_label = CHANNELS_INVERSE.get(default, labels[0])
    chosen = st.segmented_control(
        "Colour the construct by",
        labels,
        default=default_label,
        key=key,
        label_visibility="collapsed",
    )
    channel = CHANNELS.get(chosen or default_label, default)
    st.html(f'<div class="sb-note">{CHANNEL_HELP.get(channel, "")}</div>')
    return channel


# --- the redesign diff ------------------------------------------------------


def diff_frame(original: str, redesigned: str, conserved: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Both sequences as one frame, marked where they differ."""
    before = codon_frame(original, conserved)
    after = codon_frame(redesigned, conserved)
    if before.empty or after.empty:
        return pd.DataFrame()

    length = min(len(before), len(after))
    before, after = before.iloc[:length].copy(), after.iloc[:length].copy()
    changed = (before["codon"].to_numpy() != after["codon"].to_numpy())
    before["changed"], after["changed"] = changed, changed
    before["track"], after["track"] = "Original", "Redesigned"
    before["other"], after["other"] = after["codon"].to_numpy(), before["codon"].to_numpy()
    return pd.concat([before, after], ignore_index=True)


def render_diff(
    original: str,
    redesigned: str,
    channel: str = "gc3",
    conserved: Optional[Sequence[int]] = None,
    height_per_row: int = 24,
) -> None:
    """
    Two aligned tracks, before above and after below, changed codons outlined.

    The engine already guarantees the protein is unchanged and checks it on
    every candidate. Showing the two tracks aligned is what makes that
    guarantee visible rather than merely stated, because the amino acid is in
    the tooltip of both rows at every position that changed.
    """
    import altair as alt

    frame = diff_frame(original, redesigned, conserved)
    if frame.empty:
        st.caption("Nothing to compare.")
        return

    rows = int(frame["row"].max()) + 1
    charts = []
    for row_index in range(rows):
        part = frame[frame["row"] == row_index]
        base = alt.Chart(part).encode(
            x=alt.X("column:O", axis=None),
            y=alt.Y("track:N", sort=["Original", "Redesigned"],
                    axis=alt.Axis(title=None, labelColor="#9AA6B8", labelFontSize=10, ticks=False,
                                  domain=False)),
            tooltip=[
                alt.Tooltip("position:Q", title="Codon"),
                alt.Tooltip("track:N", title="Track"),
                alt.Tooltip("codon:N", title="Triplet"),
                alt.Tooltip("other:N", title="Other track"),
                alt.Tooltip("amino_acid:N", title="Amino acid"),
            ],
        )
        layers = [base.mark_rect(stroke="#0E1117", strokeWidth=1, cornerRadius=2).encode(
            color=_channel_encoding(channel)
        )]
        altered = part[part["changed"]]
        if not altered.empty:
            layers.append(
                alt.Chart(altered).mark_rect(
                    fill=None, stroke="#4C8DFF", strokeWidth=1.8, cornerRadius=2
                ).encode(
                    x=alt.X("column:O", axis=None),
                    y=alt.Y("track:N", sort=["Original", "Redesigned"], axis=None),
                )
            )
        columns = int(part["column"].max()) + 1
        charts.append(
            alt.layer(*layers).properties(
                width=CELL * max(columns, 1), height=alt.Step(height_per_row)
            )
        )

    chart = alt.vconcat(*charts, spacing=6).configure_view(strokeWidth=0).configure_legend(
        orient="bottom", direction="horizontal", titleColor="#9AA6B8", labelColor="#9AA6B8"
    )
    st.altair_chart(chart, width="content", height="content")


def changed_runs(original: str, redesigned: str) -> pd.DataFrame:
    """
    The changes grouped into runs, which is how a person reads a diff.

    Twenty scattered single codon changes are twenty rows in a table and four
    runs in a sentence. The amino acid column stays, because it is the proof
    the protein did not move.
    """
    from src.redesign import split_codons, translate_codons

    before = split_codons(original)
    after = split_codons(redesigned)
    runs: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None

    for index, (old, new) in enumerate(zip(before, after)):
        if old != new:
            if current is None:
                current = {"start": index + 1, "end": index + 1, "old": [old], "new": [new]}
            else:
                current["end"] = index + 1
                current["old"].append(old)
                current["new"].append(new)
        elif current is not None:
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)

    return pd.DataFrame([
        {
            "Codons": (
                f"{run['start']}" if run["start"] == run["end"] else f"{run['start']} to {run['end']}"
            ),
            "Original": " ".join(run["old"]),
            "Redesigned": " ".join(run["new"]),
            "Amino acids, unchanged": translate_codons(run["old"]),
        }
        for run in runs
    ])
