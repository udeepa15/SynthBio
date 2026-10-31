"""Build the SynthBio interface plan PDF.

Output: docs/synthbio-ui-plan.pdf

Writing rules for this document, per the project's standing preferences: no
hyphens, en dashes or em dashes in prose. Code identifiers, file names and
library names keep whatever characters they really have.

The visual style deliberately matches docs/make_multiagent_plan.py, so the two
documents read as one set.
"""

from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthbio-ui-plan.pdf")

INK = colors.HexColor("#15202b")
MUTED = colors.HexColor("#5a6a78")
ACCENT = colors.HexColor("#0f6f5c")
WARN = colors.HexColor("#9c3328")
RULE = colors.HexColor("#d6dde3")
BOXBG = colors.HexColor("#f2f6f7")
WARNBG = colors.HexColor("#fdf3f1")

ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["BodyText"])
    return ParagraphStyle(name, parent=base, **kw)


S_TITLE = style("t", fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=INK, spaceAfter=6)
S_SUB = style("sub", fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED)
S_META = style("meta", fontName="Helvetica", fontSize=9.5, leading=14, textColor=MUTED)
S_H1 = style("h1", fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=INK,
             spaceBefore=16, spaceAfter=7)
S_H2 = style("h2", fontName="Helvetica-Bold", fontSize=11.6, leading=14.5, textColor=ACCENT,
             spaceBefore=12, spaceAfter=4)
S_H3 = style("h3", fontName="Helvetica-BoldOblique", fontSize=10, leading=13, textColor=INK,
             spaceBefore=8, spaceAfter=3)
S_BODY = style("body", fontName="Helvetica", fontSize=9.9, leading=14.4, textColor=INK,
               spaceAfter=6, alignment=TA_LEFT)
S_BULLET = style("bul", parent=S_BODY, spaceAfter=3, leading=13.8)
S_SMALL = style("small", fontName="Helvetica", fontSize=8.6, leading=12, textColor=MUTED, spaceAfter=4)
S_CELL = style("cell", fontName="Helvetica", fontSize=8.5, leading=11.4, textColor=INK)
S_CELLH = style("cellh", fontName="Helvetica-Bold", fontSize=8.5, leading=11.4, textColor=colors.white)
S_BOXH = style("boxh", fontName="Helvetica-Bold", fontSize=9.6, leading=12.6, textColor=INK, spaceAfter=3)
S_BOX = style("box", parent=S_BODY, fontSize=9.4, leading=13.4, spaceAfter=3)
S_CODE = style("code", fontName="Courier", fontSize=8.3, leading=11.6, textColor=INK, spaceAfter=5)
S_TOC = style("toc", fontName="Helvetica", fontSize=10, leading=17, textColor=INK)

story = []
A = story.append


def para(text, st=S_BODY):
    A(Paragraph(text, st))


def h1(text):
    A(Paragraph(text, S_H1))


def h2(text):
    A(Paragraph(text, S_H2))


def h3(text):
    A(Paragraph(text, S_H3))


def bullets(items, st=S_BULLET):
    A(ListFlowable(
        [ListItem(Paragraph(i, st), leftIndent=12, value=None) for i in items],
        bulletType="bullet", start="square", bulletFontSize=5,
        bulletColor=ACCENT, leftIndent=12, spaceAfter=6,
    ))


def numbered(items, st=S_BULLET):
    A(ListFlowable(
        [ListItem(Paragraph(i, st), leftIndent=16) for i in items],
        bulletType="1", bulletFontName="Helvetica-Bold", bulletFontSize=9,
        bulletColor=ACCENT, leftIndent=16, spaceAfter=6,
    ))


def box(title, lines, warn=False, width=168 * mm):
    inner = []
    if title:
        inner.append(Paragraph(title, S_BOXH))
    for line in lines:
        inner.append(Paragraph(line, S_BOX))
    t = Table([[inner]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARNBG if warn else BOXBG),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, WARN if warn else ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    A(Spacer(1, 3))
    A(t)
    A(Spacer(1, 8))


def table(header, rows, widths, align_right=()):
    data = [[Paragraph(c, S_CELLH) for c in header]]
    for r in rows:
        data.append([c if hasattr(c, "wrap") else Paragraph(c, S_CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafb")]),
    ]
    for c in align_right:
        cmds.append(("ALIGN", (c, 1), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(cmds))
    A(Spacer(1, 2))
    A(t)
    A(Spacer(1, 9))


# ------------------------------------------------------------------ cover
A(Spacer(1, 24 * mm))
para("SynthBio", S_SUB)
A(Spacer(1, 2))
para("A plan for the<br/>interface", S_TITLE)
A(Spacer(1, 6))
para(
    "How to show one answer first, put the evidence one click away, and draw the construct "
    "instead of listing numbers about it.",
    style("lead", parent=S_SUB, fontSize=11.4, leading=16.4),
)
A(Spacer(1, 10 * mm))
t = Table([[""]], colWidths=[168 * mm], rowHeights=[2])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
A(t)
A(Spacer(1, 8 * mm))
para("Prepared for Udeepa, 31 October 2026.", S_META)
para(
    "Grounded in app/main.py as it stands at 980 lines, the four agents in src/agents/, and the "
    "gate results in reports/gates.md. Every behaviour this document proposes to show already "
    "exists in the code; nothing here asks for a new claim about biology.",
    S_META,
)

A(Spacer(1, 10 * mm))
box("Read this first", [
    "The application answers a question with money attached: is this construct worth synthesising. "
    "Today it answers by printing thirteen figures, five blocks of caveat text, a waterfall chart "
    "and four collapsible agent rows, all at the same level of emphasis, before the person has "
    "done anything beyond pasting a sequence.",
    "That is not a styling problem and it will not be fixed by picking better colours. It is a "
    "hierarchy problem: the interface has no opinion about which of its thirty or so elements the "
    "reader should look at first, so the reader has to form that opinion themselves, every time.",
    "The honesty work this project has already done is worth protecting here. The caveats are "
    "correct and they must survive this redesign. What changes is that a caveat sits next to the "
    "number it qualifies, rather than being one more paragraph competing with the answer.",
])

A(NextPageTemplate("body"))
A(PageBreak())

# ------------------------------------------------------------------ contents
h1("Contents")
for line in [
    "1. What is wrong with the interface now",
    "2. Four principles",
    "3. The three depths, and what lives at each",
    "4. The sequence map, which is the centrepiece",
    "5. The redesign diff: showing what actually changed",
    "6. Showing how the agents work",
    "7. The four verdict states on screen",
    "8. Professional polish: the specifics",
    "9. What to remove, and what to stop repeating",
    "10. Milestones, in order",
    "11. What this plan does not change",
    "12. Open decisions for Udeepa",
]:
    para(line, S_TOC)

# ------------------------------------------------------------------ 1
h1("1. What is wrong with the interface now")

para(
    "The diagnosis tab renders, in one vertical run and at one weight: a risk band, a banner "
    "repeating that band, four figures, five computed features, three further measures under a "
    "quieter heading, two heuristic diagnostics, a provenance line, a calibration notice, four "
    "agent rows and a SHAP waterfall. The redesign tab opens with a warning paragraph of about "
    "ninety words before the person reaches a control."
)
para(
    "Each of those elements was added for a good reason and most of them are things a careful "
    "reader should eventually see. The failure is that they arrive together."
)

h2("The four specific problems")
numbered([
    "<b>No answer, only outputs.</b> The person wants to know whether to order the gene. The "
    "screen gives a tier, a risk percentage, a confidence, a tier midpoint in arbitrary units and "
    "an out of range warning, and leaves them to combine those into a decision.",
    "<b>Nothing is drawn.</b> A coding sequence is a position indexed object with regions that "
    "matter: the initiation window, the rare codon clusters, the wobble positions a redesign can "
    "move. The interface reduces all of that to scalars. A person cannot see where in their "
    "construct the problem is.",
    "<b>The redesign reports a diff it never shows.</b> It says six codons changed and offers a "
    "table of positions, but the two sequences are shown as two separate text boxes. The user "
    "has to diff them by eye to see what the tool did to their gene.",
    "<b>The agents are told, not shown.</b> Four collapsible rows with a question each, a gate "
    "line and an abstention reason. The architecture, which is the interesting part of this "
    "project, is invisible: nothing conveys that four narrow models looked at the construct, "
    "that three declined, and why declining is the right behaviour.",
])

box("The thing to keep", [
    "The glyph system in VERDICT_STATES, where each risk band carries a shape as well as a "
    "colour, is already the right instinct and should be extended rather than replaced. So is "
    "the practice of writing what a number is not, directly under it.",
])

# ------------------------------------------------------------------ 2
h1("2. Four principles")

h3("One answer above the fold")
para(
    "The top of the screen holds a single statement and a single supporting line. Everything "
    "else moves down or behind a disclosure. If the system is conflicted or cannot answer, that "
    "is the statement, which is why the critic's four states matter to the layout."
)

h3("Three depths, never more")
para(
    "Depth one is the verdict and the one sentence under it. Depth two is why, which is the "
    "sequence map, the three features that moved the answer, and the agent strip. Depth three is "
    "everything auditable: all features, the full attribution, the gate reports, the provenance. "
    "A reader who wants depth three is in the minority and should not pay for it in scroll."
)

h3("Draw the molecule, not just the number")
para(
    "Where a quantity is positional, show it positionally. GC3, rare codons, the initiation "
    "window and the redesign diff are all per codon, and all of them are currently averages."
)

h3("A caveat belongs to a number, not to the page")
para(
    "Every caveat currently on screen is accurate and should stay. Attached to its own figure, "
    "as a tooltip or a line under it, a caveat reads as rigour. Stacked as paragraphs between "
    "sections, the same sentences read as noise and get skipped, which is the worst outcome: the "
    "reader sees the number and misses the warning."
)

A(PageBreak())

# ------------------------------------------------------------------ 3
h1("3. The three depths, and what lives at each")

table(
    ["Depth", "What the person sees", "How it is reached"],
    [
        ["<b>1. The verdict</b>",
         "One of the four critic states, as a headline. One sentence of plain language under it. "
         "One primary action, which is either Redesign or Fix the input.",
         "Immediately, no interaction"],
        ["<b>2. Why</b>",
         "The sequence map. Three figures at most, chosen by attribution rather than fixed. The "
         "agent strip, one line per agent. The corroboration or conflict statement.",
         "Same screen, below the fold"],
        ["<b>3. The evidence</b>",
         "All computed features. The SHAP waterfall. Per agent detail with the gate result and "
         "date. Provenance. Heuristic diagnostics. The out of range table.",
         "Expanders, collapsed by default"],
    ],
    [26 * mm, 104 * mm, 38 * mm],
)

box("Why not tabs for this", [
    "The application already learned this: a comment in app/main.py records that Streamlit "
    "renders every tab on every run, which is why the sequence source became a radio rather than "
    "a tab. Tabs also hide the existence of what is in them. Expanders on one page keep the "
    "structure visible and let a reader scan the headings before deciding to open anything.",
    "Diagnosis and Redesign stay as the two top level tabs, because they are different tasks "
    "rather than different depths of one answer.",
])

# ------------------------------------------------------------------ 4
h1("4. The sequence map, which is the centrepiece")

para(
    "One horizontal track, one cell per codon, running the length of the construct. This is the "
    "single highest value addition in this plan, because it turns six scalars into a picture of "
    "the user's own gene and because every quantity it needs is already computed."
)

h2("What the track shows")
bullets([
    "<b>Colour channel, switchable:</b> GC3 at the wobble position, rare codon membership, or "
    "local folding energy. One channel at a time, chosen by a small segmented control.",
    "<b>Region markers above the track:</b> the vector leader when one is declared, the "
    "initiation window that Agent D reads, and the first thirty codons that the rare codon head "
    "measure covers.",
    "<b>Conserved regions</b>, in the redesign tab, drawn as locked spans so the user can see "
    "what the search was not allowed to touch.",
    "<b>Hover detail:</b> codon index, the codon itself, its amino acid, and the value of the "
    "active channel. Streamlit's charting gives tooltips without extra work.",
])

h2("How to build it without a new dependency")
para(
    "Altair ships with Streamlit, so a rect mark over a per codon dataframe is enough: x is the "
    "codon index, y is a single row, colour is the channel. The free hosting constraint that the "
    "multi agent plan flagged applies here too, and this adds nothing to the dependency list."
)
para(
    "For long constructs the track wraps into rows of a fixed number of codons, the way a "
    "sequence viewer does, rather than compressing a 931 codon gene into 700 pixels.",
)

box("What this fixes, concretely", [
    "A person whose construct is flagged can see that the rare codons sit in the first ten "
    "positions rather than being told the fraction is 0.12. A person using the redesign can see "
    "that every changed codon landed outside their conserved region. Neither of those is a new "
    "measurement; both are invisible today.",
])

A(PageBreak())

# ------------------------------------------------------------------ 5
h1("5. The redesign diff: showing what actually changed")

para(
    "The redesign engine already guarantees that the protein is unchanged and checks it on every "
    "candidate. The interface should make that guarantee visible rather than stating it."
)

h2("The diff view")
numbered([
    "Two stacked tracks, original above and redesigned below, aligned by codon index, with "
    "changed positions outlined in the accent colour. At a glance the user sees how many changes "
    "there were and where they clustered.",
    "A third, thinner track showing the per codon change in the active channel, so a run of "
    "changes that raised GC3 reads as a block rather than as a row in a table.",
    "Below the tracks, the changed codons at text resolution, grouped into runs and written as "
    "position, old codon, new codon, amino acid unchanged. The existing changed_codon_table "
    "already produces exactly these columns.",
    "The amino acid column stays visible rather than being a footnote, because it is the proof "
    "that the protein did not move.",
])

h2("The objective, stated once and honestly")
para(
    "The redesign tab currently opens with ninety words of calibration warning. That text is "
    "correct and it is the wrong shape. Replace it with one line next to the objective figure "
    "naming which agent produced the score, and move the Welch per series numbers into an "
    "expander titled with the thing a reader wants to know, which is whether this score has ever "
    "been right."
)

box("Already correct, keep as is", [
    "The corroboration banner, with its four verdicts and distinct icons, is the best designed "
    "element in the application. It states a conclusion, gives the reason, and changes colour on "
    "the conclusion rather than on the number. The rest of the interface should be brought up to "
    "its standard rather than the other way around.",
])

# ------------------------------------------------------------------ 6
h1("6. Showing how the agents work")

para(
    "The architecture is the project's distinguishing idea and the interface currently conveys "
    "none of it. Two additions fix that, and neither invents anything."
)

h2("The agent strip")
para(
    "A single row of four compact cards at depth two, one per agent, each showing the agent's "
    "name, a one line question, a state chip reading applied, abstained or not built, and the "
    "gate badge. The badge is the honest signal: passed with a date, failed, or not run. Clicking "
    "a card opens that agent's detail at depth three."
)
para(
    "The state chip is what teaches the idea. Three of four agents abstaining is normal, and when "
    "the reader can see that three declined and read one line each on why, abstention stops "
    "looking like an error and starts looking like the design."
)

h2("The flow diagram")
para(
    "One small static diagram, drawn as inline SVG so it costs nothing at runtime: the construct "
    "enters on the left, fans out to four agent lanes, and converges on the critic, which emits "
    "the verdict. Lanes are drawn in full colour when the agent applied and greyed when it "
    "abstained, so the diagram describes this construct rather than the architecture in the "
    "abstract."
)
para(
    "It belongs in an expander titled How this works, open on a first visit and remembered as "
    "closed afterwards through session state."
)

table(
    ["Element", "Where the content already exists"],
    [
        ["Agent name, question", "AGENT_QUESTIONS in src/agents/base.py"],
        ["Applied or abstained, and why", "AgentOpinion.applies and abstain_reason"],
        ["Gate badge and date", "GateResult.as_line, already rendered as text"],
        ["Attribution for the agent's answer", "AgentOpinion.top_attribution"],
        ["Agreement and the verdict state", "CriticVerdict.state_label and agreement.as_line"],
    ],
    [62 * mm, 106 * mm],
)

A(PageBreak())

# ------------------------------------------------------------------ 7
h1("7. The four verdict states on screen")

para(
    "The critic returns four states and the interface currently renders a risk band instead. "
    "Each state needs its own treatment, and the conflicted state is the one to design first, "
    "because it has to look like a deliberate answer rather than a failure."
)

table(
    ["State", "Headline", "What is shown", "Primary action"],
    [
        ["Agreed", "The tier, stated plainly", "Map, top three drivers, agent strip", "Redesign"],
        ["Agreed, out of domain",
         "The tier, labelled an extrapolation",
         "Same, plus which feature is outside which range, inline",
         "Redesign, with a caution"],
        ["Conflicted",
         "No tier. A statement that the evidence disagrees",
         "Which agents disagree, on what feature, and the one measurement that would settle it",
         "Measure, not redesign"],
        ["Not answerable",
         "No tier. What is missing",
         "The validated GRAVY rule result, labelled as a rule rather than a model",
         "Fix the input"],
    ],
    [30 * mm, 42 * mm, 64 * mm, 32 * mm],
)

box("The conflicted state is the product", [
    "The multi agent plan argues that a well calibrated refusal is a better product than a "
    "confident guess, for a tool whose purpose is to stop someone spending money. If that is "
    "true then the conflicted screen is not an error state, it is the screen the whole "
    "architecture exists to be able to draw. It should look as finished as the agreed state: "
    "same layout, same weight, a different conclusion.",
])

# ------------------------------------------------------------------ 8
h1("8. Professional polish: the specifics")

h2("Typography and rhythm")
bullets([
    "One type scale with four steps: verdict, section, body, caption. The application currently "
    "mixes h3, h4, h6 and caption in ways that do not map to a hierarchy.",
    "Section headings carry a thin rule above them rather than a divider between every block. The "
    "current layout uses st.divider four times in one tab, which reads as four equal walls.",
    "A consistent number format: one decimal for percentages, three for correlations, thousands "
    "separators everywhere. Today the same screen shows 0.6 and 63.33% and 44.0.",
])

h2("Colour discipline")
bullets([
    "One accent colour, reserved for the verdict and for interactive affordances. The theme "
    "already declares a single primaryColor and the interface should stop spending it elsewhere.",
    "Status colour only on status. Green, amber and red mean low, moderate and high risk, and "
    "they are always paired with the existing glyphs so the state survives a colour blind reader "
    "or a greyscale print.",
    "The sequence map needs a sequential scale for continuous channels and a categorical pair for "
    "membership channels. They must not reuse the status palette, or a GC rich codon will read as "
    "a warning.",
])

h2("The states nobody designs")
bullets([
    "<b>Empty:</b> before a sequence is entered, show what the tool does in one sentence and an "
    "example construct that can be loaded with one click, rather than a pre filled sample that "
    "quietly becomes the thing being scored.",
    "<b>Loading:</b> the redesign already drives a progress bar; feature extraction and the "
    "agents should do the same, because ViennaRNA folding on a long construct is slow enough to "
    "look like a hang.",
    "<b>Error:</b> validation issues are already well written. They should appear next to the "
    "input, not after it, so the person reads them before pressing the button.",
])

A(PageBreak())

# ------------------------------------------------------------------ 9
h1("9. What to remove, and what to stop repeating")

table(
    ["Element", "Why it goes or changes"],
    [
        ["The risk banner",
         "It restates the risk band and the tier immediately under the heading that already "
         "states both. One of the two goes."],
        ["Tier midpoint, arbitrary units",
         "A number with no unit and no meaning beyond the tier it was derived from. It is "
         "already labelled as not a measured yield. Remove it, or move it to depth three."],
        ["The 5' folding proxy figure",
         "It is a monotonic function of 5' GC, as reports/gates.md now records at a correlation "
         "of exactly minus one. Showing it beside the real ViennaRNA energy invites a reader to "
         "treat two names as two measurements."],
        ["The second features row",
         "Additional measures is a heading that tells the reader these are less important, which "
         "is the opposite of true: GC3 is the strongest single predictor on Welch. Merge the two "
         "rows and order them by what moved this answer."],
        ["The calibration notice",
         "Correct and important, and currently a wall between sections. It becomes the one "
         "sentence under the verdict plus a line in the agent detail."],
    ],
    [46 * mm, 122 * mm],
)

box("Not removed", [
    "Every statement of what a number is not stays in the interface. The provenance line, the "
    "out of range warning, the note that the model should not be used to choose between "
    "synonymous variants, and the gate results all stay. They move closer to what they qualify "
    "and further from the top of the page.",
])

# ------------------------------------------------------------------ 10
h1("10. Milestones, in order")

para(
    "Ordered so that each one leaves the application working and so the riskiest layout work "
    "happens before the decorative work."
)

table(
    ["", "Milestone", "What it produces", "Done when"],
    [
        ["U0", "Foundations",
         "One type scale, one accent, number formatting helpers, a card primitive to replace the "
         "ad hoc render_figure calls. No content changes.",
         "The screen looks the same but is built from four primitives"],
        ["U1", "The answer first",
         "Diagnosis restructured into verdict, why and evidence. The four critic states drive the "
         "headline. Depth three moves into expanders.",
         "A first time reader can state the recommendation after five seconds"],
        ["U2", "The sequence map",
         "Per codon track with switchable channel, region markers and hover detail, on both tabs.",
         "A flagged construct shows where the problem sits"],
        ["U3", "The redesign diff",
         "Aligned before and after tracks, changed codons highlighted, runs listed at text "
         "resolution with the amino acid shown.",
         "The user can see every change without reading two text boxes"],
        ["U4", "The agents, shown",
         "Agent strip at depth two, the flow diagram in a How this works expander, per agent "
         "detail at depth three.",
         "Three agents abstaining reads as design rather than failure"],
        ["U5", "States and polish",
         "Empty, loading and error states. Copy pass. Colour and contrast check against the "
         "glyph fallback.",
         "Every path through the app has a designed screen"],
    ],
    [8 * mm, 26 * mm, 76 * mm, 58 * mm],
)

box("If time is short, stop after U2", [
    "U0 to U2 is the part that changes how the tool feels: a clear answer, evidence on demand, "
    "and a picture of the construct. U3 to U5 extend it. The sequence map is the single element "
    "most likely to make someone describe the tool as professional, because it is the one that "
    "makes it look like it understands sequences rather than spreadsheets.",
])

A(PageBreak())

# ------------------------------------------------------------------ 11
h1("11. What this plan does not change")

bullets([
    "<b>No number changes.</b> Every figure keeps its current value and its current meaning. "
    "This is a presentation plan and it must not become a quiet route to new claims.",
    "<b>No caveat is deleted.</b> They are relocated, attached to their own figures, and in "
    "several cases shortened. If a shortened caveat loses meaning it stays long.",
    "<b>No agent starts speaking.</b> The gates decide that, and three of the four agents will "
    "keep abstaining after this work exactly as they do before it.",
    "<b>No new dependency.</b> Altair and inline SVG are enough for everything described here, "
    "which matters because the application runs on free hosting alongside xgboost, shap, "
    "biopython and ViennaRNA.",
])

# ------------------------------------------------------------------ 12
h1("12. Open decisions for Udeepa")

table(
    ["", "Decision", "Options", "Recommendation"],
    [
        ["D1", "What is the headline",
         "The critic's verdict state, or the risk percentage as now",
         "The verdict state. A percentage is an input to a decision, not the decision."],
        ["D2", "Keep the tier midpoint",
         "Keep at depth three, or remove",
         "Remove. It is a restatement of the tier in units that do not exist."],
        ["D3", "Theme",
         "Keep the current dark theme, or offer light as the default",
         "Keep dark, and fix the contrast of the muted captions, which currently sit below a "
         "comfortable ratio."],
        ["D4", "Where the agents live",
         "A strip inside Diagnosis, or a third tab of their own",
         "Inside Diagnosis. A separate tab would make the architecture feel optional."],
        ["D5", "The heuristic diagnostics",
         "Inline at depth two, or at depth three with the rest of the evidence",
         "Depth three, with the GRAVY rule promoted to depth one only in the not answerable "
         "state, where it is the only thing left."],
        ["D6", "Scope",
         "U0 to U2, or the whole sequence",
         "Decide after U2 is on screen, since the map changes what the rest should look like."],
    ],
    [8 * mm, 30 * mm, 68 * mm, 62 * mm],
)

box("One thing I would like settled before starting", [
    "D1 decides the layout of every screen in the application, because the headline determines "
    "what the rest of the page is arranged around. The other five can be answered as the work "
    "reaches them.",
])


# ------------------------------------------------------------------ doc
def on_body(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(21 * mm, 13 * mm, "SynthBio, interface plan, 31 October 2026")
    canvas.drawRightString(A4[0] - 21 * mm, 13 * mm, str(doc.page))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(21 * mm, 16.5 * mm, A4[0] - 21 * mm, 16.5 * mm)
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, A4[1] - 10 * mm, A4[0], 10 * mm, stroke=0, fill=1)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 10 * mm, 56 * mm, 10 * mm, stroke=0, fill=1)
    canvas.restoreState()


doc = BaseDocTemplate(
    OUT, pagesize=A4,
    leftMargin=21 * mm, rightMargin=21 * mm,
    topMargin=20 * mm, bottomMargin=20 * mm,
    title="SynthBio interface plan",
    author="SynthBio project",
    subject="A plan for the SynthBio interface",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([
    PageTemplate(id="cover", frames=[frame], onPage=on_cover),
    PageTemplate(id="body", frames=[frame], onPage=on_body),
])
doc.build(story)
print("wrote", OUT)
