"""PDSQI-9 LLM-as-a-Judge evaluator.

Implements the Provider Documentation Summarization Quality Instrument
(PDSQI-9) from Croxford et al. (npj Digital Medicine 2025). This enables
automated evaluation of clinical summaries using the same validated
instrument used by human physician reviewers.

Supports both full-note evaluation and per-section evaluation (for
sections S, I, and E) to catch section-level quality variation that
full-note scoring can miss.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PDSQI9Score(BaseModel):
    """Scores for all 9 PDSQI-9 attributes."""

    cited: int = 0  # 1-5
    accurate: int = 0  # 1-5
    thorough: int = 0  # 1-5
    useful: int = 0  # 1-5
    organized: int = 0  # 1-5
    comprehensible: int = 0  # 1-5
    succinct: int = 0  # 1-5
    synthesized: int = 0  # 1-5
    stigmatizing: bool = False  # binary


class PDSQI9Evaluation(BaseModel):
    """Complete PDSQI-9 evaluation result."""

    scores: PDSQI9Score
    reasoning: dict[str, str]  # attribute → reasoning text
    total_score: float  # mean of 8 Likert attributes (excludes stigmatizing)
    evaluator_model: str


class PDSQI9SectionScore(BaseModel):
    """PDSQI-9 evaluation for a single section."""

    section_key: str  # "S", "I", "E"
    section_label: str
    scores: PDSQI9Score
    reasoning: dict[str, str]
    total_score: float
    attributes_evaluated: list[str]  # which attributes were scored


class PDSQI9FullEvaluation(BaseModel):
    """Combined full-note + per-section PDSQI-9 evaluation."""

    overall: PDSQI9Evaluation
    sections: list[PDSQI9SectionScore] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-section attribute configuration
# ---------------------------------------------------------------------------

# Which PDSQI-9 attributes to score for each section
SECTION_ATTRIBUTES: dict[str, tuple[str, list[str]]] = {
    "S": (
        "Summary of Major Problems and To-Do's",
        ["cited", "accurate", "thorough", "useful", "organized",
         "comprehensible", "succinct", "synthesized"],
    ),
    "I": (
        "ICU Admission Reason & Brief ICU Course",
        ["cited", "accurate", "thorough", "succinct",
         "synthesized", "comprehensible"],
    ),
}


# ---------------------------------------------------------------------------
# Default prompt (enriched with Croxford-style detail)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are a hospitalist (ward attending physician) evaluating an AI-generated \
ICU-to-ward transition brief that you would receive when accepting a patient \
from the ICU. Score the brief using the PDSQI-9 (Provider Documentation \
Summarization Quality Instrument, adapted from Croxford et al., npj Digital \
Medicine 2025). You will be given the original clinical source data and the \
transition brief generated from it. Score it on each of the 9 attributes below.

An "assertion" is a statement that can be single or multiple sentences.

## PDSQI-9 ATTRIBUTES

RENDERING NOTE: In the clinician's view, the parenthetical source tags display \
as compact superscript citations at the end of the sentence or assertion they \
support — often several together. The plain text you are reading shows them \
inline, which makes them look denser than the clinician experiences. For \
example, the plain-text line "Started on BiPAP for hypercapnic respiratory \
failure (resp 1-04 10:00)(vital 1-04 10:00)(lab 1-04 10:00)" appears to the \
clinician as that single sentence followed by three small superscript markers. \
Therefore: (a) tags clustered at the end of the assertion they support ARE \
paired with it — do not penalize this as "grouping" under cited (the grouping \
penalty applies only to citations collected in a block divorced from the \
assertions they support); (b) the number or visual density of citation tags is \
unobtrusive in the clinician's view — do not count tag density against \
succinctness, comprehensibility, or usefulness. This is distinct from genuine \
redundancy in the clinical prose itself, which may still be penalized.

DOCUMENT STRUCTURE: The summary may include system-generated review aids \
labeled "QA ISSUES (Review Required)" and "WARNINGS:". These are automated \
flags rendered separately from the clinical summary (as banners) in the \
clinician's view — they are NOT clinician-authored clinical prose.
- Do not treat their text as clinical assertions: they are never "uncited" or \
"fabricated"/inaccurate claims, and their length does not count toward \
succinctness.
- Judge cited, accurate, succinct, thorough, organized, synthesized, \
comprehensible, and stigmatizing on the clinical-summary sections only.
- You MAY weigh the overall QA/warning burden when judging useful: a large or \
noisy volume of flags can reasonably make the handoff less useful, and a few \
well-targeted flags can aid it — weigh the flags' volume and actionability, \
not their visual prominence.

<cited>
1. Are citations present and appropriate?

NOTE: Citations mean that clinical assertions in the summary reference \
specific data from the source (e.g., specific lab values, vital signs, \
medication doses, dates). Citations should be paired with individual \
assertions, not grouped together in a block. Correct citations reference \
data that actually exists in the source. Incorrect citations reference \
values not present or misattributed from the source.

CITATION FORMAT: This system uses parenthetical source tags as inline \
citations. They appear as "(lab 1/17 08:00)", "(vital 2/15 16:05)", \
"(resp 2/14 11:00)", "(med 1/18 06:00)", "(assess 2/15 11:00)", \
"(code_status 2024-12-16 13:43:00)", "(progress_note 1-05 13:06)", etc. \
Each tag references a specific entry from the source data — either structured \
data (labs, vitals, meds, etc.) OR a clinical note (progress_note, hp_note, \
consults_note, etc.). Note-type tags are valid citations exactly like \
structured-data tags; count them as citations either way.

HOW TO VERIFY A CITATION:
- Structured-data tags (lab, vital, med, resp, assess, code, proc): each \
structured source row carries its canonical tag in a "cite" field. A structured \
citation is correct when it matches a "cite" value present in the source data, \
and incorrect when no source row carries that tag.
- Note-type tags (progress_note, hp_note, consults_note, nursing_note, etc.): \
verify by note TYPE and CONTENT, not by timestamp. A note citation is correct \
when the source clinical notes contain a note of that type whose content \
supports the cited assertion. The time inside a note tag is the note's internal \
anchor and is intentionally not shown per-row in the source; do NOT mark a note \
citation incorrect merely because its timestamp is not found among the source \
rows.

1 = Multiple incorrect citations OR no citations provided at all
2 = One citation incorrect OR citations grouped together and not with \
individual assertions
3 = Citations correct but some assertions are missing a citation
4 = Every assertion correctly cited with some relevance prioritization
5 = Every assertion is correctly cited and prioritized by relevance
</cited>

<accurate>
2. Is the summary accurate in extraction (extractive summarization)?

NOTE: Extraction-based summarization involves selecting and pulling exact \
phrases or sentences directly from the original text without altering the \
wording. The focus is on identifying the most important parts of the text \
and reproducing them verbatim to form the summary.

Incorrect information can be a result of fabrication or falsification:
- FABRICATION: The response contains entirely made-up information or data \
and includes plausible but non-existent facts in the summary.
- FALSIFICATION: The response contains distorted information and includes \
changing critical details of facts so they are no longer true from the \
source notes.
- Examples of problematic assertions: It is not in the note; it was correct \
at one point but not at the time of summarization; a given assertion was \
changed to a different status (e.g., given symptoms of COVID but patient \
ended up not having COVID, however LLM generates COVID as a diagnosis).

IMPORTANT: Something can be an incorrect statement by the provider in the \
note (not clinically plausible) but if the LLM summarizes the same statement \
from the provider then it is NOT a fabrication or falsification.

1 = Multiple major errors with overt falsifications or fabrications
2 = A major error in assertion occurs with an overt falsification or fabrication
3 = At least one assertion contains a misalignment that is stated from a \
source note but the wrong context, including incorrect specificity in \
diagnosis or treatment
4 = At least one assertion is misaligned to the provider's source or timing \
but still factual in diagnosis, treatment, etc.
5 = All assertions can be traced back to the notes or structured data
</accurate>

<thorough>
3. Is the summary thorough without any omissions?

NOTE: Identify any pertinent or potentially pertinent omissions:
- PERTINENT OMISSIONS: Essential information required for the specific use \
case or intended provider, where missing details could directly impact \
patient care decisions (i.e., information that would prompt an immediate \
or future action).
- POTENTIALLY PERTINENT OMISSIONS: Relevant details for clinical understanding \
that may not directly influence the current use case but would still be \
useful to know.
- Example: A consultant recommending a DEXA scan to the family medicine \
physician is pertinent (action needed), whereas a consultant ordering a \
DEXA scan themselves is potentially pertinent (useful to know but does not \
require immediate action by the primary provider).

The summary should thoroughly cover all critical patient issues.

1 = More than one pertinent omission occurs
2 = One pertinent and multiple potentially pertinent omissions occur
3 = Only one pertinent omission occurs
4 = Some potentially pertinent omissions occur
5 = No pertinent or potentially pertinent omissions occur
</thorough>

<useful>
4. Is the summary useful?

NOTE: All the information should be useful to the target provider/intended \
audience. The summary should be extremely relevant, providing valuable \
information and/or analysis. Evaluate whether assertions are pertinent to \
the target user, and whether the level of detail is appropriate (not too \
detailed, not too sparse).

1 = No assertions are pertinent to the target user
2 = Some assertions are pertinent to the target user
3 = Assertions are pertinent to target provider but level of detail is \
inappropriate (too detailed or not detailed enough)
4 = Not adding any non-pertinent assertions but some assertions are only \
potentially pertinent to target user
5 = Not adding any non-pertinent assertions and level of detail is \
appropriate to the targeted user
</useful>

<organized>
5. Is the summary organized?

NOTE: The summary should be well-formed and structured in a way that helps \
the reader understand the patient's clinical course. Organization includes \
both logical ordering (temporal sequence) and logical grouping (by \
systems/problem-based). A score of 3 means no change from the original \
input ordering. Higher scores require active reorganization.

1 = All assertions presented out of order and groupings incoherent \
(completely disorganized)
2 = Some assertions presented out of order OR grouping incoherent
3 = No change in order or grouping (temporal or systems/problem based) \
from original input
4 = Logical order or grouping (temporal or systems/problem based) for all \
assertions but not both
5 = All assertions made with logical order and grouping (temporal or \
systems/problem based) — completely organized
</organized>

<comprehensible>
6. Is the summary comprehensible with clarity of language?

NOTE: The summary should be clear, without ambiguity or sections that are \
difficult to understand. Evaluate whether the word choice and sentence \
structure are appropriate for the target user. A score of 3 means unchanged \
from input with missed opportunities for simplification. Higher scores show \
active improvement toward plain, well-structured language.

1 = Words in sentence structure are overly complex, inconsistent, with \
terminology that is unfamiliar to the target user
2 = Any use of overly complex, inconsistent, or terminology that is \
unfamiliar to target user
3 = Unchanged choice of words from input with inclusion of overly complex \
terms when there was opportunity for improvement
4 = Some inclusion of change in structure and terminology towards improvement
5 = Plain language completely familiar and well-structured to target user
</comprehensible>

<succinct>
7. Is the summary succinct with economy of language?

NOTE: The summary should be brief, to the point, and without redundancy. \
Redundancy can be syntactic (repeating the same words/phrases) or semantic \
(repeating the same meaning in different words, i.e., contextual semantic \
redundancy). A succinct summary captures all assertions with the fewest \
words possible.

1 = Too wordy across all assertions with redundancy in syntax and semantic
2 = More than one assertion has contextual semantic redundancy
3 = At least one assertion has contextual semantic redundancy or multiple \
syntactic assertions
4 = No syntax redundancy in assertions and at least one could have been \
shorter in contextualized semantics
5 = All assertions are captured with fewest words possible and without any \
redundancy in syntax or semantics
</succinct>

<synthesized>
8. Is there a need for abstraction in the summary? (Synthesized)

NOTE: Abstraction-based summarization goes beyond simply extracting exact \
phrases from the original text. It involves paraphrasing and synthesizing \
information to produce new sentences that capture the core meaning — similar \
to how a human might read a passage and restate key ideas in their own words.

For example, instead of listing "HbA1c of 9.2%, prescribed metformin, \
reported not following dietary recommendations," an abstraction model might \
summarize as "poorly controlled diabetes due to suboptimal adherence to \
treatment and lifestyle changes."

Evaluate how well the summary captures the essence of the original content, \
whether paraphrased statements are accurate, and whether the model infers \
higher-level concepts from specific details.

If abstraction is not needed, score as NA. If abstraction is applicable:

1 = Incorrect reasoning or grouping in the connections between the assertions
2 = Abstraction performed when not needed OR groupings were made between \
assertions that were accurate but not appropriate
3 = Assertions are independently stated without any reasoning or groups over \
the assertions when there could have been one (missed opportunity to abstract)
4 = Groupings of assertions occur into themes but limited to fully formed \
reasoning for a final, clinically relevant diagnosis or treatment
5 = Goes beyond relevant groups of events and generates reasoning over the \
events into a summary that is fully integrated for an overall clinical \
synopsis with prioritized information
</synthesized>

<stigmatizing>
9. Is there presence of stigmatizing language?

NOTE: Evaluate using these guidelines (adapted from CHCS Words Matter):
- Refrain from using discrediting or exaggerated words such as "claims", \
"insists", or "reportedly"
- Watch for quotes that imitate disbelief or perpetuate stereotypes
- Watch for judgment or discrediting words that suggest doubt (e.g., "He \
claims he is in pain", "She reportedly had two seizures")
- Prefer person-first language: "patient with diabetes" instead of "diabetic \
patient"
- Minimize blame, labeling, and judgment: "She is not tolerating oxygen" \
instead of "She is refusing to wear oxygen"
- Avoid language that a person "is" the problem rather than "has" a problem: \
use "person with substance use disorder" instead of "addict", "alcoholic", \
"drug abuser"
- The term "abuse" has high association with negative judgments and punishment
- Watch for terms that evoke negative and punitive implicit cognitions such \
as "dirty urine"

Score both the presence in the source note AND in the summary:

true = Stigmatizing language is present in the summary
false = No stigmatizing language in the summary
</stigmatizing>

## OUTPUT FORMAT
Return a JSON object:
{
  "scores": {
    "cited": <1-5>,
    "accurate": <1-5>,
    "thorough": <1-5>,
    "useful": <1-5>,
    "organized": <1-5>,
    "comprehensible": <1-5>,
    "succinct": <1-5>,
    "synthesized": <1-5>,
    "stigmatizing": <true/false>
  },
  "reasoning": {
    "cited": "<justification with specific examples of cited/uncited assertions>",
    "accurate": "<justification noting any fabrications, falsifications, or misalignments found>",
    "thorough": "<justification identifying any pertinent or potentially pertinent omissions>",
    "useful": "<justification on pertinence of assertions and detail level appropriateness>",
    "organized": "<justification on ordering and grouping quality>",
    "comprehensible": "<justification on language clarity and terminology appropriateness>",
    "succinct": "<justification noting any syntactic or semantic redundancy>",
    "synthesized": "<justification with examples of abstraction/synthesis or missed opportunities>",
    "stigmatizing": "<justification noting any stigmatizing terms found, referencing guidelines above>"
  }
}
"""

_LIKERT_ATTRS = [
    "cited", "accurate", "thorough", "useful",
    "organized", "comprehensible", "succinct", "synthesized",
]


# ---------------------------------------------------------------------------
# Per-section evaluation prompt
# ---------------------------------------------------------------------------

_SECTION_SYSTEM_PROMPT = """\
You are a hospitalist (ward attending physician) evaluating a SINGLE SECTION \
of an AI-generated ICU-to-ward transition brief. You will be given the \
original clinical source data and one specific section from the brief. \
Score ONLY this section on the specified attributes.

Do NOT evaluate the brief as a whole — focus exclusively on the content \
within this one section.

{attribute_definitions}

## OUTPUT FORMAT
Return a JSON object:
{{
  "scores": {{
    {score_keys}
  }},
  "reasoning": {{
    {reasoning_keys}
  }}
}}
"""

# Attribute definitions for per-section prompts (reused from full prompt).
# DUP NOTE (v2, 2026-06-17): the judge-calibration runner uses the full
# _DEFAULT_SYSTEM_PROMPT path (PDSQI9Evaluator.evaluate), NOT this section path,
# so the v2 cited fixes (note-type tags + cite-field verification) and the
# RENDERING NOTE / DOCUMENT STRUCTURE blocks were applied there only. If the
# section-scoring path is ever put into use, mirror those edits into the
# "cited" definition below + add the two preamble blocks to _SECTION_SYSTEM_PROMPT.
_ATTRIBUTE_DEFS = {
    "cited": """\
<cited>
Are citations present and appropriate in this section?
NOTE: Citations mean assertions reference specific data from the source. \
Citations should be paired with individual assertions, not grouped. \
Incorrect citations reference values not present or misattributed.
CITATION FORMAT: This system uses parenthetical source tags as inline \
citations, e.g. "(lab 1/17 08:00)", "(vital 2/15 16:05)", \
"(resp 2/14 11:00)", "(assess 2/15 11:00)". These ARE citations — \
count them as such.
1 = Multiple incorrect citations OR no citations provided
2 = One citation incorrect OR citations grouped and not with individual assertions
3 = Citations correct but some assertions missing a citation
4 = Every assertion correctly cited with some relevance prioritization
5 = Every assertion correctly cited and prioritized by relevance
</cited>""",
    "accurate": """\
<accurate>
Is this section accurate in extraction?
NOTE: Fabrication = entirely made-up information. Falsification = distorted \
information changing critical details. If the LLM summarizes an incorrect \
statement from the provider's note, that is NOT fabrication/falsification.
1 = Multiple major errors with overt falsifications or fabrications
2 = A major error with an overt falsification or fabrication
3 = At least one misalignment from a source note but wrong context
4 = At least one misalignment to source/timing but still factual
5 = All assertions can be traced back to the notes or structured data
</accurate>""",
    "thorough": """\
<thorough>
Is this section thorough without omissions?
NOTE: Pertinent omissions = essential info where missing details could \
directly impact care. Potentially pertinent omissions = relevant but not \
directly impacting current care decisions.
1 = More than one pertinent omission
2 = One pertinent and multiple potentially pertinent omissions
3 = Only one pertinent omission
4 = Some potentially pertinent omissions
5 = No pertinent or potentially pertinent omissions
</thorough>""",
    "useful": """\
<useful>
Is this section useful to the target provider?
1 = No assertions pertinent to target user
2 = Some assertions pertinent to target user
3 = Assertions pertinent but level of detail inappropriate
4 = No non-pertinent assertions but some only potentially pertinent
5 = No non-pertinent assertions and detail level appropriate
</useful>""",
    "organized": """\
<organized>
Is this section organized?
NOTE: Evaluate both logical ordering (temporal) and grouping (systems/problem).
1 = All assertions out of order and groupings incoherent
2 = Some assertions out of order OR grouping incoherent
3 = No change in order or grouping from original input
4 = Logical order or grouping for all assertions but not both
5 = All assertions with logical order and grouping — completely organized
</organized>""",
    "comprehensible": """\
<comprehensible>
Is this section comprehensible with clarity of language?
NOTE: Evaluate word choice and structure appropriateness for target user.
1 = Overly complex, inconsistent, terminology unfamiliar to target user
2 = Any use of overly complex or unfamiliar terminology
3 = Unchanged from input with missed opportunities for improvement
4 = Some improvement in structure and terminology
5 = Plain language completely familiar and well-structured to target user
</comprehensible>""",
    "succinct": """\
<succinct>
Is this section succinct with economy of language?
NOTE: Evaluate both syntactic redundancy (repeating words/phrases) and \
semantic redundancy (repeating meaning in different words).
1 = Too wordy across all assertions with syntactic and semantic redundancy
2 = More than one assertion has contextual semantic redundancy
3 = At least one assertion has semantic redundancy or multiple syntactic issues
4 = No syntax redundancy, at least one could be shorter semantically
5 = All assertions captured with fewest words, no redundancy
</succinct>""",
    "synthesized": """\
<synthesized>
Does this section synthesize/abstract information appropriately?
NOTE: Abstraction = paraphrasing and synthesizing into new sentences that \
capture core meaning, going beyond extracting exact phrases.
1 = Incorrect reasoning or grouping in connections between assertions
2 = Abstraction when not needed OR accurate but inappropriate groupings
3 = Assertions independently stated, missed opportunity to abstract
4 = Groupings into themes but limited fully formed reasoning
5 = Fully integrated synopsis with reasoning over events and prioritized info
</synthesized>""",
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class PDSQI9Evaluator:
    """Evaluate a clinical summary using the PDSQI-9 instrument."""

    def __init__(self, settings: Settings):
        from icu_pause.eval import create_eval_llm

        self.llm: BaseLLM = create_eval_llm(
            settings, settings.pdsqi9_llm_provider, settings.pdsqi9_llm_model,
        )
        self.system_prompt = self._load_prompt(settings)

    @staticmethod
    def _load_prompt(settings: Settings) -> str:
        path = Path(settings.prompts_dir) / "pdsqi9_evaluator.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            return data.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        return _DEFAULT_SYSTEM_PROMPT

    def _parse_scores(self, response: str) -> tuple[dict, dict]:
        """Parse LLM response into scores dict and reasoning dict.

        Handles common LLM JSON issues: code fences, control chars,
        truncated JSON, and malformed strings.
        """
        import re
        from icu_pause.llm.provider import _clean_llm_output

        # DeepSeek-R1-Distill (and other small models) sometimes paraphrase the
        # requested PDSQI-9 attribute keys under synonyms instead of the rubric
        # names, which silently drops those attributes to 0. The drift is
        # stochastic (vLLM temp-0 isn't bit-deterministic) and concentrates on
        # harder/longer briefs. Map the synonyms back to canonical keys.
        aliases = {"clear": "comprehensible", "concise": "succinct",
                   "relevant": "useful"}

        def _canon(d):
            if not isinstance(d, dict):
                return d
            return {aliases.get(k, k): v for k, v in d.items()}

        cleaned = _clean_llm_output(response)

        # Try direct parse first
        try:
            parsed = json.loads(cleaned, strict=False)
            return _canon(parsed.get("scores", {})), parsed.get("reasoning", {})
        except json.JSONDecodeError:
            pass

        # Fallback 1: Try to extract just the scores block with regex
        scores = {}
        reasoning = {}

        # Extract scores
        scores_match = re.search(
            r'"scores"\s*:\s*\{([^}]+)\}', cleaned, re.DOTALL
        )
        if scores_match:
            try:
                scores = json.loads("{" + scores_match.group(1) + "}", strict=False)
            except json.JSONDecodeError:
                # Extract individual score values
                for attr in ["cited", "accurate", "thorough", "useful",
                             "organized", "comprehensible", "succinct", "synthesized"]:
                    m = re.search(rf'"{attr}"\s*:\s*(\d)', cleaned)
                    if m:
                        scores[attr] = int(m.group(1))
                for syn, canon in aliases.items():
                    if canon not in scores:
                        m = re.search(rf'"{syn}"\s*:\s*(\d)', cleaned)
                        if m:
                            scores[canon] = int(m.group(1))
                stig = re.search(r'"stigmatizing"\s*:\s*(true|false)', cleaned)
                if stig:
                    scores["stigmatizing"] = stig.group(1) == "true"

        # Extract reasoning (best effort — may be truncated)
        reasoning_match = re.search(
            r'"reasoning"\s*:\s*\{(.+)', cleaned, re.DOTALL
        )
        if reasoning_match:
            # Try to parse, but reasoning text often has unescaped chars
            try:
                reasoning = json.loads("{" + reasoning_match.group(1), strict=False)
            except json.JSONDecodeError:
                # Extract individual reasoning strings
                for attr in ["cited", "accurate", "thorough", "useful",
                             "organized", "comprehensible", "succinct",
                             "synthesized", "stigmatizing"]:
                    m = re.search(
                        rf'"{attr}"\s*:\s*"((?:[^"\\]|\\.)*)"',
                        reasoning_match.group(1),
                    )
                    if m:
                        reasoning[attr] = m.group(1)

        if scores:
            logger.info(f"PDSQI-9: recovered scores via regex fallback: {list(scores.keys())}")
        return _canon(scores), reasoning

    def evaluate(
        self,
        source_notes: str,
        summary: str,
    ) -> PDSQI9Evaluation:
        """Score a clinical summary using the PDSQI-9 instrument.

        Args:
            source_notes: Original clinical data (JSON string or plain text).
            summary: The clinical summary to evaluate.

        Returns:
            PDSQI9Evaluation with all 9 attribute scores and reasoning.
        """
        user_message = (
            f"## SOURCE CLINICAL DATA\n{source_notes}\n\n"
            f"## CLINICAL SUMMARY TO EVALUATE\n{summary}\n\n"
            f"Evaluate the summary using the PDSQI-9 rubric. Return the JSON.\n\n"
            f"IMPORTANT: When evaluating thoroughness, only count as omissions "
            f"information that IS present in the source data but MISSING from "
            f"the summary. If certain data types (e.g., pending cultures, code "
            f"status, procedures) are not present in the source data, the "
            f"summary correctly omitting them is NOT an omission."
        )

        response = None
        try:
            response = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
            )
            scores_data, reasoning = self._parse_scores(response)

            # Coerce non-integer values (e.g., "NA", null) to 0
            def _to_int(v, default=0):
                if isinstance(v, int):
                    return v
                try:
                    return int(v)
                except (ValueError, TypeError):
                    return default

            scores = PDSQI9Score(
                cited=_to_int(scores_data.get("cited", 0)),
                accurate=_to_int(scores_data.get("accurate", 0)),
                thorough=_to_int(scores_data.get("thorough", 0)),
                useful=_to_int(scores_data.get("useful", 0)),
                organized=_to_int(scores_data.get("organized", 0)),
                comprehensible=_to_int(scores_data.get("comprehensible", 0)),
                succinct=_to_int(scores_data.get("succinct", 0)),
                synthesized=_to_int(scores_data.get("synthesized", 0)),
                stigmatizing=bool(scores_data.get("stigmatizing", False)),
            )
        except Exception as e:
            raw_preview = (response[:500] + "...") if response and len(response) > 500 else response
            logger.warning(f"PDSQI-9 evaluation failed: {e}\nRaw LLM response: {raw_preview}")
            scores = PDSQI9Score()
            reasoning = {attr: f"Evaluation failed: {e}" for attr in _LIKERT_ATTRS}

        likert_values = [
            getattr(scores, attr) for attr in _LIKERT_ATTRS
            if 1 <= getattr(scores, attr) <= 5
        ]
        total = sum(likert_values) / len(likert_values) if likert_values else 0.0

        return PDSQI9Evaluation(
            scores=scores,
            reasoning=reasoning,
            total_score=round(total, 2),
            evaluator_model=self.llm.last_usage.model,
        )

    def evaluate_section(
        self,
        section_key: str,
        section_content: str,
        source_notes: str,
    ) -> PDSQI9SectionScore:
        """Score a single section using a subset of PDSQI-9 attributes.

        Args:
            section_key: ICU-PAUSE section key (e.g., "S", "I", "E").
            section_content: The text content of this section.
            source_notes: Original clinical data for grounding.

        Returns:
            PDSQI9SectionScore with relevant attribute scores.
        """
        if section_key not in SECTION_ATTRIBUTES:
            raise ValueError(f"Per-section scoring not configured for '{section_key}'")

        section_label, attrs = SECTION_ATTRIBUTES[section_key]

        # Build section-specific prompt
        attr_defs = "\n\n".join(_ATTRIBUTE_DEFS[a] for a in attrs)
        score_keys = ",\n    ".join(f'"{a}": <1-5>' for a in attrs)
        reasoning_keys = ",\n    ".join(f'"{a}": "<justification>"' for a in attrs)

        system_prompt = _SECTION_SYSTEM_PROMPT.format(
            attribute_definitions=attr_defs,
            score_keys=score_keys,
            reasoning_keys=reasoning_keys,
        )

        user_message = (
            f"## SOURCE CLINICAL DATA\n{source_notes}\n\n"
            f"## SECTION: {section_key} — {section_label}\n{section_content}\n\n"
            f"Evaluate ONLY this section on the specified attributes. Return the JSON."
        )

        try:
            response = self.llm.invoke(
                system=system_prompt,
                user=user_message,
            )
            scores_data, reasoning = self._parse_scores(response)
            scores = PDSQI9Score(
                **{a: scores_data.get(a, 0) for a in attrs},
            )
        except Exception as e:
            logger.warning(f"PDSQI-9 section evaluation failed for {section_key}: {e}")
            scores = PDSQI9Score()
            reasoning = {a: f"Evaluation failed: {e}" for a in attrs}

        likert_values = [
            getattr(scores, a) for a in attrs
            if 1 <= getattr(scores, a) <= 5
        ]
        total = sum(likert_values) / len(likert_values) if likert_values else 0.0

        return PDSQI9SectionScore(
            section_key=section_key,
            section_label=section_label,
            scores=scores,
            reasoning=reasoning,
            total_score=round(total, 2),
            attributes_evaluated=attrs,
        )

    def evaluate_full(
        self,
        source_notes: str,
        summary: str,
        sections: dict[str, str] | None = None,
    ) -> PDSQI9FullEvaluation:
        """Run full-note evaluation + per-section scoring for configured sections.

        Args:
            source_notes: Original clinical data.
            summary: Full note text for overall evaluation.
            sections: Dict of section_key -> section_content. If provided,
                per-section scoring is run on S, I, and E.

        Returns:
            PDSQI9FullEvaluation with overall + section scores.
        """
        # Full-note evaluation
        overall = self.evaluate(source_notes, summary)

        # Per-section evaluations
        section_scores: list[PDSQI9SectionScore] = []
        if sections:
            for key in SECTION_ATTRIBUTES:
                content = sections.get(key, "")
                if content and content != "Not enough information from structured data.":
                    try:
                        score = self.evaluate_section(key, content, source_notes)
                        section_scores.append(score)
                        logger.info(
                            f"PDSQI-9 section {key}: {score.total_score}/5.0"
                        )
                    except Exception as e:
                        logger.warning(f"Per-section PDSQI-9 failed for {key}: {e}")

        return PDSQI9FullEvaluation(
            overall=overall,
            sections=section_scores,
        )
