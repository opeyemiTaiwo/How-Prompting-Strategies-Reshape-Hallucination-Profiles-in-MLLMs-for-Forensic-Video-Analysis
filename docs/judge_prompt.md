# Panel-of-Judges: System Prompt and Labeling Protocol

This document specifies exactly how the three-judge LLM panel labeled each forensic
report for the six hallucination types (H1–H6), so the labeling step can be reproduced.

## Labeling protocol

- Each report is scored by **two** of the three judge models — never the model that
  produced the report (no self-evaluation), which avoids self-preference bias.
- Each judge returns a binary verdict for each of the six hallucination types against
  the UCA expert ground truth.
- The per-type label is the **two-judge agreement** rule used throughout the paper:
  both judges must flag a type for it to count (logical AND); ties are broken
  conservatively toward **no hallucination**.
- Judges were queried at **temperature 0.1** (top-p 0.8, top-k 10) with an 8192-token
  budget, after truncating the ground truth and model output to **1,500** and **3,000**
  characters respectively.

## System prompt (identical for all three judges)

```text
You are an expert forensic video analysis evaluator. Your task is
to compare a model's output against expert ground-truth annotation
and identify any hallucinations.

CRITICAL CALIBRATION RULES:
- Hallucinations are RARE. Most outputs have ZERO hallucinations
  of most types.
- Default to 0 unless STRONG, SPECIFIC evidence in the model
  output contradicts the ground truth.
- Vague descriptions, summarization, or different word choice
  are NOT hallucinations.
- The ground truth describes what ACTUALLY happens in the video.
  The crime_type is the verified category.
- A typical output flags 0 or 1 types. 3+ flags should be RARE.
- False positives are worse than false negatives.
  When in doubt, mark 0.

HALLUCINATION TAXONOMY (strict criteria):

H1 SCENE FABRICATION: Flag ONLY if the model describes a specific
WRONG setting that contradicts the ground truth.

H2 CRIME MISCLASSIFICATION: Flag ONLY if the model explicitly
identifies a DIFFERENT crime type than the verified one.

H3 CRIME OMISSION: Flag ONLY if the model explicitly states the
scene is normal/non-criminal AND the ground truth clearly
describes a crime.

H4 SEVERITY MINIMIZATION: Flag ONLY if the model accurately
describes events but uses language that explicitly downplays
criminal severity.

H5 ENTITY FABRICATION: Flag ONLY if the model mentions SPECIFIC
objects/features clearly NOT in the ground truth.

H6 PHANTOM ACTORS: Flag ONLY if the model mentions SPECIFIC
people/groups NOT in the ground truth.

Respond with ONLY a valid JSON object, no markdown:
{"H1": 0 or 1, "H2": 0 or 1, "H3": 0 or 1, "H4": 0 or 1,
 "H5": 0 or 1, "H6": 0 or 1, "confidence": "high|medium|low",
 "reasoning": "1-2 sentences citing specific evidence"}
```

## Forensic annotation rubric

Expert annotations in the UCA dataset were supplemented with a forensic analysis rubric
designed for court-admissible video evidence documentation. The rubric operationalizes
ten evaluation criteria, each mapped to a legal admissibility concern under the U.S.
Federal Rules of Evidence (FRE) and the *Daubert* standard. Annotators recorded
observable facts — visible actions, actors, objects, temporal sequences, and
environmental context — rather than inferring intent or causation. Deviations from
rubric-structured annotations constitute measurable **fabrication** (H1, H5, H6),
**omission** (H3), or **distortion** (H2, H4).

| Criterion | Admissibility concern | Refs. |
|---|---|---|
| Crime classification and intent detection | Relevance and elements of charged offense | FRE 401 |
| Temporal forensic reconstruction | Timeline authentication and sequence documentation | FRE 901; SWGDE 2024 |
| Subject identification and behavioral analysis | Identification procedures and expert testimony | *Daubert*; FRE 702 |
| Physical evidence documentation | Authentication and chain-of-custody | FRE 901 |
| Violence and weapon analysis | Aggravating factors and degree of force | — |
| Criminal network and coordination analysis | Conspiracy or joint enterprise elements | — |
| Modus operandi documentation | Prior bad acts evidence exceptions | — |
| Scene analysis and environmental context | Foundation for scene reconstruction testimony | FRE 401 |
| Escape route and exit strategy analysis | Premeditation and consciousness of guilt | — |
| Forensic narrative and court readiness | Expert testimony admissibility standards | *Daubert*; FRE 702 |

## Human-validation rubric

Human raters applied the **same H1–H6 criteria** as the judge panel, scoring from text
only (the UCA ground truth and the model output, without watching the source video),
with the panel's labels hidden. See [`rater_instructions.md`](rater_instructions.md) for
the full rater rubric and worked examples.
