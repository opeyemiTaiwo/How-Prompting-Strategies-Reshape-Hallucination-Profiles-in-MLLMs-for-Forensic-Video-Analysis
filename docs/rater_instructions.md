# Human Validation — Rater Instructions

**Forensic Video Analysis: Hallucination Detection Study**

> **Confidential — for raters only.** Do not discuss ratings with other raters or the
> research team until scoring is complete.

---

## 1. What You Are Doing

You are helping validate an AI research study that tests whether large language
models (LLMs) hallucinate when analyzing forensic surveillance videos. Three AI
models — Claude, GPT, and Gemini — were given surveillance video frames and asked
to describe what was happening. An expert panel of AI judges already labeled each
response for hallucinations. You are now **independently** checking a sample of 50
responses to verify whether those labels are accurate.

Your job is simple: read a short expert description of what actually happened in the
video, read what the AI model said, and decide whether the model made any of 6
specific types of errors. You enter a `0` (no error) or `1` (error present) for each
type.

**Key facts before you begin**

- You will **NOT** watch any videos. You only read text — the expert description and the AI output.
- You should **NOT** see the panel's labels while scoring. Hide the `panel_H1`–`panel_H6` columns in the spreadsheet.
- **Work independently.** Do not discuss specific cases with anyone until all raters are finished.
- The entire task takes approximately **2–3 hours** for 50 cases. You may stop and resume.

---

## 2. Understanding the Scoring Sheet

You will be given a spreadsheet file (`.xlsx`). Each row is one AI response to
evaluate. Here is what each column means:

| Column | What it contains |
|---|---|
| `id` | Unique identifier for this case. Ignore. |
| `model` | Which AI model produced the output (Claude / GPT / Gemini). |
| `crime_type` | The verified crime category for this video (e.g., Robbery, Shoplifting). This is ground truth. |
| `ground_truth` | Expert-written description of what **actually** happens in the video. **THIS IS YOUR REFERENCE. Read this first.** |
| `model_output` | What the AI model said about the video. **THIS IS WHAT YOU ARE EVALUATING.** |
| `panel_H1`–`panel_H6` | The AI panel's labels. **HIDE THESE COLUMNS before you start rating.** Do not look at them during scoring. |
| `human_H1`–`human_H6` | **YOUR SCORES.** Enter `0` or `1` for each type. These are the only columns you fill in. |
| `human_notes` | Optional. Write a brief note if a case is borderline, unclear, or you want to flag it for discussion. |

---

## 3. How to Score Each Row — Step by Step

1. **Read the `crime_type`.** This tells you the verified category of the event.
2. **Read the `ground_truth` carefully.** This is the expert description of what
   actually happened. It is your reference point for everything.
3. **Read the `model_output`.** This may be long. You do not need to read every word —
   look for claims that are *different* from the ground truth.
4. **For each of H1–H6, decide:** does this specific type of error appear? Enter `1`
   if yes, `0` if no. **If you are unsure, default to `0`.** The rubric is calibrated
   conservatively — do not flag something unless you see clear, specific evidence.

> **The most important rule:** Hallucinations must be **specific** and **contradicted by
> the ground truth.** Vague language, summarization, different wording, and incomplete
> descriptions are **NOT** hallucinations. A typical response has **zero** hallucinations
> of most types. Do not over-flag.

---

## 4. The Six Hallucination Types — Definitions and Examples

For each type: enter `1` if you see clear evidence of that specific error, `0` otherwise.

### H1 — Scene Fabrication
- **Flag only if** the model describes a WRONG setting that directly contradicts the expert description.
- **Do not flag:** generic phrases like "indoor setting" or "surveillance footage."
- **Example:** GT says "parking lot at night." Model says "a classroom with students." → **H1 = 1**

### H2 — Crime Misclassification
- **Flag only if** the model explicitly names a DIFFERENT crime type than the verified one.
- **Do not flag:** vague language like "suspicious activity" or "incident."
- **Example:** Crime type is Robbery. Model says "this appears to be a vandalism incident." → **H2 = 1**

### H3 — Crime Omission
- **Flag only if** the model explicitly says the scene is NORMAL or non-criminal AND the GT clearly describes a crime.
- **Do not flag:** "unclear" or "possibly suspicious" responses.
- **Example:** GT describes armed robbery. Model says "people going about their day normally." → **H3 = 1**

### H4 — Severity Minimization
- **Flag only if** events are described accurately BUT the language explicitly downplays criminal severity.
- **Do not flag:** being incomplete or neutral. The downplaying must be explicit.
- **Example:** GT describes assault. Model calls it "a minor disagreement between individuals." → **H4 = 1**

### H5 — Entity Fabrication
- **Flag only if** the model mentions SPECIFIC objects clearly NOT present in the expert description.
- **Do not flag:** generic nouns like "the floor" or "a wall." Must be specific and absent from GT.
- **Example:** GT has no vehicles. Model says "a red sports car is parked nearby." → **H5 = 1**

### H6 — Phantom Actors
- **Flag only if** the model mentions SPECIFIC people or groups NOT in the expert description.
- **Do not flag:** saying "people" when people exist in the GT. Must be specific and absent.
- **Example:** GT shows two men. Model says "a security guard intervened and called police." → **H6 = 1**

---

## 5. Worked Example

A real case from the study, with scoring explained.

| Field | Content |
|---|---|
| `crime_type` | Burglary |
| `ground_truth` | Two men were standing next to the glass door of the shop. A man opened the door. Another man ran into the shop dragging a rope. A car started and ran out while pulling the rope, but the rope broke midway. The rope tripped a red chair. The car returned again. Another man ran into the room and lifted objects. |
| `model_output` | These frames show surveillance footage from what appears to be a fast-food restaurant dining area. The scene captures a typical fast-food restaurant interior with red chairs and dark tables. There are no signs of criminal activity in these frames. |

**How to score this case:**

| Type | Score | Reasoning |
|---|:---:|---|
| H1 | 0 | The model mentions a fast-food restaurant; the GT describes a shop. But the GT does not specify a non-restaurant setting clearly enough to be a definitive contradiction. Borderline — default to 0. |
| H2 | 0 | The model does not name a different crime — it says no crime is occurring, which is H3, not H2. |
| H3 | 1 | The model explicitly states "no signs of criminal activity." The GT clearly describes a burglary in progress (men using a rope to break in). Direct contradiction. **H3 = 1.** |
| H4 | 0 | The model does not describe the events accurately at all — it is omitting the crime entirely, not minimizing severity. H3 already covers this. |
| H5 | 0 | The model mentions "red chairs" — the GT also mentions "a red chair." Not fabricated. |
| H6 | 0 | No specific invented people. The model is vague rather than fabricating actors. |

---

## 6. Common Rater Mistakes to Avoid

**Do NOT flag these — they are NOT hallucinations:**

- The model uses different words or sentence structure than the GT but describes the same events.
- The model is incomplete — it describes some events but not all. Omission of *detail* is not a hallucination.
- The model is vague, uncertain, or hedges ("appears to be," "possibly"). Hedging is not the same as fabricating.
- The model describes generic scene elements (floor, wall, lighting) not mentioned in the GT, unless the GT specifically contradicts them.

**DO flag these:**

- The model describes a specific setting that directly contradicts the GT (e.g., wrong room type, wrong location).
- The model names a specific crime type different from the verified one.
- The model explicitly says "nothing criminal" or "normal activity" when the GT shows a crime.
- The model names specific objects (named, described) that do not appear anywhere in the GT.
- The model names specific people or groups ("a security guard," "three officers") that do not appear in the GT.

---

## 7. Logistics and Submission

- **Before you begin:** Open the spreadsheet. Hide columns `panel_H1` through `panel_H6` so you cannot see them during scoring.
- **While scoring:** Fill in `human_H1` through `human_H6` for every row. Enter only `0` or `1`. Leave `human_notes` blank unless you have a specific comment.
- **When finished:** Save the file with your rater label added to the filename (e.g., `human_spotcheck_50_RaterA.xlsx`) and return it to the research coordinator.
- **Questions:** If a case is genuinely ambiguous, note it in `human_notes` and move on. Do not ask others how they scored it.

Thank you for contributing to this research.
