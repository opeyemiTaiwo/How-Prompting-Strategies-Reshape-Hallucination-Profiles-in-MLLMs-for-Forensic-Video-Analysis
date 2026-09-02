"""
Crime Video Analysis — Prompting Techniques (Consolidated)
==========================================================
Eight prompting techniques for multimodal (frames + text) crime-video
classification, gathered into one file.

  1. Zero-Shot
  2. Self-Consistency
  3. Meta-Prompting
  4. Chain-of-Thought
  5. Sequential (multi-turn)
  6. ReAct (Reasoning + Acting)
  7. Least-to-Most
  8. Iterative Refinement

WHAT WAS REMOVED
----------------
All vendor API wiring (the model SDK clients, request payloads, retry
shims) and every hard-coded model-name string have been stripped out.
Every technique now routes through a single function:

        call_model(messages, images=None)

Implement that ONE function with whatever backend you like (a local
model, a hosted API, etc.). Everything below is backend-agnostic and
focuses purely on the *prompting logic* of each technique.

`messages` is a list of {"role": "...", "content": "..."} dicts.
`images`  is an optional list of base64-encoded image strings to attach
          to the LAST user message.

Return value of call_model() must be the model's text response (str).
"""

import os
import re
import json
import time
import base64
from collections import Counter


# ================================================================
#  CONFIGURATION  -  edit these paths to match your machine
# ================================================================
FRAMES_DIR     = "FRAMES"        # root holding <CrimeType>/<VideoStem>/frame_*.jpg
SAVE_DIR       = "RESULTS"       # where JSON outputs are written
FRAME_EXT      = ".jpg"
FRAME_INTERVAL = 1               # 1 = every frame; 2 = every other; etc.
BATCH_SIZE     = 20              # frames per model call

CRIME_LABELS = ("Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
                "Fighting", "RoadAccidents", "Robbery", "Shooting",
                "Shoplifting", "Stealing", "Vandalism", "Normal")


# ================================================================
#  MODEL CALL  -  IMPLEMENT THIS (backend removed on purpose)
# ================================================================
def call_model(messages, images=None):
    """
    Single entry point for every technique below.

    Plug in your own backend here. `messages` is a chat-style list of
    {"role", "content"} dicts; `images` (optional) is a list of
    base64-encoded image strings to attach to the final user turn.

    Must return the model's text response as a string.
    """
    raise NotImplementedError(
        "Implement call_model() with your own model backend. "
        "It receives a chat-style `messages` list (and optional `images`) "
        "and must return the model's text reply as a string."
    )


# ================================================================
#  FRAME HELPERS
# ================================================================
def extract_frame_number(filename):
    """Return integer index from frame_00042.jpg style names."""
    name = os.path.splitext(filename)[0]
    m = re.search(r"frame[_\-]?(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def discover_all_videos_and_frames(frames_dir=None):
    """
    Walk FRAMES_DIR and return a manifest of all extracted videos.

    Expected layout:
        FRAMES_DIR/
            Abuse/
                Abuse001/
                    frame_00001.jpg
                    frame_00002.jpg ...
            Arrest/ ...

    Returns dict "<CrimeType>_<VideoStem>" -> {
        "crime_type", "video_id", "frames_dir", "frames"
    }
    """
    if frames_dir is None:
        frames_dir = FRAMES_DIR
    print("\n=== DISCOVERING FRAMES ===")
    print(f"    Root : {frames_dir}")
    all_videos = {}
    if not os.path.isdir(frames_dir):
        print(f"  ERROR: FRAMES_DIR not found: {frames_dir}")
        return all_videos
    crime_types = sorted([
        d for d in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, d)) and not d.startswith("_")
    ])
    print(f"  Categories : {crime_types}")
    for crime_type in crime_types:
        cat_dir = os.path.join(frames_dir, crime_type)
        video_stems = sorted([
            d for d in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, d))
        ])
        print(f"    {crime_type:20s}: {len(video_stems)} videos")
        for video_stem in video_stems:
            vdir = os.path.join(cat_dir, video_stem)
            frame_files = sorted(
                [ff for ff in os.listdir(vdir) if ff.lower().endswith(FRAME_EXT)],
                key=extract_frame_number,
            )
            if not frame_files:
                print(f"      WARNING: no {FRAME_EXT} frames in {vdir} - skipping")
                continue
            key = f"{crime_type}_{video_stem}"
            all_videos[key] = {
                "crime_type": crime_type,
                "video_id":   video_stem,
                "frames_dir": vdir,
                "frames":     frame_files,
            }
    print(f"  Total videos ready: {len(all_videos)}")
    return all_videos


def load_frames_for_video(video_info, frame_interval=1):
    """Read every frame_interval-th frame, base64-encode, return {filename: b64}."""
    vdir        = video_info["frames_dir"]
    frame_files = video_info["frames"]
    video_id    = video_info["video_id"]
    selected    = frame_files[::frame_interval]
    label = "ALL" if frame_interval == 1 else f"every {frame_interval}th"
    print(f"  Loading {len(selected)} frames ({label}) for {video_id} ...")
    frames_data = {}
    for ff in selected:
        fp = os.path.join(vdir, ff)
        try:
            with open(fp, "rb") as fh:
                frames_data[ff] = base64.b64encode(fh.read()).decode("utf-8")
        except Exception as e:
            print(f"    ERROR loading {ff}: {e}")
    print(f"  Loaded {len(frames_data)}/{len(selected)} frames OK")
    return frames_data


def batch_frame_names(frames_data, batch_size=BATCH_SIZE):
    """Sort frame names by index and split into batches of batch_size."""
    frame_names = sorted(frames_data.keys(), key=extract_frame_number)
    return [frame_names[i:i + batch_size]
            for i in range(0, len(frame_names), batch_size)]


def images_for(frames_data, frame_names):
    """Return the base64 images for a list of frame names, in order."""
    return [frames_data[f] for f in frame_names if f in frames_data]


def save_results(results, filename):
    """Write a results dict to SAVE_DIR/filename as JSON."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, filename), "w") as f:
        json.dump(results, f, indent=2)


# Shared system framing used by several techniques.
FORENSIC_SYSTEM_PROMPT = (
    "You are an expert forensic video analyst specializing in crime detection "
    "and security surveillance. You analyze video frames methodically, noting "
    "details about people, actions, environment, and potential criminal activity. "
    "Be precise, objective, and thorough."
)


# ================================================================
#  1. ZERO-SHOT
# ================================================================
class ZeroShotAnalyzer:
    """
    Zero-shot crime analysis. Sends frames in batches, collects per-batch
    observations, then synthesizes a final structured report. No examples,
    no special reasoning scaffold — just a direct instruction.
    """
    def analyze_frames(self, frames_data, video_id, crime_type):
        print(f"\n  [Zero-Shot] {video_id} | {len(frames_data)} frames ...")
        batches = batch_frame_names(frames_data)
        batch_summaries = []

        for idx, batch in enumerate(batches, 1):
            print(f"    Batch {idx}/{len(batches)} ({len(batch)} frames) ...")
            prompt = (
                f"Analyze this batch of security camera frames (Batch {idx} of {len(batches)}).\n"
                "Describe: (1) What is happening, (2) Suspicious behaviour if any, "
                f"(3) Probable crime type from: {', '.join(CRIME_LABELS)}. "
                "(4) Confidence 0-100%."
            )
            summary = call_model(
                [{"role": "user", "content": prompt}],
                images=images_for(frames_data, batch),
            )
            batch_summaries.append(summary)
            print(f"      Response: {len(summary)} chars")

        print("    Synthesizing batches ...")
        formatted = "\n\n".join(f"--- Batch {i+1} ---\n{s}"
                                for i, s in enumerate(batch_summaries))
        synth_prompt = (
            f"You analyzed {len(batches)} batches of {len(frames_data)} security camera frames.\n\n"
            f"Per-batch findings:\n{formatted}\n\n"
            "Provide the FINAL STRUCTURED REPORT:\n"
            "PRIMARY CLASSIFICATION: [crime type]\n"
            "CONFIDENCE LEVEL: [0-100%]\n"
            "SEVERITY: [Low/Medium/High/Critical]\n"
            "KEY EVIDENCE:\n- [point 1]\n- [point 2]\n- [point 3]\n"
            "ALTERNATIVE INTERPRETATIONS: [other explanations]\n"
            "RECOMMENDED LAW ENFORCEMENT RESPONSE: [actions]"
        )
        final = call_model([{"role": "user", "content": synth_prompt}])

        return {
            "video_id": video_id, "crime_type": crime_type,
            "frames_analyzed": len(frames_data), "total_batches": len(batches),
            "batch_size": BATCH_SIZE, "prompting_technique": "ZERO-SHOT",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_summaries": batch_summaries, "final_analysis": final,
        }


# ================================================================
#  2. SELF-CONSISTENCY
# ================================================================
class SelfConsistencyAnalyzer:
    """
    Sample N independent reasoning chains per batch, then aggregate by
    majority vote on the final classification. The aggregate is more robust
    to single-sample noise than any individual chain.
    """
    SYSTEM_PROMPT = FORENSIC_SYSTEM_PROMPT

    def __init__(self, batch_size=BATCH_SIZE, n_samples=5):
        self.batch_size = batch_size
        self.n_samples  = n_samples   # independent reasoning chains per batch

    def _one_sample(self, frames_data, batch, batch_num, total_batches, sample_idx):
        labels = ", ".join(CRIME_LABELS)
        framing = (
            f"Batch {batch_num}/{total_batches}, sample {sample_idx}/{self.n_samples}.\n\n"
            "Provide a step-by-step reasoning chain analyzing these video frames "
            "for criminal activity. Cover: people present, actions, interactions, "
            "objects, environment, and any indicators of crime. Then end your "
            "response with EXACTLY this final line:\n\n"
            f"FINAL_LABEL: <one of: {labels}>\n"
            "CONFIDENCE: <0-100>"
        )
        return call_model(
            [{"role": "system", "content": self.SYSTEM_PROMPT},
             {"role": "user",   "content": framing}],
            images=images_for(frames_data, batch),
        )

    @staticmethod
    def _extract_label(text):
        """Pull (label, confidence) from the structured tail of the response."""
        label_match = re.search(r"FINAL_LABEL:\s*([A-Za-z]+)", text)
        conf_match  = re.search(r"CONFIDENCE:\s*(\d+)",        text)
        label = label_match.group(1) if label_match else "Unknown"
        for canon in CRIME_LABELS:
            if label.lower() == canon.lower():
                label = canon
                break
        conf = int(conf_match.group(1)) if conf_match else 0
        return label, conf

    @staticmethod
    def _vote(samples_with_labels):
        """Majority vote on label, confidence-weighted as tiebreaker."""
        votes = Counter(lbl for lbl, _conf, _txt in samples_with_labels)
        ranked = sorted(
            votes.items(),
            key=lambda kv: (
                kv[1],
                sum(c for l, c, _ in samples_with_labels if l == kv[0]),
            ),
            reverse=True,
        )
        winner_label, winner_count = ranked[0]
        winner_confs = [c for l, c, _ in samples_with_labels if l == winner_label]
        avg_conf = sum(winner_confs) / len(winner_confs) if winner_confs else 0
        return winner_label, winner_count, avg_conf, dict(votes)

    def analyze_frames(self, frames_data, video_id, crime_type):
        batches = batch_frame_names(frames_data, self.batch_size)
        total_batches = len(batches)
        print(f"\n  [Self-Consistency] {video_id} | {len(frames_data)} frames | "
              f"{total_batches} batches | n_samples={self.n_samples}")

        per_batch_results = []
        for b_idx, batch in enumerate(batches, start=1):
            print(f"    Batch {b_idx}/{total_batches} - drawing {self.n_samples} samples ...")
            samples_with_labels = []
            for s_idx in range(1, self.n_samples + 1):
                txt = self._one_sample(frames_data, batch, b_idx, total_batches, s_idx)
                lbl, conf = self._extract_label(txt)
                samples_with_labels.append((lbl, conf, txt))
                print(f"      sample {s_idx}: {lbl} ({conf}%)")
            winner, count, avg_conf, vote_dist = self._vote(samples_with_labels)
            print(f"      -> winner: {winner} ({count}/{self.n_samples}, avg conf {avg_conf:.0f})")
            per_batch_results.append({
                "batch_num":   b_idx,
                "samples":     [{"label": l, "confidence": c, "reasoning": t}
                                for l, c, t in samples_with_labels],
                "winner_label": winner,
                "winner_count": count,
                "winner_avg_conf": avg_conf,
                "vote_distribution": vote_dist,
            })

        # Aggregate across batches: majority vote on per-batch winners
        batch_winners = Counter(r["winner_label"] for r in per_batch_results)
        final_label, final_count = batch_winners.most_common(1)[0]
        final_confs = [r["winner_avg_conf"] for r in per_batch_results
                       if r["winner_label"] == final_label]
        final_avg_conf = sum(final_confs) / len(final_confs) if final_confs else 0

        return {
            "video_id": video_id, "crime_type": crime_type,
            "frames_analyzed": len(frames_data), "total_batches": total_batches,
            "batch_size": self.batch_size, "prompting_technique": "SELF-CONSISTENCY",
            "n_samples": self.n_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_results": per_batch_results,
            "final_label": final_label,
            "final_batch_votes": final_count,
            "final_avg_confidence": final_avg_conf,
            "batch_vote_distribution": dict(batch_winners),
        }


# ================================================================
#  3. META-PROMPTING
# ================================================================
class MetaPromptingAnalyzer:
    """
    The model first GENERATES the analysis prompt it would want for this
    specific scene, then ANSWERS that self-generated prompt against the
    same frames. Two stages per batch:
      Stage 1 (planner):  design the best prompt for these frames.
      Stage 2 (executor): use the generated prompt to produce the analysis.
    """
    SYSTEM_PROMPT = (
        "You are an expert forensic video analyst specializing in crime detection "
        "and security surveillance. You also have strong skills in prompt "
        "engineering: when asked, you can design analysis prompts that elicit "
        "thorough, evidence-grounded reasoning."
    )

    META_INSTRUCTIONS = (
        "Look briefly at these video frames. Do NOT analyze them in detail yet. "
        "Instead, design the optimal analysis prompt that a forensic video "
        "analyst should follow to determine whether a crime is occurring in "
        "this specific scene. The prompt should:\n"
        "  - be tailored to what you actually see (lighting, setting, number of people, etc.),\n"
        "  - call out the most diagnostic observations to make,\n"
        f"  - require a final classification from: {', '.join(CRIME_LABELS)},\n"
        "  - require a confidence score and key evidence.\n\n"
        "Output ONLY the prompt itself, no preamble, no explanation. Start "
        "directly with the prompt text."
    )

    def __init__(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size

    def _generate_prompt(self, frames_data, batch, batch_num, total_batches):
        framing = (
            f"Batch {batch_num}/{total_batches}. Stage 1 of 2: META-PROMPT GENERATION.\n\n"
            f"{self.META_INSTRUCTIONS}"
        )
        return call_model(
            [{"role": "system", "content": self.SYSTEM_PROMPT},
             {"role": "user",   "content": framing}],
            images=images_for(frames_data, batch),
        )

    def _execute_prompt(self, frames_data, batch, batch_num, total_batches, generated_prompt):
        framing = (
            f"Batch {batch_num}/{total_batches}. Stage 2 of 2: EXECUTE THE GENERATED PROMPT.\n\n"
            "Below is the analysis prompt designed for this specific scene. "
            "Follow it precisely:\n\n"
            f"--- BEGIN GENERATED PROMPT ---\n{generated_prompt}\n--- END GENERATED PROMPT ---"
        )
        return call_model(
            [{"role": "system", "content": self.SYSTEM_PROMPT},
             {"role": "user",   "content": framing}],
            images=images_for(frames_data, batch),
        )

    def _synthesize(self, per_batch, total_frames, total_batches):
        joined = "\n\n".join(
            f"--- Batch {b['batch_num']} executed analysis ---\n{b['executed_analysis']}"
            for b in per_batch
        )
        synth_prompt = (
            f"You used meta-prompting to analyze {total_batches} batches covering "
            f"{total_frames} frames of one security video. Below is each batch's "
            f"executed analysis:\n\n{joined}\n\n"
            "Produce a FINAL combined report:\n"
            "1. SCENE DESCRIPTION.\n"
            f"2. CRIME CLASSIFICATION ({', '.join(CRIME_LABELS)}).\n"
            "3. CONFIDENCE (0-100%).\n"
            "4. KEY EVIDENCE.\n"
            "5. RECOMMENDED ACTION."
        )
        return call_model(
            [{"role": "system", "content": self.SYSTEM_PROMPT},
             {"role": "user",   "content": synth_prompt}]
        )

    def analyze_frames(self, frames_data, video_id, crime_type):
        batches = batch_frame_names(frames_data, self.batch_size)
        total_batches = len(batches)
        print(f"\n  [Meta-Prompting] {video_id} | {len(frames_data)} frames | "
              f"{total_batches} batches of up to {self.batch_size}")

        per_batch = []
        for b_idx, batch in enumerate(batches, start=1):
            print(f"    Batch {b_idx}/{total_batches} - generating prompt ...")
            gen_prompt = self._generate_prompt(frames_data, batch, b_idx, total_batches)
            print(f"      generated prompt: {len(gen_prompt)} chars")
            print(f"    Batch {b_idx}/{total_batches} - executing prompt ...")
            executed = self._execute_prompt(frames_data, batch, b_idx, total_batches, gen_prompt)
            print(f"      executed analysis: {len(executed)} chars")
            per_batch.append({
                "batch_num":         b_idx,
                "generated_prompt":  gen_prompt,
                "executed_analysis": executed,
            })

        print(f"    Synthesizing {total_batches} batches ...")
        final_report = self._synthesize(per_batch, len(frames_data), total_batches)
        print(f"      final report: {len(final_report)} chars")

        return {
            "video_id": video_id, "crime_type": crime_type,
            "frames_analyzed": len(frames_data), "total_batches": total_batches,
            "batch_size": self.batch_size, "prompting_technique": "META-PROMPTING",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_traces": per_batch, "final_analysis": final_report,
        }


# ================================================================
#  4. CHAIN-OF-THOUGHT
# ================================================================
class ChainOfThoughtAnalyzer:
    """
    Explicit step-by-step reasoning template applied to every batch. The
    model walks through fixed reasoning stages before classifying, with a
    "Let's think step by step" framing.
    """
    SYSTEM_PROMPT = (
        "You are an expert forensic video analyst specializing in crime detection "
        "and security surveillance. You analyze video frames methodically, walking "
        "through your reasoning step by step. You always ground each conclusion in "
        "specific visual evidence from the frames."
    )

    COT_TEMPLATE = (
        "Analyze these video frames for criminal activity. Let's think step by step.\n\n"
        "Walk through these reasoning stages explicitly. Label each stage in your "
        "response:\n\n"
        "STEP 1 - OBSERVATIONS: List concrete visual facts. People (count, clothing, "
        "approximate age/build), objects (vehicles, weapons, tools, packages), "
        "environment (indoor/outdoor, lighting, location type). Just facts, no inference.\n\n"
        "STEP 2 - ACTIONS: For each person, what specific actions are they performing? "
        "How do those actions change across frames?\n\n"
        "STEP 3 - INTERACTIONS: How are people interacting with each other and with "
        "objects? Who initiates contact? Are interactions cooperative, hostile, or neutral?\n\n"
        "STEP 4 - INDICATORS: What specific elements suggest criminal vs. non-criminal "
        "activity? List supporting evidence for each side.\n\n"
        "STEP 5 - HYPOTHESIS: Given STEPS 1-4, what is the most likely explanation? "
        "What alternative explanations are also plausible, and what would distinguish them?\n\n"
        "STEP 6 - CLASSIFICATION: Choose ONE of: " + ", ".join(CRIME_LABELS) + ". "
        "Give a confidence (0-100%) and the 2-3 strongest pieces of evidence."
    )

    SYNTHESIS_TEMPLATE = (
        "You used chain-of-thought reasoning across {total_batches} batches "
        "covering {total_frames} frames of one security video. Below is each "
        "batch's complete reasoning chain:\n\n{joined}\n\n"
        "Now reason step by step ONE MORE TIME, this time across batches:\n\n"
        "STEP A - CROSS-BATCH OBSERVATIONS: What is consistent across batches? "
        "What changes?\n\n"
        "STEP B - TEMPORAL TIMELINE: Build a chronological timeline of events.\n\n"
        "STEP C - FINAL CLASSIFICATION: Choose ONE of " + ", ".join(CRIME_LABELS) + ".\n"
        "STEP D - CONFIDENCE: 0-100%.\n"
        "STEP E - KEY EVIDENCE: 3-5 specific items.\n"
        "STEP F - RECOMMENDED ACTION."
    )

    def __init__(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size

    def _analyze_batch(self, frames_data, batch, batch_num, total_batches):
        framing = f"Batch {batch_num}/{total_batches}.\n\n{self.COT_TEMPLATE}"
        return call_model(
            [{"role": "system", "content": self.SYSTEM_PROMPT},
             {"role": "user",   "content": framing}],
            images=images_for(frames_data, batch),
        )

    def _synthesize(self, per_batch, total_frames, total_batches):
        joined = "\n\n".join(
            f"--- Batch {b['batch_num']} reasoning chain ---\n{b['reasoning_chain']}"
            for b in per_batch
        )
        synth_prompt = self.SYNTHESIS_TEMPLATE.format(
            total_batches=total_batches, total_frames=total_frames, joined=joined,
        )
        return call_model(
            [{"role": "system", "content": self.SYSTEM_PROMPT},
             {"role": "user",   "content": synth_prompt}]
        )

    def analyze_frames(self, frames_data, video_id, crime_type):
        batches = batch_frame_names(frames_data, self.batch_size)
        total_batches = len(batches)
        print(f"\n  [Chain-of-Thought] {video_id} | {len(frames_data)} frames | "
              f"{total_batches} batches of up to {self.batch_size}")

        per_batch = []
        for b_idx, batch in enumerate(batches, start=1):
            print(f"    Batch {b_idx}/{total_batches} ...")
            chain = self._analyze_batch(frames_data, batch, b_idx, total_batches)
            print(f"      reasoning chain: {len(chain)} chars")
            per_batch.append({"batch_num": b_idx, "reasoning_chain": chain})

        print(f"    Synthesizing {total_batches} batches ...")
        final_report = self._synthesize(per_batch, len(frames_data), total_batches)
        print(f"      final report: {len(final_report)} chars")

        return {
            "video_id": video_id, "crime_type": crime_type,
            "frames_analyzed": len(frames_data), "total_batches": total_batches,
            "batch_size": self.batch_size, "prompting_technique": "CHAIN-OF-THOUGHT",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_chains": per_batch, "final_analysis": final_report,
        }


# ================================================================
#  5. SEQUENTIAL (multi-turn)
# ================================================================
class SequentialAnalyzer:
    """
    Sequential multi-turn analysis. Stage 1 processes frames in batches
    (images); stages 2-4 are a text-only multi-turn conversation that
    progressively narrows from scene -> suspicious elements -> classification
    -> final summary.
    """
    def analyze_frames(self, frames_data, video_id, crime_type):
        print(f"\n  [Sequential] {video_id} | {len(frames_data)} frames ...")
        batches = batch_frame_names(frames_data)
        stages = {}

        # Stage 1: Batched scene description (with images)
        print(f"    Stage 1: Scene description | {len(batches)} batches ...")
        batch_summaries = []
        for idx, batch in enumerate(batches, 1):
            print(f"      Batch {idx}/{len(batches)} ({len(batch)} frames) ...")
            prompt = (
                f"Describe this batch of security camera frames (Batch {idx}/{len(batches)}): "
                "setting, people present, their actions, and notable elements."
            )
            s = call_model([{"role": "user", "content": prompt}],
                           images=images_for(frames_data, batch))
            batch_summaries.append(s)
            print(f"        Response: {len(s)} chars")

        formatted = "\n\n".join(f"--- Batch {i+1} ---\n{s}"
                                for i, s in enumerate(batch_summaries))
        synth_prompt = (
            f"Per-batch observations from {len(frames_data)} security camera frames:\n{formatted}\n\n"
            "Write a single unified SCENE DESCRIPTION covering all batches."
        )
        scene_desc = call_model([{"role": "user", "content": synth_prompt}])
        stages["stage1_batch_summaries"] = batch_summaries
        stages["stage1_scene_description"] = scene_desc
        print(f"      Scene description: {len(scene_desc)} chars")

        # Stages 2-4: Text-only sequential conversation
        conversation = [
            {"role": "user", "content":
                f"I analyzed {len(frames_data)} security camera frames. Scene description:\n\n{scene_desc}"},
            {"role": "assistant", "content":
                "Understood. I have reviewed the scene description and am ready to proceed."},
        ]

        for stage_num, (label, prompt) in enumerate([
            ("stage2_suspicious_elements",
             "Identify suspicious or unusual behaviours suggesting criminal activity. Look for patterns."),
            ("stage3_crime_classification",
             "Classify the crime: " + ", ".join(CRIME_LABELS) + ". Explain your reasoning."),
            ("stage4_final_summary",
             "Provide final structured summary: (1) Primary classification, (2) Confidence 0-100%, "
             "(3) Key evidence, (4) Alternative interpretations, (5) Recommended law enforcement actions."),
        ], start=2):
            print(f"    Stage {stage_num}: {label} ...")
            conversation.append({"role": "user", "content": prompt})
            response = call_model(conversation)
            conversation.append({"role": "assistant", "content": response})
            stages[label] = response
            print(f"      Response: {len(response)} chars")

        return {
            "video_id": video_id, "crime_type": crime_type,
            "frames_analyzed": len(frames_data), "total_batches": len(batches),
            "batch_size": BATCH_SIZE, "prompting_technique": "SEQUENTIAL",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": stages,
        }


# ================================================================
#  6. ReAct (Reasoning + Acting)
# ================================================================
class ReActAnalyzer:
    """
    ReAct interleaves Thought (reasoning) and Action (examination) steps.
    Thought 1 is text-only; Action 1 examines frames in batches (images);
    the remaining Thought/Action/Final steps are text-only multi-turn.
    """
    def analyze_frames(self, frames_data, video_id, crime_type):
        print(f"\n  [ReAct] {video_id} | {len(frames_data)} frames ...")
        batches = batch_frame_names(frames_data)
        react_log = {}
        conversation = []

        # THOUGHT 1: Pre-analysis reasoning (text-only)
        print("    Thought 1: Initial reasoning ...")
        conversation.append({"role": "user", "content": (
            f"I need to analyze {len(frames_data)} security camera frames using the ReAct framework.\n\n"
            "THOUGHT 1: Before examining frames, reason about what indicators to look for "
            "to detect criminal activity in security footage. What are key behaviours and visual cues?"
        )})
        thought1 = call_model(conversation)
        conversation.append({"role": "assistant", "content": thought1})
        react_log["thought1"] = thought1
        print(f"      Response: {len(thought1)} chars")

        # ACTION 1: Batched frame examination (with images)
        print(f"    Action 1 (images): {len(batches)} batches ...")
        batch_obs = []
        for idx, batch in enumerate(batches, 1):
            prompt = (
                f"THOUGHT 1:\n{thought1}\n\n"
                f"ACTION 1 - Examine Frames (Batch {idx}/{len(batches)}): "
                "Based on the reasoning above, what do you observe? "
                "Report people, actions, objects, events."
            )
            obs = call_model([{"role": "user", "content": prompt}],
                             images=images_for(frames_data, batch))
            batch_obs.append(obs)
            print(f"      Batch {idx}: {len(obs)} chars")

        formatted = "\n\n".join(f"--- Batch {i+1} ---\n{s}"
                                for i, s in enumerate(batch_obs))
        observation = call_model([{"role": "user", "content": (
            f"Consolidate these observations from {len(frames_data)} frames:\n{formatted}\n\n"
            "Write a single unified OBSERVATION."
        )}])
        react_log["action1_batch_obs"] = batch_obs
        react_log["observation1"] = observation
        print(f"      Consolidated observation: {len(observation)} chars")

        conversation.append({"role": "user", "content":
            f"ACTION 1 COMPLETE - OBSERVATION:\n{observation}\n\n"
            f"All {len(frames_data)} frames across {len(batches)} batches examined."})
        conversation.append({"role": "assistant", "content":
            "Observation noted. I will now reason about what these findings mean."})

        # THOUGHT 2, ACTION 2, THOUGHT 3, FINAL ANSWER — text-only
        for label, prompt in [
            ("thought2",
             "THOUGHT 2: What do the observations collectively indicate? "
             "What patterns emerge? Any contradictions or ambiguities?"),
            ("action2",
             "ACTION 2: Focused re-analysis — zero in on the most diagnostic evidence. "
             "What is most critical for determining crime type?"),
            ("thought3",
             "THOUGHT 3: Reason through crime classification. Consider each: "
             + ", ".join(CRIME_LABELS) + ". Which best fits?"),
            ("final_answer",
             "FINAL ANSWER:\n"
             "PRIMARY CLASSIFICATION: [crime type]\n"
             "CONFIDENCE LEVEL: [0-100%]\n"
             "SEVERITY: [Low/Medium/High/Critical]\n"
             "KEY EVIDENCE:\n- [point 1]\n- [point 2]\n"
             "REASONING SUMMARY: [how ReAct cycles led to conclusion]\n"
             "ALTERNATIVE INTERPRETATIONS: [other explanations]\n"
             "RECOMMENDED LAW ENFORCEMENT RESPONSE: [actions]"),
        ]:
            print(f"    {label} ...")
            conversation.append({"role": "user", "content": prompt})
            response = call_model(conversation)
            conversation.append({"role": "assistant", "content": response})
            react_log[label] = response
            print(f"      Response: {len(response)} chars")

        return {
            "video_id": video_id, "crime_type": crime_type,
            "frames_analyzed": len(frames_data), "total_batches": len(batches),
            "batch_size": BATCH_SIZE, "prompting_technique": "REACT",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "react_log": react_log,
        }


# ================================================================
#  7. LEAST-TO-MOST
# ================================================================
class LeastToMostAnalyzer:
    """
    Decompose the problem from simplest to most complex. Level 1 processes
    frames in batches (images); levels 2-5 are text-only, increasing in
    complexity from basic identification to full classification and report.
    """
    def analyze_frames(self, frames_data, video_id, crime_type):
        print(f"\n  [Least-to-Most] {video_id} | {len(frames_data)} frames ...")
        batches = batch_frame_names(frames_data)
        stages = {}

        # Level 1: Basic visual elements (images, batched) — LOW complexity
        print(f"    Level 1 (LOW): Basic visual elements | {len(batches)} batches ...")
        batch_summaries = []
        for idx, batch in enumerate(batches, 1):
            prompt = (
                f"LEVEL 1 - Basic Visual Elements only (Batch {idx}/{len(batches)}): "
                "List setting type, number of people, objects present, time of day if determinable, "
                "camera quality. Factual only."
            )
            s = call_model([{"role": "user", "content": prompt}],
                           images=images_for(frames_data, batch))
            batch_summaries.append(s)

        formatted = "\n\n".join(f"--- Batch {i+1} ---\n{s}"
                                for i, s in enumerate(batch_summaries))
        visual_summary = call_model([{"role": "user", "content": (
            f"Basic visual observations from {len(frames_data)} frames:\n{formatted}\n\n"
            "Write a unified BASIC VISUAL ELEMENTS summary. Factual only, no interpretation."
        )}])
        stages["level1_batch_summaries"] = batch_summaries
        stages["level1_basic_elements"] = visual_summary
        print(f"      Visual summary: {len(visual_summary)} chars")

        # Levels 2-5: Text-only sequential conversation — increasing complexity
        conversation = [
            {"role": "user", "content":
                f"I analyzed {len(frames_data)} security camera frames. Basic visual elements:\n\n{visual_summary}"},
            {"role": "assistant", "content":
                "Understood. I have reviewed the basic visual elements and am ready for deeper analysis."},
        ]

        for level_num, (label, complexity, prompt) in enumerate([
            ("level2_human_behaviour", "MEDIUM",
             "LEVEL 2 - Human Behaviour: Describe body language, movement patterns, facial expressions "
             "if visible, hand/arm movements, speed and urgency of actions."),
            ("level3_interactions", "MEDIUM",
             "LEVEL 3 - Interactions & Context: How are people interacting with each other and objects? "
             "Social dynamics, power imbalances, conflicts, and likely purpose of actions?"),
            ("level4_classification", "HIGH",
             "LEVEL 4 - Crime Classification: Is a crime occurring? Choose from: "
             + ", ".join(CRIME_LABELS) + ". What ruled out other types?"),
            ("level5_final_report", "HIGH",
             "LEVEL 5 - Final Integrated Report:\n"
             "PRIMARY CLASSIFICATION: [crime type]\n"
             "CONFIDENCE LEVEL: [0-100%]\n"
             "SEVERITY: [Low/Medium/High/Critical]\n"
             "EVIDENCE SUMMARY:\n- [point 1]\n- [point 2]\n"
             "TIMELINE OF EVENTS: [chronological summary]\n"
             "ALTERNATIVE INTERPRETATIONS: [other explanations]\n"
             "RECOMMENDED LAW ENFORCEMENT RESPONSE: [actions]"),
        ], start=2):
            print(f"    Level {level_num} ({complexity}): {label} ...")
            conversation.append({"role": "user", "content": prompt})
            response = call_model(conversation)
            conversation.append({"role": "assistant", "content": response})
            stages[label] = response
            print(f"      Response: {len(response)} chars")

        return {
            "video_id": video_id, "crime_type": crime_type,
            "frames_analyzed": len(frames_data), "total_batches": len(batches),
            "batch_size": BATCH_SIZE, "prompting_technique": "LEAST-TO-MOST",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": stages,
        }


# ================================================================
#  8. ITERATIVE REFINEMENT
# ================================================================
class IterativeAnalyzer:
    """
    Iterative prompting: the SAME core question is asked repeatedly, each
    round refining the previous answer. Stops when consecutive answers
    converge (high word-overlap similarity) or max_iterations is reached.
    """
    CORE_QUESTION = (
        "Analyze these video frames for criminal activity. What crime is occurring, "
        "who is involved, what evidence supports your conclusion, and how confident "
        "are you in this assessment?"
    )

    def __init__(self, batch_size=BATCH_SIZE, max_iterations=8, convergence_threshold=0.7):
        self.chunk_size            = batch_size
        self.max_iterations        = max_iterations
        self.convergence_threshold = convergence_threshold

    @staticmethod
    def calculate_similarity(text1, text2):
        """Jaccard similarity on word sets."""
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        union = words1 | words2
        if not union:
            return 0.0
        return len(words1 & words2) / len(union)

    def has_converged(self, current, previous):
        if not previous:
            return False
        sim = self.calculate_similarity(current, previous)
        print(f"    Similarity to previous: {sim:.3f} (threshold: {self.convergence_threshold})")
        return sim >= self.convergence_threshold

    @staticmethod
    def extract_confidence_score(response):
        """Extract a 0-1 confidence from the response text."""
        percentages = re.findall(r"(\d+)%", response)
        if percentages:
            return max(int(p) for p in percentages) / 100.0
        lower = response.lower()
        if any(k in lower for k in ["very confident", "highly confident", "extremely confident"]):
            return 0.9
        if any(k in lower for k in ["confident", "fairly confident"]):
            return 0.7
        if any(k in lower for k in ["somewhat confident", "moderately confident"]):
            return 0.5
        if any(k in lower for k in ["low confidence", "uncertain", "unsure"]):
            return 0.3
        return 0.5

    def analyze_frames(self, frames_data, video_id, crime_type):
        print(f"\n  [Iterative] {video_id} | {len(frames_data)} frames ...")
        frame_names = sorted(frames_data.keys(), key=extract_frame_number)
        all_b64 = [frames_data[f] for f in frame_names]

        all_iterations = {}
        previous_response = None
        converged = False
        confidence = 0.5
        iteration_num = 0

        print(f"  Core question: {self.CORE_QUESTION}")
        print(f"  Convergence threshold: {self.convergence_threshold}")

        for iteration_num in range(1, self.max_iterations + 1):
            print(f"\n=== ITERATION {iteration_num}/{self.max_iterations} ===")
            iteration_responses = []

            for i in range(0, len(all_b64), self.chunk_size):
                chunk = all_b64[i:i + self.chunk_size]

                if iteration_num == 1:
                    iterative_prompt = (
                        f"ITERATION {iteration_num} - Initial Analysis\n\n"
                        f"{self.CORE_QUESTION}\n\n"
                        "Be thorough and specific in your analysis. Include your "
                        "confidence level in your assessment."
                    )
                else:
                    iterative_prompt = (
                        f"ITERATION {iteration_num} - Refining Previous Analysis\n\n"
                        f"PREVIOUS ANALYSIS FROM ITERATION {iteration_num - 1}:\n"
                        f"{previous_response[:800]}...\n\n"
                        "Now, analyze these SAME frames again with the SAME core "
                        "question, but refine your analysis:\n\n"
                        f"{self.CORE_QUESTION}\n\n"
                        "REFINEMENT INSTRUCTIONS:\n"
                        "- Review your previous analysis carefully\n"
                        "- Look for details you may have missed\n"
                        "- Reconsider your conclusions with fresh perspective\n"
                        "- Identify any errors or oversights in your previous assessment\n"
                        "- Improve the accuracy and depth of your analysis\n"
                        "- If you're more confident now, explain why\n"
                        "- If you're less confident, explain what creates uncertainty\n"
                        "- What new insights do you have upon re-examination?\n\n"
                        "Provide your REFINED analysis of the same core question."
                    )

                chunk_no = i // self.chunk_size + 1
                total_chunks = (len(all_b64) + self.chunk_size - 1) // self.chunk_size
                print(f"  Processing chunk {chunk_no}/{total_chunks} ...")
                response = call_model(
                    [{"role": "user", "content": iterative_prompt}],
                    images=chunk,
                )
                iteration_responses.append(response)

            current_response = (iteration_responses[0]
                                if len(iteration_responses) == 1
                                else "\n\n=== NEXT CHUNK ===\n\n".join(iteration_responses))

            confidence = self.extract_confidence_score(current_response)
            if previous_response:
                converged = self.has_converged(current_response, previous_response)

            all_iterations[f"iteration_{iteration_num}"] = {
                "iteration": iteration_num,
                "type": "iterative_refinement",
                "core_question": self.CORE_QUESTION,
                "prompt_used": iterative_prompt,
                "response": current_response,
                "confidence_extracted": confidence,
                "converged": converged,
                "similarity_to_previous":
                    self.calculate_similarity(current_response, previous_response)
                    if previous_response else 0.0,
            }

            print(f"  Confidence level: {confidence:.2f}")
            if converged:
                print(f"  *** CONVERGENCE ACHIEVED at iteration {iteration_num} ***")
                break
            elif iteration_num < self.max_iterations:
                print(f"  Continuing to iteration {iteration_num + 1} (not yet converged)")

            previous_response = current_response

        convergence_summary = {
            "total_iterations_run": iteration_num,
            "max_iterations_allowed": self.max_iterations,
            "converged": converged,
            "convergence_threshold": self.convergence_threshold,
            "final_confidence": confidence,
            "methodology": "Iterative refinement - same question refined repeatedly",
        }
        if converged:
            convergence_summary["convergence_iteration"] = iteration_num

        return {
            "video_id": video_id, "crime_type": crime_type,
            "frames_analyzed": len(frames_data),
            "batch_size": self.chunk_size, "prompting_technique": "ITERATIVE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iterations": all_iterations,
            "convergence_summary": convergence_summary,
        }


# ================================================================
#  TECHNIQUE REGISTRY + DRIVER
# ================================================================
TECHNIQUES = {
    "zero_shot":        ZeroShotAnalyzer,
    "self_consistency": SelfConsistencyAnalyzer,
    "meta":             MetaPromptingAnalyzer,
    "chain_of_thought": ChainOfThoughtAnalyzer,
    "sequential":       SequentialAnalyzer,
    "react":            ReActAnalyzer,
    "least_to_most":    LeastToMostAnalyzer,
    "iterative":        IterativeAnalyzer,
}


def process_all_videos(technique="zero_shot"):
    """
    Run one technique over every video discovered under FRAMES_DIR and
    write a combined summary JSON to SAVE_DIR.
    """
    if technique not in TECHNIQUES:
        raise ValueError(f"Unknown technique '{technique}'. "
                         f"Choose from: {list(TECHNIQUES)}")
    analyzer = TECHNIQUES[technique]()
    all_videos = discover_all_videos_and_frames()
    if not all_videos:
        print("No videos found! Verify FRAMES_DIR path.")
        return {}

    results = {}
    for vkey, vinfo in all_videos.items():
        print(f"\n  [START] {vkey}")
        frames = load_frames_for_video(vinfo, FRAME_INTERVAL)
        if not frames:
            print(f"  [SKIP] {vkey}: no frames")
            continue
        try:
            results[vkey] = analyzer.analyze_frames(
                frames, vinfo["video_id"], vinfo["crime_type"])
            print(f"  [DONE] {vkey}")
        except Exception as e:
            print(f"  [ERROR] {vkey}: {e}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    save_results(results, f"{technique}_summary_{ts}.json")
    print(f"\nDone. Processed {len(results)} videos with technique '{technique}'.")
    return results


if __name__ == "__main__":
    # Pick a technique key from TECHNIQUES above.
    # Remember to implement call_model() first.
    process_all_videos(technique="zero_shot")
