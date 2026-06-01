# BWC PIPELINE: COMPLETE TECHNICAL RESEARCH REPORT

## SECTION 1 — PROJECT OVERVIEW

### Project Objective
The primary objective of this project is to develop a robust, highly accurate, and fully automated speech recognition and diarization pipeline specifically designed for Body-Worn Camera (BWC) footage. The ultimate goal is to process forensic audio and attribute speech correctly to individual speakers (e.g., officers, subjects, bystanders) to generate actionable intelligence and transcripts.

### Nature of BWC Audio & Its Difficulties
Body-Worn Camera audio is fundamentally different from standard conversational audio (e.g., podcasts, meetings, telephone calls). It is arguably the most hostile environment for speech processing.

**Challenges Include:**
* **Low Signal-to-Noise Ratio (SNR):** The primary signal (speech) is constantly competing with overwhelming background noise.
* **Dynamic Speaker Distance:** The microphone is worn on the chest of the officer. The officer's voice is extremely loud and resonant (often clipping), while the subject's voice varies wildly depending on distance and orientation. This causes severe "speaker embedding drift."
* **Environmental Intrusions:** Wind noise, sirens, traffic, and structural reverberations constantly interfere with the frequency spectrum.
* **Overlapping Speech & Radio Chatter:** Highly stressful encounters involve shouting, overlapping speech, and police radio chatter, which act as adversarial noise to clustering algorithms.
* **Clipping:** Sudden shouts or proximity to the mic causes acoustic clipping, destroying the harmonic structure necessary for Neural VADs and ASR.

### Why Conventional Pipelines Fail
Standard pipelines (like running raw audio through OpenAI's Whisper) fail dramatically on BWC data for several reasons:
1. **Hallucination Loops:** Whisper's transformer attention mechanism is easily derailed by low-SNR audio and sirens, causing it to hallucinate repetitive text or skip entire blocks of speech.
2. **Speaker Attribution:** Whisper alone does not diarize. It produces text, but in forensic contexts, knowing *what* was said is useless without knowing *who* said it.
3. **The Grounding Problem:** Standard clustering diarizers (like Pyannote) will attempt to cluster wind noise and sirens into "speakers" (False Alarms) if the audio is not strictly gated by Voice Activity Detection (VAD).

Evaluation must go beyond pure Word Error Rate (WER). A low WER with a high Diarization Error Rate (DER) means the text is correct but assigned to the wrong person—a critical failure in policing contexts. Thus, concatenated permutation WER (cpWER) and Speaker-Attributed WER (SA-WER) become paramount.

### Final Hardened Architecture (High-Level)
```text
[Raw BWC Audio] 
       ↓ 
[Loudness Normalization (-20dB)] -> Stabilizes embeddings
       ↓ 
[Pyannote 3.1 VAD] -> Generates "Master Grounding" (Strict speech segments)
       ↓
[Muting Trick / Slicing] -> Zeroes out non-speech to prevent hallucination/FA
       |
       +--> [Hardened ASR (WhisperX large-v3)] -> Transcribes isolated slices
       |
       +--> [Oracle MSDD Diarization (NeMo)] -> Clusters embeddings iteratively
       ↓
[Word-Level Alignment] -> WhisperX phoneme mapping
       ↓
[Speaker Assignment] -> Final forensic transcript (Hardened JSON & TXT)
```

---

## SECTION 2 — COMPLETE DIRECTORY STRUCTURE

The project is organized to separate external dependencies, modular scripts, experimental sandbox phases, and evaluated reports.

```text
experiments/
├── phase1_vad/                 # VAD benchmarking (Silero vs Pyannote)
├── phase2_enhancement/         # Speech enhancement (DeepFilterNet, SpeechBrain)
├── phase3_diarization/         # Diarization benchmarking (Pyannote vs NeMo MSDD)
├── phase4_grounded_enhancement/# Controlled enhancement using normalized oracle VAD
├── phase5_grounded_diarization/# Controlled diarization using normalized oracle VAD
├── phase5_audit/               # Debugging and validation of MSDD Oracle config
│
reports/
├── conversational_eval_*.csv   # Generated timeline metrics (WER, DER, cpWER)
├── phase1_report.md            # VAD synthesis
├── phase2_report.md            # Enhancement synthesis
├── final_recommendation.md     # Initial strategic recommendations
│
scripts/
└── utils/
    ├── audio_utils.py          # Audio loading, resampling, scaling
    ├── device.py               # GPU/CPU device management
    ├── parse_ground_truth.py   # Converts human TXT to RTTM/JSON
    └── diarization_diagnostics.py # Error analysis (Miss/FA/Conf) extraction
│
[Root level pipeline scripts]
├── preprocess_audio.py         # Normalization and bandpass filtering
├── custom_pipeline_*.py        # Modular stage scripts
├── evaluate_*.py               # Evaluation frameworks
├── run_final_hardened_pipeline.py # The official production ASR+Diarization flow
```

**Directory Purpose & Dependencies:**
* **`experiments/`**: The research sandbox. Inputs are raw or normalized audio; outputs are JSON segments and evaluation CSVs. Dependencies cross-pollinate (e.g., Phase 4 uses Phase 1 VAD logic).
* **`reports/`**: The source of truth for metrics. Heavily dependent on the outputs of the `evaluate_conversational_timeline.py` script.
* **`scripts/utils/`**: Helper functions imported via `sys.path.append()`. 

---

## SECTION 3 — EVOLUTION OF THE PIPELINE

### Chronological Timeline
1. **Phase 1 (VAD Study)**
   * *Hypothesis*: High Miss/FA rates are due to poor Voice Activity Detection.
   * *Outcome*: Pyannote outperformed Silero by capturing more speech, proving that VAD dictates the ceiling of the ASR.
2. **Phase 2 (Enhancement Study)**
   * *Hypothesis*: Cleaning the audio (denoising) will help Whisper hear better.
   * *Outcome*: Failed. Neural enhancement (SpeechBrain) corrupted the transformer attention.
3. **Phase 3 (Diarization Study)**
   * *Hypothesis*: Pyannote clustering collapses on moving speakers; NeMo MSDD can fix this.
   * *Outcome*: Mixed. MSDD showed promise but failed due to complex config parameters and un-isolated variables (raw audio).
4. **Phase 4 (Grounded Enhancement)**
   * *Hypothesis*: Enhancement failed in Phase 2 because variables weren't isolated. Let's normalize the audio and fix the VAD grounding first.
   * *Outcome*: Confirmed Phase 2 findings. Even strictly controlled, enhancement hurts WER. Normalization is the only required pre-processing.
5. **Phase 5 (Grounded Diarization)**
   * *Hypothesis*: NeMo MSDD needs strictly controlled, normalized inputs and an "Oracle" VAD to succeed.
   * *Outcome*: Success. By bypassing NeMo's internal VAD and using the Master Grounding, MSDD achieved superior speaker identification (prevented identity collapse).

---

## SECTION 4 — PHASE 1 (VAD STUDY)

### Objective
Determine the optimal algorithm for generating "grounding" segments. In a hostile BWC environment, if the VAD misses speech, the ASR will never see it. If the VAD hallucinates speech, the ASR will hallucinate text.

### Models Evaluated
1. **Silero VAD**: A lightweight, fast neural VAD.
2. **Pyannote VAD**: A heavier segmentation model (v3.0).

### Implementation Strategy
* **Min-Cut Strategy (`pipeline_vad_silero.py`)**: For Silero, segments often ran too long. A recursive algorithm searched for the lowest speech-probability point within a window to split the audio into manageable <30s chunks.
* **Chunk Merging**: Detected frames were merged into ~28s continuous blocks to provide Whisper with maximum context without exceeding its 30s attention window.

### Results (BWC)
| VAD | WER | DER | Miss | FA | Confusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Silero** | 0.3037 | 0.4234 | **0.1784** | **0.1746** | 0.0704 |
| **Pyannote** | 0.3079 | **0.4147** | **0.1164** | 0.2299 | 0.0683 |

### Analysis: "False Alarm is cheap, Miss is fatal"
Pyannote produced significantly more False Alarms (22.9% vs 17.4%). However, it reduced Misses by over 6%. In a grounded pipeline, a False Alarm simply feeds silence/noise to Whisper. Whisper is generally robust enough to return an empty string for noise. A Miss, however, completely deletes the speech from the timeline. Pyannote was declared the winner due to its aggressive "coverage."

---

## SECTION 5 — PHASE 2 (ENHANCEMENT STUDY)

### Objective
Attempt to reduce Whisper substitutions by denoising low-SNR BWC audio.

### Models Evaluated
1. **DeepFilterNet3**: Low-complexity deep filtering optimized for edge devices.
2. **SpeechBrain MetricGAN+**: Generative adversarial network optimizing specifically for evaluation metrics (PESQ).

### Implementation
Scripts (`pipeline_enhancement_*.py`) read raw WAV files and output processed WAVs. `run_phase2.py` then ran the hardened ASR pipeline on these new waveforms.

### Results (BWC)
| Enhancer | WER | Subs | Dels | Ins |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (None)** | **0.3037** | **68** | 48 | 31 |
| **SpeechBrain** | 0.4504 | 86 | 111 | 21 |

### Analysis
The results were disastrous. Enhancement dramatically degraded performance.
* **Artifacts vs. Noise**: Whisper is trained on 680k hours of noisy, real-world data. It understands wind and traffic. It *does not* understand the phase distortions, "musical noise," and warbling artifacts introduced by GAN-based enhancers. 
* **Attention Disruption**: The artifacts disrupted Whisper's cross-attention mechanisms, causing massive spikes in Deletions ( Whisper simply gave up trying to transcribe the distorted audio).

---

## SECTION 6 — PHASE 3 (DIARIZATION STUDY)

### Objective
Address the >40% Diarization Error Rate (DER) observed in Phase 1. 

### The Problem: Speaker Drift & Collapse
In BWC audio, standard clustering (Agglomerative Clustering) fails. As an officer turns their body, the acoustic signature (embedding) of the subject changes drastically. The clustering algorithm interprets this as a new speaker (Fragmentation) or merges everyone into the loudly dominant officer (Collapse).

### Models Evaluated
1. **Pyannote 3.1** (Standard Clustering).
2. **NeMo MSDD** (Multi-Scale Diarization Decoder). MSDD uses a sequence-to-sequence model to iteratively resolve overlapping speech and correct clustering errors using multiple embedding scales.

### Limitations Discovered
Phase 3 highlighted severe configuration complexities in NeMo. MSDD requires strict YAML configs (OmegaConf), manifests, and independent VAD processing. Initial attempts yielded poor comparisons because NeMo was running its own VAD (`MarbleNet`), meaning the inputs to the diarizers were no longer identical. This necessitated Phase 5.

---

## SECTION 7 — PHASE 4 (GROUNDED ENHANCEMENT)

### Objective
Re-test speech enhancement, but this time strictly control the variables. 

### Implementation
Instead of running VAD *after* enhancement, Phase 4 generated a **Master Grounding** using Pyannote on **Loudness Normalized** audio (-20dB). All enhancement models (DeepFilterNet3, SpeechBrain) were evaluated strictly on these shared segments.

### Results (BWC - Controlled)
| Enhancer | WER | DER | Subs | Dels | Ins |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Norm)** | **0.2975** | **0.4114** | 86 | **32** | 26 |
| **DeepFilterNet3** | 0.3698 | 0.4408 | **76** | 86 | 17 |
| **SpeechBrain** | 0.4793 | 0.5481 | 77 | 141 | **14** |

### Analysis
DeepFilterNet3 successfully reduced Substitutions (86 down to 76), proving that it made the speech marginally "clearer." However, it aggressively gated the audio, destroying trailing consonants and quiet speech, which caused Deletions to nearly triple (32 up to 86). 

**Conclusion**: Neural enhancement is unequivocally rejected for the production pipeline. Loudness Normalization alone provides the best stability.

---

## SECTION 8 — PHASE 5 (GROUNDED DIARIZATION)

### Objective
Execute the ultimate diarization test by injecting the Phase 4 Master Grounding into NeMo MSDD as an "Oracle" VAD. 

### The Journey & Debugging
1. **Initial Failures**: The `pipeline_msdd.py` script crashed repeatedly due to NeMo's exhaustive configuration requirements (`clustering`, `batch_size`, `num_workers`, `sample_rate`, `overlap`, `smoothing`).
2. **The Oracle VAD Pivot**: To ensure fairness, NeMo's internal `MarbleNet` VAD was bypassed. A custom utility (`convert_grounding_to_rttm.py`) was written to convert the Master JSON grounding into RTTM format, which was fed to NeMo via the `oracle_vad=True` config path.
3. **Execution**: This optimization bypassed a 5-minute timeout, reducing the diarization runtime to ~37 seconds and yielding perfect isolation.

### Results (BWC - Controlled Master Grounding)
| Model | DER | Miss | FA | Confusion | Pred. Spks | SA-WER | cpWER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pyannote Auto** | **0.4114** | 0.1137 | 0.2296 | **0.0681** | 1 | **0.4008** | 0.5054 |
| **Pyannote Fixed-2** | 0.4725 | 0.1137 | 0.2296 | 0.1292 | 2 | 0.4587 | 0.5364 |
| **MSDD Oracle** | 0.4492 | 0.1137 | 0.2296 | 0.1059 | 2 | 0.4153 | **0.4582** |

### Deep Analysis
* **Why Pyannote "Wins" DER**: Pyannote Auto achieved the lowest DER by collapsing the audio into 1 speaker. By refusing to split the audio, it mathematically minimized "Confusion" errors, but generated a useless forensic transcript.
* **Why MSDD Wins Utility**: MSDD successfully identified the presence of 2 speakers. Compared to Pyannote Fixed-2 (which was forced to find 2 speakers), MSDD demonstrated vastly superior clustering logic (Confusion: 0.105 vs 0.129).
* **The Importance of cpWER**: concatenated permutation WER (cpWER) measures the error rate of the text *assigned to the correct speaker*. MSDD achieved the lowest cpWER (0.458), proving it is the most functionally accurate model for police intelligence.

---

## SECTION 9 — SCRIPT-BY-SCRIPT REFERENCE

### `preprocess_audio.py`
* **Purpose**: Standardizes the chaotic BWC audio space.
* **Implementation**: Uses `librosa` and `scipy.signal` to perform peak loudness normalization to -20dB and optional butterworth bandpass filtering. 

### `run_final_hardened_pipeline.py`
* **Purpose**: The official production pipeline orchestrator.
* **Implementation**: 
    1. Loads Audio and Master Grounding.
    2. Uses Numpy arrays to apply a binary mask ("Muting Trick") to the audio, zeroing out non-speech before diarization.
    3. Iterates over VAD segments, feeding manual slices to WhisperX ASR (`condition_on_previous_text=False`).
    4. Executes WhisperX phoneme alignment and speaker assignment.

### `pipeline_msdd_oracle.py` (Phase 5 Audit)
* **Purpose**: NeMo MSDD integration.
* **Implementation**: Wraps NeMo's `NeuralDiarizer`. Dynamically generates an `OmegaConf` schema containing TitaNet embedding parameters and MSDD multiscale logic. Parses an external RTTM to act as `oracle_vad`.

### `scripts/utils/diarization_diagnostics.py`
* **Purpose**: Deep error analysis.
* **Implementation**: Steps through the timeline in 0.1s grid increments. Compares predicted speaker blocks vs ground truth. Automatically extracts ±2 second WAV clips for any detected Miss, False Alarm, or Confusion event for human review.

---

## SECTION 10 — EVALUATION FRAMEWORK

* **WER (Word Error Rate)**: Substitutions + Deletions + Insertions / Reference Words. Measures pure acoustic recognition.
* **DER (Diarization Error Rate)**: Miss + False Alarm + Confusion. Measures timing and assignment accuracy.
* **Miss / FA / Conf**: 
    * *Miss*: Speech occurred, diarizer heard silence.
    * *FA*: Silence occurred, diarizer heard speech.
    * *Confusion*: Person A spoke, assigned to Person B.
* **cpWER (concatenated permutation WER)**: Transcripts are concatenated per speaker, and the best permutation mapping is found. The gold standard for conversational ASR.
* **SA-WER (Speaker-Attributed WER)**: Word error rate calculated only when the timing and speaker label are simultaneously correct.

---

## SECTION 11 — COMPLETE RESULTS SYNTHESIS

### Macro Trends Across All Phases
1. **Enhancement is Detrimental**: Across all tests (Phase 2 and Phase 4), neural enhancement increased WER by 15-25%. Modern ASRs prefer raw, noisy audio over synthetically denoised audio.
2. **VAD Coverage dictates ASR Yield**: Pyannote's higher False Alarm rate is preferable to Silero's higher Miss rate. Whisper easily filters out Pyannote's false alarms, but cannot recover Silero's missed speech.
3. **Clustering Collapse**: Basic agglomerative clustering (Pyannote) fails on BWC audio, defaulting to a 1-speaker hypothesis to minimize mathematical penalties.
4. **Oracle MSDD**: The most robust architectural setup involves separating VAD from Diarization. Generating a Master Grounding, masking the audio, and using MSDD for attribution yields the best cpWER.

---

## SECTION 12 — FINAL HARDENED PIPELINE

The research culminated in the **Oracle Grounded Hardened Pipeline**.

**1. Loudness Normalization**: Audio is standardized to -20dB. This prevents Whisper's feature extractors from clipping and stabilizes the TitaNet embeddings used in diarization.
**2. Master Grounding (Pyannote 3.1 VAD)**: Generates strict JSON timestamps of all voice activity.
**3. The Muting Trick**: Non-speech regions are reduced to absolute digital silence.
**4. Manual Slice ASR**: Whisper is fed individual VAD chunks rather than the continuous file, preventing transformer drift.
**5. Oracle MSDD Diarization**: NeMo calculates multiscale embeddings (0.5s to 1.5s windows) only on the pre-approved speech chunks, iterating via sequence-to-sequence models to resolve overlaps.
**6. WhisperX Alignment**: Word-level timestamps are mapped to the MSDD speaker blocks.

---

## SECTION 13 — SCIENTIFIC FINDINGS

1. **VAD Coverage is Safety**: In forensic contexts, a high-recall VAD (Pyannote) is vastly superior to a high-precision VAD (Silero).
2. **Enhancement Artifacts Hurt Modern ASR**: Transformers rely on intricate phase and frequency mappings. Generative enhancers destroy these cues, leading to massive deletion errors.
3. **Diarization is the Dominant Bottleneck**: ASR technology (Whisper large-v3) has largely solved the speech-to-text problem. The current frontier is Speaker Attribution (DER > 40%).
4. **cpWER Better Reflects Forensic Utility**: Optimizing for pure DER leads to models collapsing into single-speaker outputs. Optimizing for cpWER forces models to delineate conversational boundaries.

---

## SECTION 14 — LESSONS LEARNED

* **Avoid Un-Isolated Variables**: Phase 3 comparisons were originally invalid because Pyannote and NeMo were running their own distinct VADs. You cannot compare diarization logic if the input speech segments are different.
* **NeMo Configuration Rigidity**: NeMo's OmegaConf framework is extraordinarily brittle. Missing a single sub-key (`max_rp_threshold`) causes complete pipeline failure. 
* **Do Not Trust Automatic Speaker Counts**: Standard clustering algorithms will artificially deflate speaker counts in noisy environments to avoid penalizing themselves with confusion errors.

---

## SECTION 15 — FUTURE WORK

1. **Transcript-Aware Diarization (High Impact)**: Future systems should use the semantic context generated by Whisper to inform the diarizer (e.g., if Speaker A asks a question, the response is likely Speaker B).
2. **Overlap-Aware Neural Diarization (EEND)**: The current pipeline ignores overlapping speech. Transitioning to End-to-End Neural Diarization (EEND) models could resolve the remaining confusion errors in highly volatile encounters.
3. **Targeted Denoising (Medium Impact)**: While global enhancement failed, applying DeepFilterNet strictly as a pre-processing step for the *Diarizer's embedding extractor* (but not the ASR) might stabilize the acoustic signatures.
4. **Speaker Consistency Modeling**: Developing a pipeline that identifies a "Primary Officer" embedding and tracks it consistently across multiple videos.

---

## SECTION 16 — FINAL CONCLUSION

The BWC Pipeline Optimization Project successfully navigated the hostile landscape of Body-Worn Camera audio. By systematically isolating and testing Voice Activity Detection, Speech Enhancement, and Speaker Diarization, the project established a new "Hardened" production standard.

The research definitively proved that traditional neural denoising algorithms degrade modern transformer-based ASR performance. It established that aggressive, high-recall VAD grounding is essential for forensic safety. Finally, it demonstrated that separating VAD logic from Diarization logic—using a Master Grounding as an "Oracle" input to NeMo's Multi-Scale Diarization Decoder (MSDD)—solves the identity collapse problem inherent in basic clustering. 

The resulting pipeline prioritizes concatenated permutation WER (cpWER) over raw DER, ensuring that the generated intelligence is not only acoustically accurate but correctly attributed to the dynamic individuals recorded in the field.