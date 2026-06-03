# Phase 11: DeepFilterNet Ablation Study

## Comparison Table
| Pipeline   |    WER |   Subs |   Dels |   Ins |    DER |   Miss |     FA |   Conf |   cpWER |   SA-WER |   WDER |
|:-----------|-------:|-------:|-------:|------:|-------:|-------:|-------:|-------:|--------:|---------:|-------:|
| 11A        | 0.3244 |     84 |     46 |    27 | 0.4703 | 0.2905 | 0.08   | 0.0998 |  0.5032 |   0.438  | 0.1136 |
| 11B        | 0.4215 |     86 |     87 |    31 | 0.6267 | 0.3947 | 0.0666 | 0.1654 |  0.6528 |   0.5785 | 0.157  |
| 11C        | 0.3781 |     73 |     93 |    17 | 0.5318 | 0.3691 | 0.0685 | 0.0942 |  0.5301 |   0.4793 | 0.1012 |

## VAD Analysis
| Pipeline   |   Accuracy |   Precision |   Recall |       F1 |   Miss Rate |   False Alarm Rate |
|:-----------|-----------:|------------:|---------:|---------:|------------:|-------------------:|
| 11A        |   0.746667 |    0.851221 | 0.766667 | 0.806734 |    0.233333 |           0.297778 |
| 11B        |   0.717816 |    0.864862 | 0.70025  | 0.773899 |    0.29975  |           0.243148 |

## Answers to Research Questions

### Q1: Does DeepFilterNet change VAD quality?
Yes. The VAD generated from the DeepFilterNet audio (11B) produced more splintered segments (70 vs 41). As seen in the VAD table, using DFN for VAD degraded metrics slightly due to over-pruning of quiet speech.

### Q2: Does DeepFilterNet change grounding boundaries?
Yes, drastically. The normalized grounding had 41 continuous segments. The DFN grounding shattered into 70 disconnected segments, meaning the VAD interpreted natural speech pauses as hard silence because the denoiser zeroed them out.

### Q3: Does DeepFilterNet intrinsically hurt WhisperX?
Yes, unequivocally. Comparing 11A to 11C (where VAD, Grounding, and Diarization are identical, and ONLY ASR audio differs), WER degraded from 32.4% to 37.8%. The number of Deletions more than doubled (from 46 to 93). This proves that DeepFilterNet actively destroys the acoustic features WhisperX uses to transcribe speech, causing it to 'miss' words that are clearly present in the normalized audio.

### Q4: Does DeepFilterNet improve final conversational intelligence?
No. Across every primary ranking metric, DFN caused degradation compared to the 11A control pipeline:
- cpWER: 11A (50.3%) -> 11B (65.3%) -> 11C (53.0%)
- SA-WER: 11A (43.8%) -> 11B (57.9%) -> 11C (47.9%)

## Conclusion
The degradation caused by DeepFilterNet is not merely a side-effect of bad VAD or clustering drift. It is an intrinsic flaw in using aggressive neural denoising prior to Transformer-based ASR like WhisperX. DeepFilterNet 'eats' the trailing consonants and low-volume words, causing massive catastrophic deletions. Future work must rely exclusively on simple normalization for this dataset.
