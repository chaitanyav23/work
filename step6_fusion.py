import json
from pathlib import Path
from collections import Counter
import datetime

def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_msdd_segments_from_rttm(rttm_path):
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append({"start": start, "end": start+duration, "speaker": speaker})
    return segments

def assign_speakers_to_words(aligned_words, msdd_segments, tolerance=0.5):
    for word_info in aligned_words:
        if "start" not in word_info or "end" not in word_info or word_info["start"] is None:
            word_info["speaker"] = "UNK"
            continue
            
        w_start, w_end = word_info["start"], word_info["end"]
        
        overlaps = []
        for msdd in msdd_segments:
            if msdd["start"] <= w_end and msdd["end"] >= w_start:
                overlap_start = max(w_start, msdd["start"])
                overlap_end = min(w_end, msdd["end"])
                overlap_dur = overlap_end - overlap_start
                if overlap_dur > 0:
                    overlaps.append({
                        "speaker": msdd["speaker"],
                        "overlap_dur": overlap_dur
                    })
        
        assigned_spk = None
        if len(overlaps) >= 1:
            overlaps.sort(key=lambda x: x["overlap_dur"], reverse=True)
            assigned_spk = overlaps[0]["speaker"]
        else:
            # Fallback: nearest speaker turn within tolerance
            nearest_spk = None
            min_dist = float('inf')
            for msdd in msdd_segments:
                dist = min(abs(msdd["start"] - w_end), abs(msdd["end"] - w_start))
                if dist < min_dist:
                    min_dist = dist
                    nearest_spk = msdd["speaker"]
            
            if min_dist <= tolerance and nearest_spk:
                assigned_spk = nearest_spk
            else:
                assigned_spk = "UNK"
        
        word_info["speaker"] = assigned_spk
    
    return aligned_words

def group_words_into_segments(aligned_words):
    if not aligned_words:
        return []
        
    segments = []
    current_segment = None
    
    # We can group by speaker and some gap threshold, or just use the speaker change
    # For Phase 13, let's group by speaker change or gap > 2s
    
    for word in aligned_words:
        if word["word"] is None: continue
        
        if current_segment is None:
            current_segment = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "words": [word],
                "text": word["word"]
            }
        else:
            gap = (word["start"] - current_segment["end"]) if (word["start"] is not None and current_segment["end"] is not None) else 0
            if word["speaker"] == current_segment["speaker"] and gap < 2.0:
                current_segment["end"] = word["end"]
                current_segment["words"].append(word)
                current_segment["text"] += " " + word["word"]
            else:
                segments.append(current_segment)
                current_segment = {
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": word["speaker"],
                    "words": [word],
                    "text": word["word"]
                }
    
    if current_segment:
        segments.append(current_segment)
        
    return segments

def main():
    BASE_DIR = Path("experiments/phase13_karen_generalization")
    ALIGNED_WORDS_PATH = BASE_DIR / "outputs" / "aligned_words.json"
    MSDD_RTTM_PATH = BASE_DIR / "diarization" / "karen_msdd.rttm"
    OUTPUT_JSON = BASE_DIR / "outputs" / "karen_final.json"
    OUTPUT_TXT = BASE_DIR / "outputs" / "karen_final.txt"

    # 1. Load data
    with open(ALIGNED_WORDS_PATH) as f:
        aligned_words = json.load(f)
    
    msdd_segments = get_msdd_segments_from_rttm(MSDD_RTTM_PATH)

    # 2. Assign Speakers
    attributed_words = assign_speakers_to_words(aligned_words, msdd_segments)

    # 3. Group into readable segments
    segments = group_words_into_segments(attributed_words)

    # 4. Save JSON
    output_data = {
        "file": "karen_normalized.wav",
        "words": attributed_words,
        "segments": segments
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)

    # 5. Save TXT
    # Map speaker_0 -> Speaker_1 etc for readability
    spk_map = {}
    next_idx = 1
    
    with open(OUTPUT_TXT, "w") as f:
        for s in segments:
            raw_spk = s["speaker"]
            if raw_spk not in spk_map and raw_spk != "UNK":
                spk_map[raw_spk] = f"Speaker_{next_idx}"
                next_idx += 1
            
            friendly_spk = spk_map.get(raw_spk, "Speaker_Unknown")
            start_ts = format_timestamp(s['start']) if s['start'] is not None else "00:00:00"
            end_ts = format_timestamp(s['end']) if s['end'] is not None else "00:00:00"
            
            f.write(f"{friendly_spk} [{start_ts} - {end_ts}]\n")
            f.write(f"{s['text']}\n\n")

    print(f"Fusion complete. Saved to {OUTPUT_JSON} and {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
