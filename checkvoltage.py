from pathlib import Path
import mne
import numpy as np

original_path = Path(
    "/home/yaroh/NBF-GS/GS-Files/brennan2019_processed/S01/meg-sr120-hp0-filtered-raw.fif"
)

generated_path = Path(
    "/home/yaroh/NBF-GS/GS-Files/brennan2019_processed/S01/meg-sr120-hp0-filtered-validation-4dgs-generated-raw.fif"
)
raw_orig = mne.io.read_raw_fif(original_path, preload=False, verbose=True)
raw_gen = mne.io.read_raw_fif(generated_path, preload=False, verbose=True)

time_sec = 10
sfreq = raw_gen.info["sfreq"]
sample_idx = int(round(time_sec * sfreq))

original_channels = [ch for ch in raw_orig.ch_names if ch in raw_gen.ch_names]

print("\n==============================")
print("COMPARE ORIGINAL CHANNELS")
print("==============================")

for ch in original_channels[:20]:
    v_orig = raw_orig.get_data(picks=[ch], start=sample_idx, stop=sample_idx + 1)[0, 0]
    v_gen = raw_gen.get_data(picks=[ch], start=sample_idx, stop=sample_idx + 1)[0, 0]
    diff = v_gen - v_orig

    print(
        f"{ch:>5} | "
        f"original = {v_orig:.12e} V | "
        f"generated = {v_gen:.12e} V | "
        f"diff = {diff:.12e} V"
    )
