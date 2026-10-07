from pathlib import Path
import mne
import numpy as np
import csv

fif_path = Path(
    "/home/yaroh/NBF-GS/GS-Files/brennan2019_processed/S01/meg-sr120-hp0-filtered-validation-4dgs-generated-raw.fif"
)

if not fif_path.exists():
    raise FileNotFoundError(f"File does not exist: {fif_path}")

raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=True)

print("\n==============================")
print("CHANNEL NAMES + POSITIONS")
print("==============================\n")

rows = []

for idx, ch_name in enumerate(raw.ch_names):
    ch_info = raw.info["chs"][idx]

    # electrode place x, y, z
    loc = ch_info["loc"][:3]

    x, y, z = loc

    # check place
    is_nan = np.any(np.isnan(loc))
    is_zero = np.allclose(loc, [0.0, 0.0, 0.0])

    if is_nan:
        status = "INVALID - NaN"
    elif is_zero:
        status = "WARNING - ZERO POSITION"
    else:
        status = "OK"

    print(
        f"name={ch_name:>5} | "
        f"x={x: .6f} | "
        f"y={y: .6f} | "
        f"z={z: .6f} | "
        f"{status}"
    )

    rows.append([idx, ch_name, x, y, z, status])


# save CSV file
csv_path = fif_path.parent / "channel_positions.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "channel_name", "x", "y", "z", "status"])
    writer.writerows(rows)

print("\n==============================")
print("CSV SAVED")
print("==============================")
print(csv_path)
