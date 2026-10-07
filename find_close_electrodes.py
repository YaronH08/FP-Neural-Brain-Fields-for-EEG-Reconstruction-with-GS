from pathlib import Path
import mne
import numpy as np
import csv
from itertools import combinations

# ==============================
# CHANGE THIS PATH
# ==============================

fif_path = Path(
    "/home/yaroh/NBF-GS/GS-Files/brennan2019_processed/S01/meg-sr120-hp0-filtered-validation-huber-std005-generated-raw.fif"
)

# ==============================
# VOLTAGE TIME TO PRINT
# ==============================
# The voltage is a time signal, so choose which time sample to print
# Example: 0.0 means the first sample, 10.0 means voltage at second 10
time_sec = 10

# Threshold in mm
# Print 2 electrodes that the distance between then less then 20mm
threshold_mm = 20.0

# How much pairs to print
top_k = 30

# The new electrodes
new_electrodes = [
    "T7", "T8", "TP7", "TP8",
    "FC5", "FC6", "FT7", "FT8",
    "P7", "P8", "CP5", "CP6",
    "C5", "C6", "F7", "F8",
    "P5", "P6"
]

if not fif_path.exists():
    raise FileNotFoundError(f"File does not exist: {fif_path}")

raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=True)

# ==============================
# GET SAMPLE INDEX FOR VOLTAGE
# ==============================

sfreq = raw.info["sfreq"]
sample_idx = int(round(time_sec * sfreq))

if sample_idx < 0:
    raise ValueError("time_sec must be >= 0")

if sample_idx >= raw.n_times:
    raise ValueError(
        f"time_sec={time_sec} is outside the recording. "
        f"Max time is approximately {raw.times[-1]:.3f} sec"
    )

actual_time_sec = raw.times[sample_idx]

print("\n==============================")
print("VOLTAGE SAMPLE INFO")
print("==============================")
print(f"Requested time: {time_sec:.6f} sec")
print(f"Actual sample index: {sample_idx}")
print(f"Actual sample time: {actual_time_sec:.6f} sec")
print(f"Sampling frequency: {sfreq:.2f} Hz")

# ==============================
# READ VOLTAGE OF ALL CHANNELS AT THE REQUESTED TIME
# ==============================

# Shape: number_of_channels x 1
sample_data = raw.get_data(
    start=sample_idx,
    stop=sample_idx + 1
)[:, 0]

voltage_at_time = {}

for idx, ch_name in enumerate(raw.ch_names):
    voltage_at_time[ch_name] = sample_data[idx]

# ==============================
# EXTRACT VALID POSITIONS
# ==============================

positions = {}

for idx, ch_name in enumerate(raw.ch_names):
    loc = raw.info["chs"][idx]["loc"][:3]

    # check place
    if np.any(np.isnan(loc)):
        print(f"Skipping {ch_name}: position is NaN")
        continue

    if np.allclose(loc, [0.0, 0.0, 0.0]):
        print(f"Skipping {ch_name}: position is zero")
        continue

    positions[ch_name] = loc

print("\n==============================")
print("VALID CHANNEL POSITIONS")
print("==============================")
print(f"Valid channels with position: {len(positions)} / {len(raw.ch_names)}")

# ==============================
# COMPUTE ALL PAIRWISE DISTANCES
# ==============================

pairs = []

for ch1, ch2 in combinations(positions.keys(), 2):
    p1 = positions[ch1]
    p2 = positions[ch2]

    distance_m = np.linalg.norm(p1 - p2)
    distance_mm = distance_m * 1000.0

    pairs.append((ch1, ch2, distance_mm))

# sort from the closest to the furthest
pairs.sort(key=lambda x: x[2])

# ==============================
# PRINT TOP K CLOSEST PAIRS
# ==============================

print("\n==============================")
print(f"TOP {top_k} CLOSEST ELECTRODE PAIRS")
print("==============================\n")

for ch1, ch2, dist in pairs[:top_k]:
    print(f"{ch1:>5}  <-->  {ch2:<5} | distance = {dist:.2f} mm")

# ==============================
# PRINT ONLY PAIRS BELOW THRESHOLD
# ==============================

print("\n==============================")
print(f"PAIRS CLOSER THAN {threshold_mm} mm")
print("==============================\n")

close_pairs = [pair for pair in pairs if pair[2] <= threshold_mm]

if len(close_pairs) == 0:
    print(f"No pairs found closer than {threshold_mm} mm")
else:
    for ch1, ch2, dist in close_pairs:
        print(f"{ch1:>5}  <-->  {ch2:<5} | distance = {dist:.2f} mm")

# ==============================
# FOR EACH NEW ELECTRODE:
# FIND CLOSEST EXISTING ELECTRODES
# PRINT NEW ELECTRODE LOCATION IN METERS + VOLTAGE IN VOLTS
# ALSO PRINT VOLTAGE OF EACH NEAREST ELECTRODE AT THE SAME TIME
# ==============================

print("\n===================================================")
print("NEAREST ELECTRODES FOR EACH NEW ELECTRODE")
print("INCLUDING NEW ELECTRODE LOCATION [m] AND VOLTAGE [V]")
print("INCLUDING NEAREST ELECTRODES VOLTAGE [V]")
print("==================================================\n")

new_electrode_rows = []

for new_ch in new_electrodes:
    if new_ch not in positions:
        print(f"{new_ch}: NOT FOUND or invalid position")
        continue

    if new_ch not in raw.ch_names:
        print(f"{new_ch}: NOT FOUND in raw.ch_names")
        continue

    # Get location of the new electrode in meters
    new_pos = positions[new_ch]
    x, y, z = new_pos

    # Get voltage of the new electrode at the requested time
    # MNE usually stores EEG voltage in Volts
    new_voltage_v = voltage_at_time[new_ch]

    distances = []

    for other_ch, other_pos in positions.items():
        if other_ch == new_ch:
            continue

        # Skip the other new electrodes, so we only show original/existing electrodes
        if other_ch in new_electrodes:
            continue

        dist_mm = np.linalg.norm(positions[new_ch] - other_pos) * 1000.0

        # Get voltage of the existing/nearest electrode at the same time
        other_voltage_v = voltage_at_time[other_ch]

        distances.append((other_ch, dist_mm, other_voltage_v))

    distances.sort(key=lambda x: x[1])

    print(f"\n{new_ch}:")
    print(f"    location [m]: x={x:.6f}, y={y:.6f}, z={z:.6f}")
    print(f"    voltage (t={actual_time_sec:.2f} [sec]): {new_voltage_v:.12e} [V]")

    print("    5 nearest existing electrodes:")
    for rank, (other_ch, dist_mm, other_voltage_v) in enumerate(distances[:5], start=1):
        print(
            f"        {rank}. nearest: {other_ch:>5} | "
            f"distance = {dist_mm:.2f} mm | "
            f"voltage = {other_voltage_v:.12e} [V]"
        )

        new_electrode_rows.append([
            new_ch,
            x,
            y,
            z,
            actual_time_sec,
            new_voltage_v,
            rank,
            other_ch,
            dist_mm,
            other_voltage_v
        ])

# ==============================
# SIGNAL STATISTICS FOR NEW ELECTRODES VS NEAREST EXISTING ELECTRODES
# ==============================

print("\n===================================================")
print("SIGNAL STATISTICS OVER FULL RECORDING")
print("MEAN / STD / MIN / MAX / RMS")
print("NEW ELECTRODES VS 5 NEAREST EXISTING ELECTRODES")
print("==================================================\n")

# Load the full data matrix once
# Shape: [n_channels, n_times]
all_data = raw.get_data()

channel_to_index = {
    ch_name: idx for idx, ch_name in enumerate(raw.ch_names)
}


def compute_signal_stats(signal):
    """
    Compute basic statistics for one EEG channel signal.
    Signal is assumed to be in Volts.
    """
    mean_v = np.mean(signal)
    std_v = np.std(signal)
    min_v = np.min(signal)
    max_v = np.max(signal)
    rms_v = np.sqrt(np.mean(signal ** 2))

    return {
        "mean_v": mean_v,
        "std_v": std_v,
        "min_v": min_v,
        "max_v": max_v,
        "rms_v": rms_v,
    }


def v_to_uv(value_v):
    """
    Convert Volts to microvolts.
    """
    return value_v * 1e6


stats_detail_rows = []
stats_summary_rows = []

for new_ch in new_electrodes:
    if new_ch not in positions:
        print(f"{new_ch}: NOT FOUND or invalid position - skipping statistics")
        continue

    if new_ch not in channel_to_index:
        print(f"{new_ch}: NOT FOUND in raw.ch_names - skipping statistics")
        continue

    new_idx = channel_to_index[new_ch]
    new_signal = all_data[new_idx, :]
    new_stats = compute_signal_stats(new_signal)

    # Find nearest existing electrodes only
    nearest_existing = []

    for other_ch, other_pos in positions.items():
        if other_ch == new_ch:
            continue

        # Skip other generated/new electrodes
        if other_ch in new_electrodes:
            continue

        dist_mm = np.linalg.norm(positions[new_ch] - other_pos) * 1000.0
        nearest_existing.append((other_ch, dist_mm))

    nearest_existing.sort(key=lambda x: x[1])
    nearest_existing = nearest_existing[:5]

    nearest_stats_list = []

    print(f"\n{new_ch}:")
    print(
        "    NEW electrode stats: "
        f"mean={v_to_uv(new_stats['mean_v']): .6f} uV | "
        f"std={v_to_uv(new_stats['std_v']): .6f} uV | "
        f"min={v_to_uv(new_stats['min_v']): .6f} uV | "
        f"max={v_to_uv(new_stats['max_v']): .6f} uV | "
        f"rms={v_to_uv(new_stats['rms_v']): .6f} uV"
    )

    print("    5 nearest existing electrodes statistics:")

    for rank, (other_ch, dist_mm) in enumerate(nearest_existing, start=1):
        other_idx = channel_to_index[other_ch]
        other_signal = all_data[other_idx, :]
        other_stats = compute_signal_stats(other_signal)
        nearest_stats_list.append(other_stats)

        print(
            f"        {rank}. {other_ch:>5} | "
            f"distance={dist_mm: .2f} mm | "
            f"mean={v_to_uv(other_stats['mean_v']): .6f} uV | "
            f"std={v_to_uv(other_stats['std_v']): .6f} uV | "
            f"min={v_to_uv(other_stats['min_v']): .6f} uV | "
            f"max={v_to_uv(other_stats['max_v']): .6f} uV | "
            f"rms={v_to_uv(other_stats['rms_v']): .6f} uV"
        )

        stats_detail_rows.append([
            new_ch,
            rank,
            other_ch,
            dist_mm,

            new_stats["mean_v"],
            new_stats["std_v"],
            new_stats["min_v"],
            new_stats["max_v"],
            new_stats["rms_v"],

            other_stats["mean_v"],
            other_stats["std_v"],
            other_stats["min_v"],
            other_stats["max_v"],
            other_stats["rms_v"],

            v_to_uv(new_stats["mean_v"]),
            v_to_uv(new_stats["std_v"]),
            v_to_uv(new_stats["min_v"]),
            v_to_uv(new_stats["max_v"]),
            v_to_uv(new_stats["rms_v"]),

            v_to_uv(other_stats["mean_v"]),
            v_to_uv(other_stats["std_v"]),
            v_to_uv(other_stats["min_v"]),
            v_to_uv(other_stats["max_v"]),
            v_to_uv(other_stats["rms_v"]),
        ])

    if len(nearest_stats_list) > 0:
        nearest_mean_std_v = np.mean([s["std_v"] for s in nearest_stats_list])
        nearest_mean_rms_v = np.mean([s["rms_v"] for s in nearest_stats_list])

        eps = 1e-20
        std_ratio = new_stats["std_v"] / (nearest_mean_std_v + eps)
        rms_ratio = new_stats["rms_v"] / (nearest_mean_rms_v + eps)

        print(
            "    SUMMARY compared to nearest existing electrodes: "
            f"new_std / mean_neighbor_std = {std_ratio:.6f} | "
            f"new_rms / mean_neighbor_rms = {rms_ratio:.6f}"
        )

        if std_ratio < 0.3:
            print("    WARNING: new electrode STD is much lower than nearby real electrodes")
        elif std_ratio > 2.0:
            print("    WARNING: new electrode STD is much higher than nearby real electrodes")
        else:
            print("    STD ratio looks reasonable")

        stats_summary_rows.append([
            new_ch,

            new_stats["mean_v"],
            new_stats["std_v"],
            new_stats["min_v"],
            new_stats["max_v"],
            new_stats["rms_v"],

            nearest_mean_std_v,
            nearest_mean_rms_v,
            std_ratio,
            rms_ratio,

            v_to_uv(new_stats["mean_v"]),
            v_to_uv(new_stats["std_v"]),
            v_to_uv(new_stats["min_v"]),
            v_to_uv(new_stats["max_v"]),
            v_to_uv(new_stats["rms_v"]),

            v_to_uv(nearest_mean_std_v),
            v_to_uv(nearest_mean_rms_v),
        ])

# ==============================
# SAVE SIGNAL STATISTICS TO CSV
# ==============================

stats_detail_csv_path = fif_path.parent / "new_electrodes_signal_stats_detail.csv"

with open(stats_detail_csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "new_channel",
        "nearest_rank",
        "nearest_existing_channel",
        "distance_mm",

        "new_mean_v",
        "new_std_v",
        "new_min_v",
        "new_max_v",
        "new_rms_v",

        "nearest_mean_v",
        "nearest_std_v",
        "nearest_min_v",
        "nearest_max_v",
        "nearest_rms_v",

        "new_mean_uV",
        "new_std_uV",
        "new_min_uV",
        "new_max_uV",
        "new_rms_uV",

        "nearest_mean_uV",
        "nearest_std_uV",
        "nearest_min_uV",
        "nearest_max_uV",
        "nearest_rms_uV",
    ])
    writer.writerows(stats_detail_rows)

print("\n==============================")
print("SIGNAL STATS DETAIL CSV SAVED")
print("==============================")
print(stats_detail_csv_path)


stats_summary_csv_path = fif_path.parent / "new_electrodes_signal_stats_summary.csv"

with open(stats_summary_csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "new_channel",

        "new_mean_v",
        "new_std_v",
        "new_min_v",
        "new_max_v",
        "new_rms_v",

        "mean_neighbor_std_v",
        "mean_neighbor_rms_v",
        "std_ratio_new_over_neighbors",
        "rms_ratio_new_over_neighbors",

        "new_mean_uV",
        "new_std_uV",
        "new_min_uV",
        "new_max_uV",
        "new_rms_uV",

        "mean_neighbor_std_uV",
        "mean_neighbor_rms_uV",
    ])
    writer.writerows(stats_summary_rows)

print("\n==============================")
print("SIGNAL STATS SUMMARY CSV SAVED")
print("==============================")
print(stats_summary_csv_path)

# ==============================
# SAVE RESULTS TO CSV
# ==============================

csv_path = fif_path.parent / "electrode_distances_sorted.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["channel_1", "channel_2", "distance_mm"])
    writer.writerows(pairs)

print("\n==============================")
print("CSV SAVED")
print("==============================")
print(csv_path)

# ==============================
# SAVE NEW ELECTRODE DETAILS TO CSV
# ==============================

new_csv_path = fif_path.parent / "new_electrodes_nearest_with_voltage.csv"

with open(new_csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "new_channel",
        "x_m",
        "y_m",
        "z_m",
        "time_sec",
        "new_voltage_v",
        "nearest_rank",
        "nearest_existing_channel",
        "distance_mm",
        "nearest_existing_voltage_v"
    ])
    writer.writerows(new_electrode_rows)

print("\n==============================")
print("NEW ELECTRODES CSV SAVED")
print("==============================")
print(new_csv_path)
