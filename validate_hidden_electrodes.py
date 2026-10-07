from pathlib import Path
import csv
import numpy as np
import mne


fif_path = Path(
    "/home/yaroh/NBF-GS/GS-Files/brennan2019_processed/S01/meg-sr120-hp0-filtered-validation-4dgs-generated-raw.fif"
)

validation_pairs = [
    ("17", "VAL_17"),
    ("11", "VAL_11"),
    ("52", "VAL_52"),
    ("57", "VAL_57"),
    ("19", "VAL_19"),
]


def compute_metrics(true_signal, pred_signal):
    true_signal = np.asarray(true_signal).flatten()
    pred_signal = np.asarray(pred_signal).flatten()

    eps = 1e-20
    error = pred_signal - true_signal

    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(mse)

    true_var = np.var(true_signal)

    r2 = 1.0 - (
        np.sum(error ** 2)
        / (np.sum((true_signal - np.mean(true_signal)) ** 2) + eps)
    )

    if np.std(true_signal) < eps or np.std(pred_signal) < eps:
        pcc = np.nan
    else:
        pcc = np.corrcoef(true_signal, pred_signal)[0, 1]

    signal_power = np.mean(true_signal ** 2)
    noise_power = np.mean(error ** 2)
    snr = 10.0 * np.log10(signal_power / (noise_power + eps))

    true_mean = np.mean(true_signal)
    pred_mean = np.mean(pred_signal)

    true_std = np.std(true_signal)
    pred_std = np.std(pred_signal)

    true_min = np.min(true_signal)
    pred_min = np.min(pred_signal)

    true_max = np.max(true_signal)
    pred_max = np.max(pred_signal)

    true_rms = np.sqrt(np.mean(true_signal ** 2))
    pred_rms = np.sqrt(np.mean(pred_signal ** 2))

    std_ratio = pred_std / (true_std + eps)
    rms_ratio = pred_rms / (true_rms + eps)

    nmse = mse / (true_var + eps)

    return {
        "mse_v2": mse,
        "mae_v": mae,
        "rmse_v": rmse,
        "r2": r2,
        "pcc": pcc,
        "snr_db": snr,
        "nmse": nmse,

        "true_mean_v": true_mean,
        "pred_mean_v": pred_mean,
        "true_std_v": true_std,
        "pred_std_v": pred_std,
        "true_min_v": true_min,
        "pred_min_v": pred_min,
        "true_max_v": true_max,
        "pred_max_v": pred_max,
        "true_rms_v": true_rms,
        "pred_rms_v": pred_rms,
        "std_ratio_pred_over_true": std_ratio,
        "rms_ratio_pred_over_true": rms_ratio,

        "mae_uV": mae * 1e6,
        "rmse_uV": rmse * 1e6,
        "true_mean_uV": true_mean * 1e6,
        "pred_mean_uV": pred_mean * 1e6,
        "true_std_uV": true_std * 1e6,
        "pred_std_uV": pred_std * 1e6,
        "true_min_uV": true_min * 1e6,
        "pred_min_uV": pred_min * 1e6,
        "true_max_uV": true_max * 1e6,
        "pred_max_uV": pred_max * 1e6,
        "true_rms_uV": true_rms * 1e6,
        "pred_rms_uV": pred_rms * 1e6,
    }


if not fif_path.exists():
    raise FileNotFoundError(f"File does not exist: {fif_path}")

raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=True)

print("\n==============================")
print("VALIDATION HIDDEN ELECTRODES")
print("==============================")
print(f"File: {fif_path}")
print(f"Channels: {len(raw.ch_names)}")
print(f"Samples: {raw.n_times}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")

rows = []

for true_ch, pred_ch in validation_pairs:
    if true_ch not in raw.ch_names:
        print(f"Missing true channel: {true_ch}")
        continue

    if pred_ch not in raw.ch_names:
        print(f"Missing predicted channel: {pred_ch}")
        continue

    true_signal = raw.get_data(picks=[true_ch])[0]
    pred_signal = raw.get_data(picks=[pred_ch])[0]

    metrics = compute_metrics(true_signal, pred_signal)

    print("\n" + "-" * 60)
    print(f"{true_ch}  vs  {pred_ch}")
    print("-" * 60)
    print(f"PCC:       {metrics['pcc']:.6f}")
    print(f"R2:        {metrics['r2']:.6f}")
    print(f"SNR:       {metrics['snr_db']:.6f} dB")
    print(f"MAE:       {metrics['mae_uV']:.6f} uV")
    print(f"RMSE:      {metrics['rmse_uV']:.6f} uV")
    print(f"NMSE:      {metrics['nmse']:.6f}")
    print(f"STD true:  {metrics['true_std_uV']:.6f} uV")
    print(f"STD pred:  {metrics['pred_std_uV']:.6f} uV")
    print(f"STD ratio: {metrics['std_ratio_pred_over_true']:.6f}")
    print(f"RMS true:  {metrics['true_rms_uV']:.6f} uV")
    print(f"RMS pred:  {metrics['pred_rms_uV']:.6f} uV")
    print(f"RMS ratio: {metrics['rms_ratio_pred_over_true']:.6f}")

    row = {
        "true_channel": true_ch,
        "pred_channel": pred_ch,
    }
    row.update(metrics)
    rows.append(row)


csv_path = fif_path.parent / "validation_hidden_electrodes_metrics.csv"

fieldnames = [
    "true_channel",
    "pred_channel",

    "mse_v2",
    "mae_v",
    "rmse_v",
    "mae_uV",
    "rmse_uV",

    "r2",
    "pcc",
    "snr_db",
    "nmse",

    "true_mean_v",
    "pred_mean_v",
    "true_std_v",
    "pred_std_v",
    "true_min_v",
    "pred_min_v",
    "true_max_v",
    "pred_max_v",
    "true_rms_v",
    "pred_rms_v",
    "std_ratio_pred_over_true",
    "rms_ratio_pred_over_true",

    "true_mean_uV",
    "pred_mean_uV",
    "true_std_uV",
    "pred_std_uV",
    "true_min_uV",
    "pred_min_uV",
    "true_max_uV",
    "pred_max_uV",
    "true_rms_uV",
    "pred_rms_uV",
]

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n==============================")
print("VALIDATION METRICS CSV SAVED")
print("==============================")
print(csv_path)

if len(rows) > 0:
    pcc_values = [r["pcc"] for r in rows if not np.isnan(r["pcc"])]
    r2_values = [r["r2"] for r in rows]
    mae_values = [r["mae_uV"] for r in rows]
    rmse_values = [r["rmse_uV"] for r in rows]
    std_ratios = [r["std_ratio_pred_over_true"] for r in rows]
    rms_ratios = [r["rms_ratio_pred_over_true"] for r in rows]

    print("\n==============================")
    print("SUMMARY")
    print("==============================")
    print(f"Mean PCC:       {np.mean(pcc_values):.6f}")
    print(f"Mean R2:        {np.mean(r2_values):.6f}")
    print(f"Mean MAE:       {np.mean(mae_values):.6f} uV")
    print(f"Mean RMSE:      {np.mean(rmse_values):.6f} uV")
    print(f"Mean STD ratio: {np.mean(std_ratios):.6f}")
    print(f"Mean RMS ratio: {np.mean(rms_ratios):.6f}")
