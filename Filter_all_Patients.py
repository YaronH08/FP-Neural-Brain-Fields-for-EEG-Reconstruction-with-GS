import mne
import os
from pathlib import Path

base_dir = r"C:\NBF_DATA\brennan2019_processed"

print("=" * 60)
print("FILTER ALL PATIENTS")
print("=" * 60)
print("Base directory:", base_dir)

if not os.path.exists(base_dir):
    print("ERROR: base_dir does not exist.")
    print("Fix this path:")
    print(base_dir)
    raise SystemExit

print("Folders found in base_dir:")
print(os.listdir(base_dir)[:20])

for patient_num in range(1, 49):
    patient_dir = f"S{patient_num:02d}"

    patient_path = os.path.join(base_dir, patient_dir)
    fif_path = os.path.join(patient_path, "meg-sr120-hp0-raw.fif")

    if not os.path.exists(patient_path):
        print(f"skipping {patient_dir}: directory does not exist at {patient_path}")
        continue

    if not os.path.exists(fif_path):
        print(f"skipping {patient_dir}: FIF file does not exist at {fif_path}")
        continue

    try:
        print(f"\nProcessing {patient_dir}")
        print("FIF path:", fif_path)

        raw = mne.io.read_raw_fif(fif_path, preload=True)

        raw.filter(l_freq=0.5, h_freq=59.5, fir_design="firwin")

        raw.save(fif_path, overwrite=True)

        print(f"successfully processed {patient_dir}")

    except Exception as e:
        print(f"error processing {patient_dir}: {str(e)}")