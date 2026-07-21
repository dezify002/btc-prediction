"""
Run this in your btc prediction folder to check if files are updated.
"""
import os

files_to_check = {
    "features/indicators.py": ["atr_14", "realized_vol_20", "atr_pct_rank_100"],
    "models/baseline_xgb.py": ["cal_probs_fit = model.predict_proba"],
    "models/train_final_model.py": ['barrier_mode="pct"'],
    "models/prediction_core.py": ["time_decay_factor", "vol_regime_multiplier"],
    "models/price_target_probability.py": ["time_decay_factor"],
    "webapp/app.py": ["regime_warning"],
}

print("Checking which files have been updated...\n")

for path, markers in files_to_check.items():
    full_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(full_path):
        print(f"❌ {path} — FILE NOT FOUND")
        continue

    with open(full_path, "r") as f:
        content = f.read()

    found = all(marker in content for marker in markers)
    status = "✅ UPDATED" if found else "❌ OLD VERSION"
    print(f"{status} — {path}")

    if not found:
        for marker in markers:
            if marker not in content:
                print(f"   Missing: '{marker}'")

print("\nIf any file shows ❌ OLD VERSION, you need to replace it with the updated file.")