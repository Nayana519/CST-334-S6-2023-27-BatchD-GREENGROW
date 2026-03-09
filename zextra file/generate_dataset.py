"""
generate_dataset.py
────────────────────────────────────────────────────────────────
Generates a realistic, BALANCED fertilizer recommendation dataset.

Strategy:
  - For each target fertilizer, define the NPK deficit profile that
    agronomically justifies that fertilizer.
  - Sample soil NPK values around those profiles using the crop's
    optimal NPK requirements.
  - This guarantees every fertilizer class gets ~1143 rows (8000 / 7).

Fertilizers & their agronomic trigger:
  Urea (46% N)            → N deficit dominant (>55% of total deficit)
  DAP (18:46:0)           → P deficit dominant (>50%), some N deficit
  NPK Complex 14:35:14    → P moderate-high, K also deficient
  Iffco NPK 17:17:17      → Balanced N, P, K deficit
  Kribhco NPK 20:20:0     → N + K deficit dominant, P sufficient
  MAP 28:28:0             → N and P roughly equal, low K deficit
  Tata Paras NPK 10:26:26 → P + K dominant, low N deficit
"""

import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# ── Crop optimal NPK ────────────────────────────────────────────────────────
CROPS = {
    'Maize':       (35, 26, 30),
    'Sugarcane':   (25, 30, 20),
    'Cotton':      (20, 20, 20),
    'Tobacco':     (30, 25, 25),
    'Paddy':       (35, 20, 20),
    'Barley':      (25, 25, 20),
    'Wheat':       (40, 20, 20),
    'Millets':     (20, 20, 15),
    'Oil seeds':   (15, 25, 15),
    'Pulses':      (10, 30, 10),
    'Ground Nuts': (15, 30, 15),
}

SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']

ROWS_PER_FERT = 1143   # ~8001 total (≈ 7 × 1143 + extras)

# ── Fertilizer deficit profiles ─────────────────────────────────────────────
# Each entry:  (frac_n_range, frac_p_range, frac_k_range)
# Fractions describe what % of the crop's OPTIMAL is MISSING from the soil.
# e.g. frac_n=0.70 means soil Nitrogen = 30% of the crop's optimal N target.
FERT_PROFILES = {
    'Urea (46% N)': {
        # High N deficit: soil N very low (20-50% of optimal)
        # P and K mostly sufficient: soil P/K at 70-100%+ of optimal
        'n_soil_frac': (0.10, 0.45),   # soil N is 10-45% of crop optimum
        'p_soil_frac': (0.70, 1.10),   # soil P is 70-110% of crop optimum
        'k_soil_frac': (0.72, 1.10),   # soil K is 72-110% of crop optimum
    },
    'DAP (18:46:0)': {
        # High P deficit: soil P very low
        # Some N deficit but not dominant; K mostly OK
        'n_soil_frac': (0.45, 0.78),
        'p_soil_frac': (0.05, 0.42),   # soil P is 5-42% of crop optimum
        'k_soil_frac': (0.70, 1.10),
    },
    'NPK Complex 14:35:14': {
        # P is highest deficit, K also deficient, N moderate
        'n_soil_frac': (0.40, 0.75),
        'p_soil_frac': (0.10, 0.50),
        'k_soil_frac': (0.20, 0.60),
    },
    'Iffco NPK 17:17:17': {
        # All three moderately deficient — balanced NPK needed
        'n_soil_frac': (0.30, 0.70),
        'p_soil_frac': (0.30, 0.70),
        'k_soil_frac': (0.30, 0.70),
    },
    'Kribhco NPK 20:20:0': {
        # N and K both low, P sufficient (rain-leached K)
        'n_soil_frac': (0.15, 0.55),
        'p_soil_frac': (0.72, 1.10),   # P mostly sufficient
        'k_soil_frac': (0.15, 0.55),   # K deficit similar to N
    },
    'MAP 28:28:0': {
        # N and P roughly equal deficit, K largely available
        'n_soil_frac': (0.25, 0.60),
        'p_soil_frac': (0.25, 0.60),
        'k_soil_frac': (0.65, 1.10),   # K mostly OK
    },
    'Tata Paras NPK 10:26:26': {
        # N mostly sufficient, P and K both very low (acidic/laterite soils)
        'n_soil_frac': (0.72, 1.10),   # N mostly sufficient
        'p_soil_frac': (0.05, 0.45),
        'k_soil_frac': (0.05, 0.45),
    },
}

# ── Regeneration logic ──────────────────────────────────────────────────────
def make_fert_block(fert_name: str, profile: dict, n_rows: int) -> pd.DataFrame:
    rows = []
    crop_list   = list(CROPS.keys())
    soil_list   = SOIL_TYPES

    for _ in range(n_rows):
        crop = rng.choice(crop_list)
        opt_n, opt_p, opt_k = CROPS[crop]
        soil = rng.choice(soil_list)

        # Sample soil NPK as a fraction of crop optimum
        n_frac = rng.uniform(*profile['n_soil_frac'])
        p_frac = rng.uniform(*profile['p_soil_frac'])
        k_frac = rng.uniform(*profile['k_soil_frac'])

        # Convert to integer soil values, clamp to 0-60
        soil_n = int(np.clip(round(opt_n * n_frac), 0, 60))
        soil_p = int(np.clip(round(opt_p * p_frac), 0, 60))
        soil_k = int(np.clip(round(opt_k * k_frac), 0, 60))

        # Realistic environmental values
        temp     = round(rng.uniform(15, 40), 2)
        humidity = round(rng.uniform(40, 90), 2)
        moisture = round(rng.uniform(20, 70), 2)
        ph       = round(rng.uniform(5.5, 7.5), 2)

        rows.append({
            'Temparature':    temp,
            'Humidity':       humidity,
            'Moisture':       moisture,
            'Soil Type':      soil,
            'Crop Type':      crop,
            'Nitrogen':       soil_n,
            'Potassium':      soil_k,
            'Phosphorous':    soil_p,
            'Fertilizer Name': fert_name,
            'pH':             ph,
        })

    return pd.DataFrame(rows)


# ── Build full dataset ──────────────────────────────────────────────────────
print("[INFO] Generating balanced fertilizer dataset...")
frames = []
for fert, profile in FERT_PROFILES.items():
    block = make_fert_block(fert, profile, ROWS_PER_FERT)
    print(f"  {fert:<35}: {len(block)} rows")
    frames.append(block)

df = pd.concat(frames, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"\n[INFO] Total rows: {len(df)}")
print("\n[INFO] Final distribution:")
vc = df['Fertilizer Name'].value_counts()
for k, v in vc.items():
    print(f"  {k:<35}: {v:>5} rows  ({100*v/len(df):.1f}%)")

df.to_csv('dataset.csv', index=False)
print("\n[SUCCESS] dataset.csv saved with balanced fertilizer distribution.")
