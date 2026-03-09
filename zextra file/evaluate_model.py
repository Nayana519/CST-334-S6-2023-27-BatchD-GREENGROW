"""
evaluate_model.py
─────────────────────────────────────────────────────
Generates model evaluation visuals:
  1. Confusion Matrix heatmap  → static/confusion_matrix.png
  2. Per-class metrics bar chart → static/metrics_chart.png
  3. Prints full classification report to console
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs('static', exist_ok=True)

# ── Load model & encoders ────────────────────────────────────────────────────
print("[INFO] Loading model and encoders...")
model   = joblib.load('models/model.pkl')
le_soil = joblib.load('models/soil_encoder.pkl')
le_crop = joblib.load('models/crop_encoder.pkl')
le_fert = joblib.load('models/fertilizer_encoder.pkl')
OPTIMAL_NPK = joblib.load('models/optimal_npk.pkl')
FEATURE_COLS = joblib.load('models/feature_cols.pkl')

# ── Rebuild dataset exactly as in train_model.py ─────────────────────────────
print("[INFO] Loading and preprocessing dataset...")
data = pd.read_csv('dataset.csv')

FERT_NPK = {
    'Urea (46% N)':            np.array([46,  0,  0]),
    'DAP (18:46:0)':           np.array([18, 46,  0]),
    'NPK Complex 14:35:14':    np.array([14, 35, 14]),
    'Iffco NPK 17:17:17':      np.array([17, 17, 17]),
    'Kribhco NPK 20:20:0':     np.array([20,  0, 20]),
    'MAP 28:28:0':             np.array([28,  0, 28]),
    'Tata Paras NPK 10:26:26': np.array([10, 26, 26]),
}

def assign_fertilizer(row):
    opt = OPTIMAL_NPK.get(row['Crop Type'], (20, 20, 20))
    dn = max(0, opt[0] - row['Nitrogen'])
    dp = max(0, opt[1] - row['Phosphorous'])
    dk = max(0, opt[2] - row['Potassium'])
    total = dn + dp + dk
    if total == 0:
        return 'Iffco NPK 17:17:17'
    pn, pp, pk = dn/total, dp/total, dk/total
    if pn >= 0.55 and dn >= 8:                              return 'Urea (46% N)'
    if pp >= 0.50 and dp >= 8 and dn >= 2:                  return 'DAP (18:46:0)'
    if pn <= 0.22 and (pp+pk) >= 0.78 and (dp+dk) >= 10:   return 'Tata Paras NPK 10:26:26'
    if pp >= 0.38 and pk >= 0.14 and pp > pn:               return 'NPK Complex 14:35:14'
    if pn >= 0.35 and pk >= 0.28 and pp <= 0.25:            return 'Kribhco NPK 20:20:0'
    if pn >= 0.28 and pp >= 0.25 and pk <= 0.30 and abs(pn-pp) <= 0.15: return 'MAP 28:28:0'
    return 'Iffco NPK 17:17:17'

data['Fertilizer Name'] = data.apply(assign_fertilizer, axis=1)

def_n_list, def_p_list, def_k_list = [], [], []
for _, row in data.iterrows():
    opt = OPTIMAL_NPK.get(row['Crop Type'], (20, 20, 20))
    def_n_list.append(max(0, opt[0] - row['Nitrogen']))
    def_p_list.append(max(0, opt[1] - row['Phosphorous']))
    def_k_list.append(max(0, opt[2] - row['Potassium']))

data['def_n']         = def_n_list
data['def_p']         = def_p_list
data['def_k']         = def_k_list
data['total_deficit'] = data['def_n'] + data['def_p'] + data['def_k']
data['pct_n'] = data['def_n'] / data['total_deficit'].replace(0, 1)
data['pct_p'] = data['def_p'] / data['total_deficit'].replace(0, 1)
data['pct_k'] = data['def_k'] / data['total_deficit'].replace(0, 1)

data['Soil Type']       = le_soil.transform(data['Soil Type'])
data['Crop Type']       = le_crop.transform(data['Crop Type'])
data['Fertilizer Name'] = le_fert.transform(data['Fertilizer Name'])

X = data[FEATURE_COLS]
y = data['Fertilizer Name']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)
class_names = le_fert.classes_

# ── Print classification report ───────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'='*60}")
print(f"  MODEL EVALUATION REPORT")
print(f"{'='*60}")
print(f"  Test Accuracy  : {acc*100:.2f}%")
print(f"  Weighted F1    : {f1:.4f}")
print(f"  Test Samples   : {len(y_test)}")
print(f"{'='*60}\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 1. CONFUSION MATRIX ───────────────────────────────────────────────────────
print("[INFO] Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

# Short labels for readability
short_names = [
    'DAP\n(18:46:0)',
    'Iffco NPK\n17:17:17',
    'Kribhco\n20:20:0',
    'MAP\n28:28:0',
    'NPK Complex\n14:35:14',
    'Tata Paras\n10:26:26',
    'Urea\n(46% N)',
]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlGn',
    xticklabels=short_names,
    yticklabels=short_names,
    linewidths=0.5,
    linecolor='#e0e0e0',
    ax=ax,
    annot_kws={'size': 11, 'weight': 'bold'},
)

ax.set_xlabel('Predicted Fertilizer', fontsize=12, fontweight='bold', labelpad=12)
ax.set_ylabel('Actual Fertilizer', fontsize=12, fontweight='bold', labelpad=12)
ax.set_title(
    f'Confusion Matrix — GreenGrow Fertilizer Model\nTest Accuracy: {acc*100:.2f}%  |  Weighted F1: {f1:.4f}',
    fontsize=13, fontweight='bold', pad=16
)
ax.tick_params(axis='x', labelsize=8.5, rotation=0)
ax.tick_params(axis='y', labelsize=8.5, rotation=0)

plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SUCCESS] Saved → static/confusion_matrix.png")

# ── 2. PER-CLASS METRICS BAR CHART ───────────────────────────────────────────
print("[INFO] Generating per-class metrics chart...")
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred, average=None)
recall    = recall_score(y_test, y_pred, average=None)
f1_class  = f1_score(y_test, y_pred, average=None)

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width, precision*100, width, label='Precision', color='#4da528', alpha=0.88)
bars2 = ax.bar(x,         recall*100,    width, label='Recall',    color='#2d6a0f', alpha=0.88)
bars3 = ax.bar(x + width, f1_class*100,  width, label='F1-Score',  color='#a8d870', alpha=0.88)

# Value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, h + 0.5,
            f'{h:.1f}%', ha='center', va='bottom',
            fontsize=7.5, fontweight='bold'
        )

ax.set_xlabel('Fertilizer Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title(
    'Per-Class Precision, Recall & F1-Score\nGreenGrow Fertilizer Recommendation Model',
    fontsize=13, fontweight='bold'
)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=8.5)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# Overall accuracy banner
ax.axhline(y=acc*100, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
ax.text(len(class_names)-0.5, acc*100+1.5,
        f'Overall Accuracy: {acc*100:.2f}%',
        color='red', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('static/metrics_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SUCCESS] Saved → static/metrics_chart.png")

print(f"\n✅ Done! Both charts saved to the static/ folder.")
print(f"   View via Flask: http://127.0.0.1:5000/static/confusion_matrix.png")
print(f"   View via Flask: http://127.0.0.1:5000/static/metrics_chart.png")
