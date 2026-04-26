# Ground Truth Data - Next Steps

## ✅ Completed: Target Variable Processing

**Output**: `data/project_options_processed.csv`
- ✓ 1,126 projects × 5 columns
- ✓ One row per project (deduplicated via max PV filter)
- ✓ Three target variables: `total_pv_kwp`, `total_inverter_kw`, `total_battery_kwh`
- ✓ All metrics normalized to standard units (kWp, kW, kWh)
- ✓ Zero missing values, validated for data quality

**Key Finding**: Dataset is **battery-storage dominant** (99.6% of projects). Only 3 PV systems detected.

---

## 🔄 Next: Feature Matrix Processing

**Input**: `CSV/projects_status_quo.csv`
**Goal**: Extract household features (X variables) for ML model training

### Recommended Script: `src/process_projects_status_quo.py`

Process the status quo CSV to:
1. Load and profile household features
2. Identify feature types (numeric, categorical, temporal)
3. Handle missing values with domain logic
4. Create derived features (e.g., energy_demand_per_sqm)
5. Return clean feature matrix indexed by `project_id`

**Expected Output**: `data/projects_features_processed.csv`
- Index: `project_id` (match with targets)
- Columns: Household characteristics (demand, size, EV presence, etc.)
- Quality: No missing values in critical features

---

## 🎯 Final Training Dataset

Once both X and Y are processed:

```python
# Merge features (X) with targets (Y)
X = pd.read_csv('data/projects_features_processed.csv')
Y = pd.read_csv('data/project_options_processed.csv')

# Join on project_id
data = X.merge(Y, on='project_id', how='inner')

# Train/test split with stratification
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Multi-output regression pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

features = [col for col in X.columns if col != 'project_id']
targets = ['total_pv_kwp', 'total_inverter_kw', 'total_battery_kwh']

X_train, y_train = train[features], train[targets]
X_test, y_test = test[features], test[targets]

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
```

---

## 📊 Model Training Considerations

### Data Imbalance
- **Battery (Y2)**: 1,121 non-zero samples ✓ (robust prediction)
- **PV (Y1)**: 3 non-zero samples ⚠ (data sparsity—consider synthetic augmentation)
- **Inverter (Y3)**: 6 non-zero samples ⚠ (sparse—use as auxiliary task)

### Recommended Approach
1. **Primary task**: Predict `total_battery_kwh` (well-distributed)
2. **Auxiliary tasks**: PV and inverter (use regularization L1/L2 to handle sparsity)
3. **Evaluation**: 
   - Battery: R², MAE, RMSE (main metric: R²)
   - PV/Inverter: MAE (less sensitive to extreme values with few samples)

### Handling Sparse Targets
- Use **weighted loss functions** to penalize large errors on rare PV/inverter cases
- Consider **multi-task learning** to share representation between battery/PV/inverter predictions
- **Regularization**: L1/L2 prevents overfitting on sparse targets
- **Cross-validation**: k-fold (k=5) to maximize sample efficiency

---

## 📁 File Organization

```
HackBerlin_Solar/
├── CSV/
│   ├── project_options_parts.csv          (raw components)
│   └── projects_status_quo.csv            (raw features)
├── src/
│   ├── process_project_options.py         ✓ DONE (Y variables)
│   └── process_projects_status_quo.py     ⏳ TODO (X variables)
├── data/
│   ├── project_options_processed.csv      ✓ OUTPUT (targets)
│   └── projects_features_processed.csv    ⏳ TODO (features)
├── models/
│   └── renewable_energy_model.pkl         ⏳ TODO
└── README.md
    DATA_QUALITY_REPORT.md                 ✓ Documentation
    NEXT_STEPS.md                          ← You are here
```

---

## 🚀 Quick Start: Feature Processing

To process the status quo CSV:

```bash
# Using @Renewable Energy ML Engineer agent:
# "Process projects_status_quo.csv to extract household features for our ML model"
```

Or manually run (once created):
```bash
python src/process_projects_status_quo.py
```

---

## 🔗 References

- **Data Quality Report**: [DATA_QUALITY_REPORT.md](DATA_QUALITY_REPORT.md)
- **Processing Script**: [src/process_project_options.py](src/process_project_options.py)
- **Agent Documentation**: [.github/agents/renewable-energy-ml-engineer.agent.md](.github/agents/renewable-energy-ml-engineer.agent.md)

---

*Last Updated: 2026-04-25*
