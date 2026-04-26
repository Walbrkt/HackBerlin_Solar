# Reonic Renewable Energy Dataset - Processing Report

## Overview

**Processed Output**: `data/project_options_processed.csv`
- **Total Projects**: 1,126
- **Target Variables**: 3 (total_pv_kwp, total_inverter_kw, total_battery_kwh)
- **Data Format**: One row per project_id (maximum viable system design)

---

## Dataset Characteristics

### System Composition
| System Type | Count | % |
|---|---|---|
| Battery-only (SES) | 1,121 | 99.6% |
| PV-only (Solar) | 3 | 0.3% |
| Hybrid (PV + Battery) | 0 | 0.0% |
| No system | 2 | 0.2% |

**Insight**: Dataset is dominated by energy storage systems (SES), reflecting European residential focus on decentralized energy storage rather than PV generation.

### Target Variable Distributions

#### Solar PV Capacity (total_pv_kwp)
- **Non-zero projects**: 3 / 1,126 (0.3%)
- **Range**: 0.455 - 2.375 kWp
- **Mean (non-zero)**: 1.102 kWp
- **Challenge**: Extremely sparse—insufficient data for robust solar capacity prediction

#### Battery Storage (total_battery_kwh)
- **Non-zero projects**: 1,121 / 1,126 (99.6%)
- **Range**: 3.5 - 26.98 kWh
- **Mean**: 8.73 kWh
- **Median**: 10.0 kWh
- **Std Dev**: 3.70 kWh
- **Quality**: ✓ Sufficient for regression modeling

#### Inverter Power (total_inverter_kw)
- **Non-zero projects**: 6 / 1,126 (0.5%)
- **Range**: 5.0 - 15.0 kW
- **Mean (non-zero)**: 8.33 kW
- **Challenge**: Sparse data—only 6 projects with hybrid inverter systems

---

## Data Processing Pipeline

### 1. **Input Data**
- Source: `CSV/project_options_parts.csv` (19,257 rows)
- Contains: Line-item components per project-option combination

### 2. **Filtering**
- **Removed**: 17,759 rows (92.2%)
  - InstallationFee, ServiceFee, ModuleFrameConstruction, Accessories
  - Labor, logistics, and service components
- **Retained**: 1,498 hardware rows (7.8%)
  - Module (4), BatteryStorage (1,488), Inverter (6)

### 3. **Metric Extraction**
Data mapping (note: column names appear swapped in source CSV):
```
Module rows           → module_watt_peak (PV capacity in W)
Inverter rows         → battery_capacity_kwh (inverter capacity in Wh)
BatteryStorage rows   → inverter_power_kw (battery capacity in Wh)
```

### 4. **Aggregation**
- **Grouped by**: project_id + option_id
- **Aggregated**: Sum of all components per option
- **Result**: 1,444 option-level records

### 5. **Maximum PV Filter**
- **Rationale**: Each project may have multiple design proposals (Good/Better/Best)
- **Action**: Retained only the option with max PV capacity per project
- **Result**: 1,126 projects × 1 design (maximum viable system)
- **Discarded**: 318 alternative options

### 6. **Unit Conversion**
- PV capacity: W → kWp (÷ 1,000)
- Inverter power: Wh → kW (÷ 1,000)
- Battery storage: Wh → kWh (÷ 1,000)

---

## Data Quality Notes

### ✓ Production Ready
- **One row per project**: Deduplicated via maximum PV filter
- **No missing values**: All projects have numeric values (zero allowed)
- **Positive values**: No negative capacities
- **Consistent units**: All metrics standardized (kWp, kW, kWh)

### ⚠ Data Imbalance
- **Battery target**: 99.6% of projects have non-zero values (suitable for regression)
- **PV target**: 0.3% of projects (sparse—may limit PV prediction accuracy)
- **Inverter target**: 0.5% of projects (sparse—battery-storage dominant design pattern)

### ⚠ Missing Hybrid Systems
- No projects with both PV and battery in this dataset
- Real-world hybrid systems would mix the two; dataset may represent different customer segments

---

## Recommended Next Steps

### For Model Training
1. **Primary target**: Focus on `total_battery_kwh` (well-distributed, 1,121 samples)
2. **Multi-task learning**: Treat PV and inverter as auxiliary tasks despite sparsity
3. **Data augmentation**: Consider synthetic hybrid systems by combining PV + battery projects
4. **Regularization**: Use L1/L2 to handle sparse targets (PV, inverter)

### For Feature Engineering
- Load `projects_status_quo.csv` to extract household features (X variables)
- Merge on `project_id` with this processed data (Y variables)
- Handle missing values in features with domain-aware imputation

### For Model Validation
- **Train/test split**: Stratify on `system_type` (PV vs Battery) to balance both targets
- **Cross-validation**: Use k-fold to maximize use of small PV sample
- **Metrics**: 
  - Battery: R², MAE, RMSE (abundant data)
  - PV/Inverter: Focus on MAE (sparse samples, outliers dangerous)

---

## File Structure
```
data/
├── project_options_processed.csv      # ✓ Processed Y variables (1,126 × 5 cols)
```

## Script Reference
- **Source**: `src/process_project_options.py`
- **Usage**: `python src/process_project_options.py` (auto-saves to `data/`)
- **Functions**:
  - `load_project_options()`: Load CSV
  - `filter_hardware_components()`: Remove labor/service
  - `extract_metrics()`: Unpack technical specs
  - `aggregate_by_option()`: Sum per option
  - `filter_maximum_pv_option()`: Keep max design
  - `process_project_options()`: Full pipeline

---

## Data Dictionary: `project_options_processed.csv`

| Column | Type | Unit | Description |
|---|---|---|---|
| `project_id` | string | — | Unique customer project identifier |
| `option_id` | string | — | Unique design option identifier (max PV selected) |
| `total_pv_kwp` | float | kWp | Solar PV capacity (sum of module wattages) |
| `total_inverter_kw` | float | kW | Inverter power rating (DC-AC conversion capacity) |
| `total_battery_kwh` | float | kWh | Battery storage capacity (energy storage) |

---

*Generated: 2026-04-25*
*Pipeline: project_options_parts.csv → Ground Truth System Designs*
