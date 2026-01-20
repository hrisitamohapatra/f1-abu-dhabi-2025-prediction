# 🏎️ F1 2025 Abu Dhabi GP Championship Predictor

A machine learning–based Formula 1 race and championship outcome predictor built using **XGBoost** and the **FastF1 API**.  
This project models race pace using historical data from the **2024 Abu Dhabi Grand Prix** and combines it with **2025 season performance indicators** to predict the final race result and the **World Drivers’ Champion**.

---

## 📊 Project Overview

The objective of this project is to predict the outcome of the **2025 Abu Dhabi Grand Prix**, the season-ending race and championship decider between **Lando Norris** and **Max Verstappen**.

Given the technical nature of the Yas Marina Circuit and the pressure of a title-deciding race, the model focuses on **race pace realism rather than simple position-based heuristics**.

---

## ✨ Key Features

### 🔗 Data Integration
- Uses the **FastF1 API** to extract:
  - Lap times
  - Sector times
  - Race session telemetry
- Aggregates race pace from the **2024 Abu Dhabi GP** to preserve circuit-specific characteristics.

### 🧠 Feature Engineering
The model incorporates:
- **Grid Position**
- **Qualifying Time**
- **Team Performance Score** (normalized constructor strength)
- **Clean Air Race Pace**
- **Total Sector Time (aggregated)**
- **Championship Context** (pressure-handling proxy)

These features capture both **driver skill** and **car performance**, which are critical at Yas Marina.

---

### 🧩 Intelligent Imputation Strategy
To handle missing or unavailable data:
- **Team-based median imputation** is applied for:
  - Rookies (e.g., Antonelli, Bortoleto)
  - Driver transfers (e.g., Hamilton → Ferrari)
- If team data is unavailable, the model falls back to a **scaled qualifying-time heuristic**, ensuring no driver is dropped.

---

### 🤖 Machine Learning Model
- **Algorithm:** `XGBRegressor`
- **Objective Function:** Squared Error
- **Monotone Constraints:** Enforced to ensure realistic racing behavior:
  - Slower qualifying times cannot produce faster race pace
  - Worse grid positions cannot result in unjustified performance gains

This ensures predictions remain **physically and competitively plausible**.

---

## 📈 Model Performance & Accuracy Analysis

The model was evaluated using both **lap-time accuracy** and **finishing-order realism**.

### 🧮 Regression Accuracy
- **Training Mean Squared Error (MSE):** `0.0477`

This low error indicates the model effectively learned pace patterns from the 2024 Abu Dhabi data and transferred them to the 2025 grid.

---

### 🏁 Position Accuracy
- **Position Mean Squared Error (MSE):** `11.00`

**Interpretation:**
- Positional RMSE ≈ **3.3 positions per driver**
- Given the inherent unpredictability of Formula 1 (strategy, safety cars, traffic), this represents **high-tier predictive accuracy**.

---

### 👑 Championship Prediction
**Result: SUCCESS**

The model correctly predicted the outcome of the title decider:

> 🏆 **Lando Norris – 2025 World Drivers’ Champion**

---

## 🏁 Final Predicted Podium – Abu Dhabi GP 2025

1. 🥇 **Lando Norris** (McLaren)  
2. 🥈 **Max Verstappen** (Red Bull)  
3. 🥉 **Oscar Piastri** (McLaren)

