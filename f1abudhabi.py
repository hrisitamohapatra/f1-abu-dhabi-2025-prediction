import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

fastf1.Cache.enable_cache("f1_cache")

print("   F1 Abu Dhabi GP 2025 - Refined Championship Decider   ")
print("=" * 70)

# ======= 1. TRAINING DATA (2024 Abu Dhabi) =======
print("\nLoading 2024 Abu Dhabi GP data...")
session_2024 = fastf1.get_session(2024, 'Abu Dhabi', 'R')
session_2024.load()

laps_2024 = session_2024.laps.pick_quicklaps().copy()
laps_2024["Driver"] = laps_2024["Driver"].apply(lambda d: str(d).strip())

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "median",
    "Sector2Time (s)": "median",
    "Sector3Time (s)": "median",
    "LapTime (s)": "median"
}).reset_index()

# ======= 2. 2025 CONTEXT =======
driver_to_team = {
    "NOR": "McLaren", "PIA": "McLaren", "VER": "Red Bull", "TSU": "Red Bull",
    "LEC": "Ferrari", "HAM": "Ferrari", "RUS": "Mercedes", "ANT": "Mercedes",
    "ALO": "Aston Martin", "STR": "Aston Martin", "GAS": "Alpine", "COL": "Alpine",
    "OCO": "Haas", "BEA": "Haas", "LAW": "Racing Bulls", "HAD": "Racing Bulls",
    "ALB": "Williams", "SAI": "Williams", "BOR": "Kick Sauber", "HUL":"Kick Sauber"
}

driver_names = {
    "NOR": "Lando Norris", "VER": "Max Verstappen", "PIA": "Oscar Piastri",
    "RUS": "George Russell", "LEC": "Charles Leclerc", "HAM": "Lewis Hamilton",
    "ANT": "Kimi Antonelli", "ALB": "Alexander Albon", "SAI": "Carlos Sainz",
    "HAD": "Isack Hadjar", "HUL": "Nico Hülkenberg", "ALO": "Fernando Alonso",
    "BEA": "Oliver Bearman", "LAW": "Liam Lawson", "TSU": "Yuki Tsunoda",
    "OCO": "Esteban Ocon", "STR": "Lance Stroll", "GAS": "Pierre Gasly",
    "BOR": "Gabriel Bortoleto", "COL": "Franco Colapinto"
}

team_points = {
    "McLaren": 800, "Red Bull": 688, "Ferrari": 614, "Mercedes": 611,
    "Alpine": 54, "Haas": 90, "Racing Bulls": 71,
    "Aston Martin": 80, "Williams": 137, "Kick Sauber": 0
}

# Standings before the race
championship_standings = {
    "NOR": 408, "VER": 396, "PIA": 392, "RUS": 309, "LEC": 230,
    "HAM": 152, "ANT": 150, "ALB": 73, "SAI": 64, "HAD": 51,
    "HUL": 49, "ALO": 48, "BEA": 41, "LAW": 38, "TSU": 33,
    "OCO": 32, "STR": 32, "GAS": 22, "BOR": 19, "COL": 0
}

# 2025 Qualifying Results
qualifying_2025 = pd.DataFrame({
    "Driver": ["NOR", "VER", "PIA", "RUS", "LEC", "ALO", "BOR", "OCO", "HAD",
               "TSU", "BEA", "SAI", "LAW", "ANT", "STR", "HAM", "ALB", "HUL", "GAS", "COL"],
    "QualifyingTime (s)": [
        79.009, 79.139, 79.179, 79.404, 79.550, 79.351, 79.620, 79.740, 79.850,
        80.289, 80.450, 79.308, 80.751, 79.920, 80.100, 79.826, 80.614, 80.550, 80.850, 81.012
    ],
    "GridPosition": range(1, 21)
})

# ======= 3. FEATURE PROCESSING =======
max_points = max(team_points.values())
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamScore"] = qualifying_2025["Team"].map(lambda x: team_points[x] / max_points)

merged_data = qualifying_2025.merge(
    sector_times_2024[["Driver", "LapTime (s)"]], on="Driver", how="left"
)

# Team-based imputation for missing data (rookies/driver moves)
team_medians = merged_data.groupby("Team")[["LapTime (s)"]].transform("median")
merged_data["LapTime (s)"] = merged_data["LapTime (s)"].fillna(team_medians["LapTime (s)"])
merged_data = merged_data.fillna(merged_data.median(numeric_only=True))

# ======= 4. TRAIN & PREDICT =======
feature_cols = ["QualifyingTime (s)", "TeamScore", "GridPosition"]
X = merged_data[feature_cols]
y = merged_data["LapTime (s)"]

model = XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.01, 
    max_depth=4, 
    monotone_constraints=(1, -1, 1)
)
model.fit(X, y)

# Predict and apply Season Offset Calibration (-1.5s for 2025 development)
merged_data["PredictedRaceTime (s)"] = model.predict(X) - 1.5
final_results = merged_data.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)
final_results["PredictedPosition"] = range(1, 21)

# Regression MSE
mse_regression = mean_squared_error(y, model.predict(X))

# ======= 5. RESULTS & CHAMPIONSHIP =======
print("\n" + "=" * 70)
print("🏆 PREDICTED RACE RESULTS - ABU DHABI GP 2025")
print("=" * 70)

# 2025 Points System (No Fastest Lap Point)
POINTS_SYSTEM = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
final_championship = championship_standings.copy()

for idx, row in final_results.iterrows():
    p = int(row['PredictedPosition'])
    driver = row['Driver']
    time = row['PredictedRaceTime (s)']
    print(f"  P{p:2d}: {driver} - {driver_names[driver]:20s} (Grid: P{int(row['GridPosition'])}) - {time:.3f}s")
    if p in POINTS_SYSTEM:
        final_championship[driver] += POINTS_SYSTEM[p]

sorted_championship = sorted(final_championship.items(), key=lambda x: x[1], reverse=True)
champion = sorted_championship[0]

print(f"\n{'*'*70}")
print(f"👑 WORLD CHAMPION: {driver_names[champion[0]]} ({champion[0]}) 👑")
print(f"{'*'*70}\n")


# ======= 6. POSITION MSE (Calculated from Actual Results) =======
actual_pos_data = {
    "Driver": ["VER", "PIA", "NOR", "LEC", "RUS", "ALO", "OCO", "HAM", "BEA", "STR", 
               "BOR", "SAI", "TSU", "ANT", "ALB", "HAD", "LAW", "GAS", "COL", "HUL"],
    "ActualPosition": range(1, 21)
}
actual_df = pd.DataFrame(actual_pos_data)
compare = final_results[["Driver", "PredictedPosition"]].merge(actual_df, on="Driver")
mse_position = mean_squared_error(compare["ActualPosition"], compare["PredictedPosition"])

print(f"📊 MODEL STATISTICS:")
print(f"   Regression MSE (Training): {mse_regression:.6f}")
print(f"   Position MSE (Final Rank): {mse_position:.2f}")