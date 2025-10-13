import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load dataset
# -----------------------------
file_path = "E:/TDW/P01/data/raw/youtube_daily_metrics.csv"  # adjust if needed
df = pd.read_csv(file_path, parse_dates=['date'])

# Sort by video_id and date
df = df.sort_values(['video_id', 'date'])

# -----------------------------
# 2. Feature Engineering
# -----------------------------
# Create lag features (using shorter windows since dataset is small)
df['views_t-1'] = df.groupby('video_id')['views'].shift(1)
df['views_t-3'] = df.groupby('video_id')['views'].shift(3)

# 3-day moving average
df['views_ma3'] = (
    df.groupby('video_id')['views']
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Drop missing values caused by shifting
df = df.dropna(subset=['views_t-1', 'views_t-3', 'views_ma3'])

print(f"[INFO] Data shape after lag features: {df.shape}")

# -----------------------------
# 3. Train/Test Split
# -----------------------------
features = ['views_t-1', 'views_t-3', 'views_ma3', 'likes', 'comments', 'watch_time', 'subs']
X = df[features]
y = df['views']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -----------------------------
# 4. Model Training
# -----------------------------
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# 5. Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n[MODEL PERFORMANCE]")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# -----------------------------
# 6. Save Outputs
# -----------------------------
output_dir = "E:/TDW/P01/outputs"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "view_forecasts.csv")
plot_file = os.path.join(output_dir, "view_forecast_plot.png")

output = pd.DataFrame({
    'actual_views': y_test.reset_index(drop=True),
    'predicted_views': y_pred
})
output.to_csv(output_file, index=False)
print(f"[SUCCESS] Forecasts saved to: {output_file}")

# -----------------------------
# 7. Visualization
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label="Actual Views", marker='o')
plt.plot(y_pred[:100], label="Predicted Views", marker='x')
plt.title("Actual vs Predicted Views (Sample of 100)")
plt.xlabel("Sample Index")
plt.ylabel("Views")
plt.legend()
plt.tight_layout()
plt.savefig(plot_file)
print(f"[SUCCESS] Forecast plot saved to: {plot_file}")
plt.show()
