# ============================================================
# TRAIN BASELINE MODEL 
# ============================================================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Load Data
# ============================================================
print("[INFO] Loading dataset...")
df = pd.read_csv("E:/TDW/P01/data/processed/daily_metrics_clean.csv", parse_dates=['date'])

if df.empty:
    raise ValueError("The dataset is empty. Please check 'daily_metrics_clean.csv'.")

# Sort by date
df = df.sort_values('date')
print("[INFO] Dataset loaded successfully with shape:", df.shape)

# ============================================================
# Feature Engineering - Create Lag Features
# ============================================================
print("[INFO] Creating lag and moving average features...")

# Create lag features
df['views_t-1'] = df.groupby('video_id')['views'].shift(1)
df['views_t-7'] = df.groupby('video_id')['views'].shift(7)

# Moving average of last 7 days per video
df['views_ma7'] = (
    df.groupby('video_id')['views']
    .rolling(window=7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Fill missing lag values with 0 (safe option)
df[['views_t-1', 'views_t-7', 'views_ma7']] = df[['views_t-1', 'views_t-7', 'views_ma7']].fillna(0)

# ============================================================
# Prepare Dataset for Training
# ============================================================
X = df[['views_t-1', 'views_t-7', 'views_ma7']]
y = df['views']

# Sanity check
print("[INFO] Features and target prepared.")
print("X shape:", X.shape, "| y shape:", y.shape)

if len(X) == 0:
    raise ValueError("No samples found after preprocessing. Check data integrity.")

# ============================================================
# Split into Train and Test Sets
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("[INFO] Data successfully split!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ============================================================
# Save Processed Data
# ============================================================
df.to_csv("E:/TDW/P01/data/processed/daily_metrics_with_lags.csv", index=False)
print("[INFO] Processed file saved as 'daily_metrics_with_lags.csv'.")``

# ============================================================
# 📈 BASELINE REGRESSION MODEL (CONTINUATION)
# ============================================================

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# ============================================================
# 5️⃣ Train Ridge Regression Model
# ============================================================
print("[INFO] Training Ridge Regression model...")
model = Ridge(alpha=1.0)  # You can also try LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
print("[INFO] Prediction completed.")

# ============================================================
# 6️⃣ Evaluate Model Performance
# ============================================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"[RESULT] RMSE: {rmse:.2f}, MAE: {mae:.2f}")
print("[INFO] Model evaluation complete.")

# ============================================================
# 7️⃣ Save Forecast Results
# ============================================================
results = pd.DataFrame({
    'date': df.iloc[y_test.index]['date'].values,
    'video_id': df.iloc[y_test.index]['video_id'].values,
    'actual': y_test.values,
    'predicted': y_pred
})

results.to_csv("E:/TDW/P01/data/processed/forecast_baseline.csv", index=False)
print("[INFO] Forecast results saved as 'forecast_baseline.csv'.")

# ============================================================
# 8️⃣ Summary
# ============================================================
print("✅ Baseline Ridge Regression model training and evaluation completed successfully.")
