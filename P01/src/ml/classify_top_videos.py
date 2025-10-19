# =========================================
# Lesson 3 — Top Quartile Classification
# =========================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 2️⃣ Load Processed Data
df = pd.read_csv("E:/TDW/P01/data/processed/daily_metrics_clean.csv")
df.columns = df.columns.str.strip()  # Remove extra spaces

# 3️⃣ Create Missing Features if They Don't Exist
if 'views_t-1' not in df.columns:
    df['views_t-1'] = df.groupby('video_id')['views'].shift(1)
if 'views_t-7' not in df.columns:
    df['views_t-7'] = df.groupby('video_id')['views'].shift(7)
if 'views_ma7' not in df.columns:
    # Reduce rolling window to 3 if dataset is small
    df['views_ma7'] = df.groupby('video_id')['views'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
if 'engagement_rate' not in df.columns:
    df['engagement_rate'] = df['likes'] / df['views']
if 'video_age_days' not in df.columns:
    df['video_age_days'] = (pd.to_datetime(df['date']) - df.groupby('video_id')['date'].transform(lambda x: pd.to_datetime(x).min())).dt.days

# 4️⃣ Fill missing values instead of dropping all rows
df['views_t-1'].fillna(0, inplace=True)
df['views_t-7'].fillna(0, inplace=True)
df['views_ma7'].fillna(df['views'], inplace=True)
df['engagement_rate'].fillna(0, inplace=True)
df['video_age_days'].fillna(0, inplace=True)

# 5️⃣ Create Target Column for Top Quartile
threshold = df['views'].quantile(0.75)
df['target_top'] = (df['views'] >= threshold).astype(int)

# 6️⃣ Select Features and Target
features = ['views_t-1', 'views_t-7', 'views_ma7', 'engagement_rate', 'video_age_days']
X = df[features]
y = df['target_top']

# Check number of samples
if X.shape[0] == 0:
    raise ValueError("[ERROR] No data available for training. Check dataset or feature creation steps.")

# 7️⃣ Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8️⃣ Train Logistic Regression Classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# 9️⃣ Make Predictions
preds = model.predict(X_scaled)

# 🔟 Evaluate Model
acc = accuracy_score(y, preds)
print(f"[RESULT] Accuracy: {acc:.2f}")
print(classification_report(y, preds))

# 1️⃣1️⃣ Save Predictions
df_out = df[['video_id', 'date']].copy()
df_out['predicted_top'] = preds
df_out.to_csv("E:/TDW/P01/data/processed/predictions.csv", index=False)
print("[INFO] Predictions saved to 'predictions.csv'")
