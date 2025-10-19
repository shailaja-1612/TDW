
---

# YouTube Video Performance Prediction Project

This repository contains Python scripts, processed data, and outputs for forecasting YouTube video views and classifying top-performing videos.


##  Project Structure



##  Lessons Overview

### **Lesson 1 – Data Preparation**
- Script: `prepare_features.py`  
- Tasks: Create lag features and 7-day moving average  
- Output: `daily_metrics_clean.csv`

### **Lesson 2 – Baseline Regression**
- Script: `train_baseline.py`  
- Features: `views_t-1`, `views_t-7`, `views_ma7`  
- Model: Ridge Regression  
- Output: `forecast_baseline.csv`  
- Evaluation: RMSE and MAE printed in console  

### **Lesson 3 – Top-Quartile Classification**
- Script: `classify_top_videos.py`  
- Features: `views_t-1`, `views_t-7`, `views_ma7`, `engagement_rate`, `video_age_days`  
- Model: Logistic Regression  
- Output: `predictions.csv`  
- Evaluation: Accuracy and classification report printed in console  

---





