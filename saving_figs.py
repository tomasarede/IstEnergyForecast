from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import timedelta
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from xgboost import XGBRegressor
import json
import plotly.io as pio

CSV_FILE = "proj1/Power_Meteo_Merged.csv"
df = pd.read_csv(CSV_FILE)
df = df.drop(columns=["Year", "Month", "Day", "Hour"])
df["Date_start"] = pd.to_datetime(df["Date_start"])
df.sort_values(by="Date_start", inplace=True)

# List available features (all columns except "Date_start")
available_features = [col for col in df.columns if col != "Date_start"]


#-------------
# Correlation
#--------------


corr = df.drop(columns = ["Date_start"]).corr()
fig_corr = px.imshow(corr, text_auto='.2f',
                     aspect="auto",
                     color_continuous_scale="RdBu_r",
                     zmin = -1,
                     zmax= 1)

fig_corr.update_layout(title="Correlation Heatmap", template="seaborn")

fig_json = fig_corr.to_json()
with open("dash_apps/corr_figure.json", "w") as f:
    f.write(fig_json)


#------------
# F-Test
#------------

features = df.drop(columns=["Date_start","Power_kW"]).columns.tolist()
df_copy = df.dropna()
feature_selector = SelectKBest(k=3, score_func=f_regression)
fit = feature_selector.fit(df_copy[features].dropna(), df_copy["Power_kW"])
scores = fit.scores_

# Create a Plotly bar chart of the scores
fig_kbest = go.Figure()
fig_kbest.add_trace(
    go.Bar(
        x=features,        # feature names
        y=scores,          # their corresponding scores
        marker_color="royalblue",
    )
)
fig_kbest.update_layout(
    title="F-Test Feature Scores",
    xaxis_title="Features",
    yaxis_title="Score",
    template="seaborn",
    xaxis_tickangle=-90,

)


fig_json = fig_kbest.to_json()
with open("dash_apps/kbest_figure.json", "w") as f:
    f.write(fig_json)


# ----------------------------
# Dash App for Mutual Information (SelectKBest)
# ----------------------------
feature_mutual = SelectKBest(k=3, score_func=mutual_info_regression)
fit_mutual = feature_mutual.fit(df_copy[features], df_copy["Power_kW"])
scores_mutual = fit_mutual.scores_

fig_mutual = go.Figure()
fig_mutual.add_trace(
    go.Bar(
        x=features,
        y=scores_mutual,
        marker_color="mediumseagreen"
    )
)
fig_mutual.update_layout(
    title="Mutual Information Feature Scores",
    xaxis_title="Features",
    yaxis_title="Score",
    template="seaborn",
    xaxis_tickangle=-90

)


fig_json = fig_mutual.to_json()
with open("dash_apps/mutual_figure.json", "w") as f:
    f.write(fig_json)



# ----------------------------
# Dash App for Wrapper Method: RFE
# ----------------------------
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=3)
selector.fit(df_copy[features], df_copy["Power_kW"])
rankings = selector.ranking_

fig_rfe = go.Figure()
fig_rfe.add_trace(
    go.Bar(
        x=features,
        y=rankings,
        marker_color="indianred"
    )
)
fig_rfe.update_layout(
    title="RFE Rankings",
    xaxis_title="Features",
    yaxis_title="Ranking",
    template="seaborn",
    xaxis_tickangle=-90

)


fig_json = fig_rfe.to_json()
with open("dash_apps/rfe_figure.json", "w") as f:
    f.write(fig_json)



#------------------
# NEW FEATURES
#------------------
CSV_FILE = "proj1/Added_Features.csv"
df_new_features = pd.read_csv(CSV_FILE)
df_new_features["Date_start"] = pd.to_datetime(df_new_features["Date_start"])
df_new_features.sort_values(by="Date_start", inplace=True)
available_features = [col for col in df_new_features.columns if col != "Date_start" and col!= "Power_kW"]
df_new_features = df_new_features.dropna()
# ----------------------------
# 1. F-Test using f_regression on Cleaned Data
# ----------------------------
features_clean = available_features  # use all candidate features
f_test_selector_clean = SelectKBest(k=3, score_func=f_regression)
fit_clean = f_test_selector_clean.fit(df_new_features[features_clean], df_new_features["Power_kW"])
f_scores = fit_clean.scores_
f_features = np.array(features_clean)
sorted_indices = np.argsort(f_scores)[::-1]  # descending order
sorted_features = f_features[sorted_indices]
sorted_scores = f_scores[sorted_indices]

fig_clean_f_test = go.Figure()
fig_clean_f_test.add_trace(go.Bar(
    x=sorted_features,
    y=sorted_scores,
    marker_color="royalblue"
))
fig_clean_f_test.update_layout(
    title="F-Test Feature Importance",
    xaxis_title="Features",
    yaxis_title="F-Score",
    template="seaborn",
    yaxis=dict(range=[0, sorted_scores[0]*1.1]),
    xaxis_tickangle=-90

)

fig_json = fig_clean_f_test.to_json()
with open("dash_apps/clean_f_test_figure.json", "w") as f:
    f.write(fig_json)



# ----------------------------
# 2. Mutual Information on Cleaned Data
# ----------------------------
mi_selector_clean = SelectKBest(k=3, score_func=mutual_info_regression)
mi_fit_clean = mi_selector_clean.fit(df_new_features[features_clean], df_new_features["Power_kW"])
mi_scores = mi_fit_clean.scores_
mi_sorted_indices = np.argsort(mi_scores)[::-1]
mi_sorted_features = np.array(features_clean)[mi_sorted_indices]
mi_sorted_scores = mi_scores[mi_sorted_indices]

fig_clean_mi = go.Figure()
fig_clean_mi.add_trace(go.Bar(
    x=mi_sorted_features,
    y=mi_sorted_scores,
    marker_color="mediumseagreen"
))
fig_clean_mi.update_layout(
    title="Mutual Information Feature Importance",
    xaxis_title="Features",
    yaxis_title="Mutual Information Score",
    template="seaborn",
    xaxis_tickangle=-90
)

fig_json = fig_clean_mi.to_json()
with open("dash_apps/clean_mi_figure.json", "w") as f:
    f.write(fig_json)


# ----------------------------
# 3. RFE (Recursive Feature Elimination) on Cleaned Data
# ----------------------------
estimator_clean = LinearRegression()
rfe_selector_clean = RFE(estimator_clean, n_features_to_select=3)
rfe_selector_clean.fit(df_new_features[features_clean], df_new_features["Power_kW"])
rfe_ranking = rfe_selector_clean.ranking_
rfe_sorted_indices = np.argsort(rfe_ranking)  # lower ranking = more important
rfe_sorted_features = np.array(features_clean)[rfe_sorted_indices]
rfe_sorted_scores = rfe_ranking[rfe_sorted_indices]

fig_clean_rfe = go.Figure()
fig_clean_rfe.add_trace(go.Bar(
    x=rfe_sorted_features,
    y=rfe_sorted_scores,
    marker_color="indianred"
))
fig_clean_rfe.update_layout(
    title="RFE Feature Ranking",
    xaxis_title="Features",
    yaxis_title="RFE Rank (Lower is better)",
    template="seaborn",
    xaxis_tickangle=-90
)

fig_json = fig_clean_rfe.to_json()
with open("dash_apps/clean_rfe_figure.json", "w") as f:
    f.write(fig_json)


# ----------------------------
# 4. Random Forest Feature Importance on Cleaned Data
# ----------------------------
rf_model_clean = RandomForestRegressor()
rf_model_clean.fit(df_new_features[features_clean], df_new_features["Power_kW"])
rf_importances = rf_model_clean.feature_importances_
rf_sorted_indices = np.argsort(rf_importances)[::-1]
rf_sorted_features = np.array(features_clean)[rf_sorted_indices]
rf_sorted_scores = rf_importances[rf_sorted_indices]

fig_clean_rf = go.Figure()
fig_clean_rf.add_trace(go.Bar(
    x=rf_sorted_features,
    y=rf_sorted_scores,
    marker_color="darkorange"
))
fig_clean_rf.update_layout(
    title="Random Forest Feature Importance",
    xaxis_title="Features",
    yaxis_title="Importance",
    template="seaborn",
    yaxis=dict(range=[0, rf_sorted_scores[0]*1.1]),
    xaxis_tickangle=-90
)

fig_json = fig_clean_rf.to_json()
with open("dash_apps/clean_rf_figure.json", "w") as f:
    f.write(fig_json)


# -----------------------------
# Correlation for Cleaned Data
# -----------------------------
# Drop the Date_start column and compute the correlation matrix
corr_clean = df_new_features.drop(columns=["Date_start"]).corr()

# Create a heatmap figure with square cells and two-decimal labels
fig_corr_clean = px.imshow(
    corr_clean,
    text_auto='.2f',
    aspect="equal",                  # square cells
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1
)
fig_corr_clean.update_layout(
    title="Correlation Heatmap",
    template="seaborn"
)

fig_json = fig_corr_clean.to_json()
with open("dash_apps/clean_corr_figure.json", "w") as f:
    f.write(fig_json)



#------------------------
# TEST DATA - VALIDATION
#------------------------


# Load test data
CSV_FILE = "proj1/2019data.csv"
df_test = pd.read_csv(CSV_FILE)
df_test["Date"] = pd.to_datetime(df_test["Date"])
df_test.sort_values(by="Date", inplace=True)
# Exclude Date_start and target (Power_kW) from features
available_features = [col for col in df_test.columns if col not in ["Date", "Power_kW"]]


CSV_FILE = "proj1/Power_Cleaned.csv"
df_final = pd.read_csv(CSV_FILE)
df_final["Date_start"] = pd.to_datetime(df_final["Date_start"])
df_final.sort_values(by="Date_start", inplace=True)


# ------------------------
# XGBoost MODEL PREDICTIONS
# ------------------------
XGB_model = joblib.load('proj1\XGB_model.sav')
y_pred2019_XGB = XGB_model.predict(df_test[available_features])

# ------------------------
# NEURAL NETWORK MODEL PREDICTIONS
# ------------------------
# For the neural network, we'll assume the same feature set is used.
features = available_features

# Define the model architecture
class EnergyPredictor(nn.Module):
    def __init__(self):
        super(EnergyPredictor, self).__init__()
        self.fc1 = nn.Linear(len(features), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

loaded_model = EnergyPredictor()
loaded_model.load_state_dict(torch.load(r"proj1\best_model.pth"))

# Prepare test data for NN prediction
scaler_x = MinMaxScaler()
df_final[features] = scaler_x.fit_transform(df_final[features])

scaler_y = StandardScaler()
df_final["Power_kW"] = scaler_y.fit_transform(df_final[["Power_kW"]])


df_test_cleaned = df_test.copy()

df_test_cleaned["holiday"] = df_test_cleaned["holiday"].astype(int)

# Scale features using the same scaler used during training
df_test_cleaned[available_features] = scaler_x.transform(df_test_cleaned[available_features])
X_test_tensor = torch.tensor(df_test_cleaned[available_features].values, dtype=torch.float32)

with torch.no_grad():
    y_pred_NN = loaded_model(X_test_tensor)
y_pred_NN = y_pred_NN.cpu().numpy()
# Convert predictions back to the original scale
y_pred_NN_orig = scaler_y.inverse_transform(y_pred_NN)


# ------------------------
# XGBoost Figure
# ------------------------

fig_validation_xgb = go.Figure()
fig_validation_xgb.add_trace(go.Scatter(
    x=df_test["Date"],
    y=df_test["Power_kW"],
    mode='lines',
    name='True Values',
    line=dict(color="black")
))
fig_validation_xgb.add_trace(go.Scatter(
    x=df_test["Date"],
    y=y_pred2019_XGB,
    mode='lines',
    name='Predicted (XGBoost)',
    line=dict(color="red")
))
fig_validation_xgb.update_layout(
    title="XGBoost: True vs Predicted Power Consumption",
    xaxis_title="Date",
    yaxis_title="Power (kW)",
    template="seaborn",
    legend=dict(
        orientation="h",    # horizontal layout
        yanchor="bottom",   # anchor the legend’s y position
        y=1.02,             # position above the plotting area
        xanchor="center",   # center horizontally
        x=0.5
    )
)

# ------------------------
# NN Figure
# ------------------------
fig_validation_nn = go.Figure()
fig_validation_nn.add_trace(go.Scatter(
    x=df_test["Date"],
    y=df_test["Power_kW"],
    mode='lines',
    name='True Values',
    line=dict(color="black")
))
fig_validation_nn.add_trace(go.Scatter(
    x=df_test["Date"],
    y=y_pred_NN_orig.flatten(),
    mode='lines',
    name='Predicted (Neural Network)',
    line=dict(color="red")
))



fig_validation_nn.update_layout(
    title="Neural Network: True vs Predicted Power Consumption",
    xaxis_title="Date",
    yaxis_title="Power (kW)",
    template="seaborn",
    legend=dict(
        orientation="h",    # horizontal layout
        yanchor="bottom",   # anchor the legend’s y position
        y=1.02,             # position above the plotting area
        xanchor="center",   # center horizontally
        x=0.5
    )
)


fig_json = fig_validation_xgb.to_json()
with open("dash_apps/val_xgb_figure.json", "w") as f:
    f.write(fig_json)

fig_json = fig_validation_nn.to_json()
with open("dash_apps/val_nn_figure.json", "w") as f:
    f.write(fig_json)