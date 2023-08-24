# BUSINESS SCIENCE UNIVERSITY
# LENDING CLUB LOAN DEFAULT SCORING  
# PART 1: ANALYSIS
# ----

# GOAL: Predict Loan Credit Score ("Good" or "Bad")

# ABOUT THE DATA

# These data were downloaded from the Lending Club access site and are from the first quarter of 2016. The outcome is in the variable Class and is either "good(meaning that the loan was fully paid back or currently on-time) or "bad" (charged off, defaulted, of 21-120 days late). 

# LIBRARIES

import pandas as pd
import pycaret.classification as clf
import plotly.express as px

# import matplotlib
# matplotlib.use('Agg')  # Use a non-GUI backend
# import matplotlib.pyplot as plt

# DATA

# lending_data_raw_df = pd.read_csv("BONUS_SHINY_APP_1/data/lending_club.csv")
lending_data_raw_df = pd.read_csv("data/lending_club.csv")

lending_data_raw_df


# 1.0 QUICK MACHINE LEARNING WITH PYCARET ----

# * Subset the data ----
df = lending_data_raw_df

df.info()

# Numeric Columns
float_cols = df.select_dtypes(include="float64").columns.to_list()

int_cols = df.select_dtypes(include="int64").columns.to_list()

numeric_columns = [*float_cols, *int_cols]

# Categorical Columns
cat_columns = df.select_dtypes('object').drop("Class", axis=1).columns.to_list()

# * Setup the Classifier ----
clf_1 = clf.setup(
    data       = df, 
    target     = "Class",
    train_size = 0.8,
    session_id = 123,
    
    categorical_features=cat_columns,
    numeric_features=numeric_columns
    
)

# * Make A Machine Learning Model ----
xgb_model = clf.create_model(
    estimator = 'xgboost'
)

lgb_model = clf.create_model(
    estimator = 'lightgbm'
)

# CHECK MODEL ----

# clf.plot_model(xgb_model, plot="auc")
import matplotlib
matplotlib.use("TkAgg")

try:
    clf.plot_model(xgb_model, plot="auc")
except Exception as e:
    print("Error loading model:", e)


# FEATURE IMPORTANCE -----

# Basic Feature Importance
clf.plot_model(xgb_model, plot = 'feature')

# Shap Feature Importance
clf.interpret_model(xgb_model, plot='summary')



# FINALIZE MODEL ----
xgb_model_finalized = clf.finalize_model(xgb_model)

lgb_model_finalized = clf.finalize_model(lgb_model)

# MAKE PREDICTIONS & RANKING LOAN DEFAULT RISK ----

predictions_df = clf.predict_model(
    xgb_model_finalized, 
    data      = lending_data_raw_df,
    raw_score = True
)

# predictions_df \
#     .sort_values("Score_bad", ascending=False)
try:
    predictions_df \
        .sort_values("prediction_score_bad", ascending=False)
except Exception as e:
    print("Error loading model:", e)

# SAVE MODEL ----

clf.save_model(xgb_model_finalized, "models/xgb_model_finalized")

clf.save_model(lgb_model_finalized, "models/lgb_model_finalized")

# VISUALS FOR APP ----

# User Inputs:

fraction = 0.10

model = xgb_model_finalized
model = lgb_model_finalized

# Predictions:

df_sample = lending_data_raw_df.sample(
    frac=fraction,
    random_state=123
)

predictions_df_sample = clf.predict_model(
    estimator = model, 
    data      = df_sample,
    raw_score = True
)

# * Loan Interest Rate
px.scatter(
    data_frame = predictions_df_sample,
    x = 'int_rate',
    y = 'prediction_score_bad',
    color = 'prediction_score_bad',
    trendline='lowess',
    trendline_color_override="white",
    template='plotly_dark',
    log_x=True,
    log_y=True,
    title = "Interest Rate vs Loan Default Score"
)

# * Total Balance
px.scatter(
    data_frame = predictions_df_sample,
    x = 'total_bal_il',
    y = 'prediction_score_bad',
    color = 'prediction_score_bad',
    trendline='lowess',
    trendline_color_override="white",
    template='plotly_dark',
    log_x=True,
    log_y=True,
    title="Total Balance Vs Bad Loan Score"
)

