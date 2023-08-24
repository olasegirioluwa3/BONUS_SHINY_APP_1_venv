# BUSINESS SCIENCE UNIVERSITY
# LENDING CLUB LOAN DEFAULT SCORING  
# PART 2: SHINY FOR PYTHON APP
# ----

# GOAL: Predict Loan Credit Score ("Good" or "Bad")

# INSTRUCTIONS:
# shiny run --reload BONUS_SHINY_APP_1/02_shiny_app.py --port 8001
# Ctrl + C to shut down


# IMPORTS 
from shiny import (
    App, ui,  reactive, Session
)
from shinywidgets import (
    output_widget, register_widget
)
import shinyswatch


import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go

import pycaret.classification as clf

from pathlib import Path

# DIRECTORY PATHS ----

# data_dir = Path(__file__).parent / "data"
# models_dir = Path(__file__).parent / "models"
# www_dir = Path(__file__).parent / "www"
data_dir = "data"
models_dir = "models"
www_dir = "www"

# DEFAULT APP INPUTS ----

TITLE = "Loan Default Scoring App"

# DATA & MODEL PREP ----

lending_data_raw_df = pd.read_csv(data_dir+"/lending_club.csv")

df_sample = lending_data_raw_df.sample(
    frac=0.1,
    random_state=123
) 

# xgb_model = clf.load_model(models_dir+"/xgb_model_finalized")
# lgb_model = clf.load_model(models_dir / "lgb_model_finalized")
try:
    xgb_model = clf.load_model(models_dir+"/xgb_model_finalized")
    lgb_model = clf.load_model(models_dir+"/lgb_model_finalized")
except Exception as e:
    print("Error loading model:", e)




df_predictions_sample = clf.predict_model(
    xgb_model, 
    df_sample,
    raw_score=True
)


# LAYOUT ----
page_dependencies = ui.tags.head(
    ui.tags.link(
        rel="stylesheet", type="text/css", href="style.css"
    )
)

# Navbar Page ----
app_ui = ui.page_navbar(
    
    # Bootswatch Themes: https://bootswatch.com/
    shinyswatch.theme.lux(),
    
    ui.nav(
        "Loan Insights",
        ui.layout_sidebar(
            sidebar=ui.panel_sidebar(
                
                ui.h2("Loan Default Controls"),
                
                ui.input_slider(
                    "filter_loans", 
                    "Loan Default Score:", 
                    0, 1, 0.0
                ),             
                
                ui.input_slider(
                    "fraction", 
                    "Proportion of Data Used:", 
                    0, 1, 0.10
                ),
                
                ui.input_selectize(
                    "model_selected", 
                    "Model Selected:",
                    ['XGBoost', 'LightGBM'],
                    selected='XGBoost',
                    multiple=False
                ),
                
                ui.input_action_button(
                    "submit", "Submit", 
                    class_="btn-info"
                ),
                
                width=3,
                # class_ = "well-gray",
            ),
            main = ui.panel_main(
                
                ui.column(
                    12,
                    ui.div(
                        output_widget("interest_rate_chart"),
                        class_="card",
                        style="margin:10px;"
                    )                    
                ),
                ui.column(
                    12,
                    ui.div(
                        output_widget("total_balance_chart"),
                        class_="card",
                        style="margin:10px;"
                    )
                    
                )
                
            )
        ),
        
        
    ),
    title=ui.tags.div(
        ui.img(src="business-science-logo.png", height="50px", style="margin:5px;"),
        ui.h4(" " + TITLE, style="color:white;margin-top:auto; margin-bottom:auto;"), 
        style="display:flex;-webkit-filter: drop-shadow(2px 2px 2px #222);"
    ),
    bg="#0062cc",
    inverse=True,
    header=page_dependencies
)


def server(input, output, session: Session):
    
    # Reactivity
    model     = reactive.Value(xgb_model)
    sample_df = reactive.Value(df_sample)
    preds_df  = reactive.Value(df_predictions_sample)
    
    
    @reactive.Effect
    @reactive.event(input.submit)
    def _1():
        if input.model_selected() == "XGBoost":
            model.set(xgb_model)
        else:
            model.set(lgb_model)
    
    
    @reactive.Effect
    @reactive.event(input.submit)
    def _2():
        df_sample = lending_data_raw_df.sample(
            frac=float(input.fraction()),
            random_state=123
        ) 
        
        sample_df.set(df_sample)
    
    @reactive.Effect
    @reactive.event(input.submit)
    def _3():
        df = clf.predict_model(
            model(), 
            sample_df(),
            raw_score=True
        )
        
        filtered_df = df[df['prediction_score_bad'] > input.filter_loans()]
        
        preds_df.set(filtered_df)
    
    
    @reactive.Effect
    def _3():
        
        print(preds_df())
        
        fig = px.scatter(
            data_frame = preds_df(),
            x = 'int_rate',
            y = 'prediction_score_bad',
            color = 'prediction_score_bad',
            trendline='lowess',
            trendline_color_override="green",
            # template='plotly_dark',
            log_x=True,
            log_y=True,
            # color_continuous_scale="RdBu",
            title = "Interest Rate vs Loan Default Score"
        )
        
        # fig.update_layout(
        #     plot_bgcolor="rgba(0, 0, 0, 0)",
        #     paper_bgcolor="rgba(0, 0, 0, 0)"
        # )
        
        register_widget("interest_rate_chart", go.FigureWidget(fig))
        
    @reactive.Effect
    def _4():
        
        fig = px.scatter(
            data_frame = preds_df(),
            x = 'total_bal_il',
            y = 'prediction_score_bad',
            color = 'prediction_score_bad',
            trendline='lowess',
            trendline_color_override="green",
            # template='plotly_dark',
            log_x=True,
            log_y=True,
            # color_continuous_scale="RdBu",
            title="Total Balance Vs Bad Loan Score"
        )
        
        # fig.update_layout(
        #     plot_bgcolor="rgba(0, 0, 0, 0)",
        #     paper_bgcolor="rgba(0, 0, 0, 0)"
        # )
        
        register_widget("total_balance_chart", go.FigureWidget(fig))
    

import os

# get the current working directory
current_working_directory = os.getcwd()
print(current_working_directory)

# try:
app = App(
    app_ui, server, 
    static_assets=current_working_directory+'/'+www_dir, 
    debug=False
)
# except Exception as e:
#     print("Error loading model:", e) 