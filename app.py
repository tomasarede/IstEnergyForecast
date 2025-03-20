from flask import Flask, render_template, jsonify, request
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
from datetime import timedelta
import plotly.io as pio
import base64
import google.generativeai as genai
from PIL import Image, ImageOps
from io import BytesIO

app = Flask(__name__)

# ----------------------------
# Data Loading & Preparation
# ----------------------------
CSV_FILE = "proj1/Power_Meteo_Merged.csv"
df = pd.read_csv(CSV_FILE)
df = df.drop(columns=["Year", "Month", "Day", "Hour"])
df["Date_start"] = pd.to_datetime(df["Date_start"])
df.sort_values(by="Date_start", inplace=True)

# List available features (all columns except "Date_start")
available_features = [col for col in df.columns if col != "Date_start"]

@app.route("/")
def home():
    return render_template("index.html", features=available_features)

# Set external stylesheet to use your main.css from the static folder.
external_stylesheets = [app.static_url_path + "/css/main.css"]

# ----------------------------
# Dash App for Data Preparation (South Tower Power Forecast)
# ----------------------------
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/",
    external_stylesheets=external_stylesheets,
)

dash_app.layout = html.Div(
    className="dark-background",
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[
        dcc.Dropdown(
            id="feature-dropdown",
            options=[{"label": col, "value": col} for col in available_features],
            value=available_features[0],
            style={"width": "50%", "fontFamily": "var(--default-font)"}
        ),
        dcc.Graph(id="graph")
    ]
)

@dash_app.callback(
    Output("graph", "figure"),
    Input("feature-dropdown", "value")
)
def update_graph(feature):
    x = df["Date_start"]
    y = df[feature]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=feature
    ))
    fig.update_layout(
        title=f"{feature} Over Time",
        xaxis_title="Date",
        yaxis_title=feature,
        template="seaborn",
        xaxis=dict(rangeslider=dict(visible=True))
    )
    return fig





#-------------------------------
# Dash app for EDA
#-------------------------------


df_box = df.drop(columns=["Date_start"])
available_features = df_box.columns.tolist()


dash_boxplots = dash.Dash(
    __name__,
    server=app,  
    url_base_pathname="/dash/boxplots/",
    external_stylesheets=external_stylesheets  
)

dash_boxplots.layout = html.Div([
    dcc.Dropdown(
        id="boxplot-dropdown",
        options=[{"label": feature, "value": feature} for feature in available_features],
        value=available_features[0], 
        multi=False,  
        placeholder="Select a feature...",
        style={"width": "50%", "fontFamily": "var(--default-font)"}

    ),
    dcc.Graph(id="boxplot-graph")
], style={"padding": "20px", "fontFamily": "var(--default-font)"})

@dash_boxplots.callback(
    Output("boxplot-graph", "figure"),
    [Input("boxplot-dropdown", "value")]
)
def update_boxplot(selected_feature):
    if not selected_feature:
        return go.Figure()  

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df_box[selected_feature].dropna(),
        name=selected_feature,
        marker=dict(color='royalblue')
    ))

    fig.update_layout(
        title=f"Box Plot for {selected_feature}",
        yaxis_title="Values",
        template="seaborn"
    )
    return fig


# ----------------
# Correlation
# ----------------
with open("dash_apps/corr_figure.json", "r") as f:
    fig_json = f.read()

fig_corr = pio.from_json(fig_json)


# Create a new Dash app for the correlation heatmap; mount it at "/dash/correlation/"
dash_app_corr = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/correlation/",
    external_stylesheets=external_stylesheets,
)

dash_app_corr.layout = html.Div(
    className="dark-background",
    style={
        "padding": "20px",
        "fontFamily": "var(--default-font)",
        "height": "100vh"  # Full viewport height for the container
    },
    children=[
        dcc.Graph(
            figure=fig_corr,
            style={"height": "80vh", "width": "100%"}  # Graph takes up 80% of the viewport height
        )
    ]
)


# ----------------
# F-Test
# ----------------

with open("dash_apps/kbest_figure.json", "r") as f:
    fig_json = f.read()

fig_kbest = pio.from_json(fig_json)


# Create a new Dash instance for F-Test
dash_app_kbest = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/kbest/",
    external_stylesheets=external_stylesheets,
)

dash_app_kbest.layout = html.Div(
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[
        dcc.Graph(figure=fig_kbest)
    ]
)
# ----------------------------
# Dash App for Mutual Information (SelectKBest)
# ----------------------------
with open("dash_apps/mutual_figure.json", "r") as f:
    fig_json = f.read()

fig_mutual = pio.from_json(fig_json)

dash_app_mutual = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/mutualinfo/",
    external_stylesheets=external_stylesheets,
)
dash_app_mutual.layout = html.Div(
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[ dcc.Graph(figure=fig_mutual) ]
)

# ----------------------------
# Dash App for Wrapper Method: RFE
# ----------------------------
with open("dash_apps/rfe_figure.json", "r") as f:
    fig_json = f.read()

fig_rfe = pio.from_json(fig_json)

dash_app_rfe = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/rfe/",
    external_stylesheets=external_stylesheets,
)
dash_app_rfe.layout = html.Div(
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[ dcc.Graph(figure=fig_rfe) ]
)




#----------------------------------
# Drop Features and Clean Data
#----------------------------------
CSV_FILE = "proj1/Data_Droped_Interpolation.csv"
df_clean = pd.read_csv(CSV_FILE)
df_clean["Date_start"] = pd.to_datetime(df_clean["Date_start"])
df_clean.sort_values(by="Date_start", inplace=True)
available_features = [col for col in df_clean.columns if col != "Date_start" and col!= "Power_kW"]

# --- Dash App for Plotting Cleaned Data with Added Data Highlighted ---
dash_app_cleaned = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/plot_cleaned_completed/",
    external_stylesheets=external_stylesheets,
)

dash_app_cleaned.layout = html.Div(
    className="dark-background",
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[
        html.H2(
            "Dash: Cleaned and Completed Data",
            style={
                "color": "var(--heading-color)",
                "fontFamily": "var(--heading-font)",
                "marginBottom": "20px"
            }
        ),
        dcc.Dropdown(
            id="cleaned-feature-dropdown",
            options=[{"label": col, "value": col} for col in available_features],
            value=available_features[0],
            style={"width": "50%", "fontFamily": "var(--default-font)"}
        ),
        dcc.Graph(id="graph-cleaned")
    ]
)


def segment_data(df_seg, x_col, y_col, threshold=timedelta(days=1)):

    segments = []
    current_segment = []
    previous = None
    for idx, row in df_seg.iterrows():
        if previous is None:
            current_segment.append(row)
        else:
            if (row[x_col] - previous[x_col]) > threshold:
                segments.append(pd.DataFrame(current_segment))
                current_segment = [row]
            else:
                current_segment.append(row)
        previous = row
    if current_segment:
        segments.append(pd.DataFrame(current_segment))
    return segments

@dash_app_cleaned.callback(
    Output("graph-cleaned", "figure"),
    Input("cleaned-feature-dropdown", "value")
)
def update_graph_data_cleaned(feature):
    # Separate the cleaned data into "original" and "added" parts
    mask_orig = df_clean["Date_start"].isin(df.dropna()["Date_start"])
    df_orig = df_clean[mask_orig].sort_values("Date_start")
    df_added = df_clean[~mask_orig].sort_values("Date_start")
    
    # Segment each group based on a threshold gap (here, 1 day)
    threshold = timedelta(days=1)  # adjust if needed
    orig_segments = segment_data(df_orig, "Date_start", feature, threshold)
    added_segments = segment_data(df_added, "Date_start", feature, threshold)
    
    fig = go.Figure()
    # Plot each original segment as its own trace
    for i, seg in enumerate(orig_segments):
        fig.add_trace(go.Scatter(
            x=seg["Date_start"],
            y=seg[feature],
            mode="lines",
            name="Initial Data",
            line=dict(color="royalblue"),
        ))
    # Plot each added segment as its own trace
    for i, seg in enumerate(added_segments):
        fig.add_trace(go.Scatter(
            x=seg["Date_start"],
            y=seg[feature],
            mode="lines",
            name="Added Data",
            line=dict(color="red"),
        ))
    
    fig.update_layout(
         title=f"{feature} Over Time (Original + Added)",
         xaxis_title="Date",
         yaxis_title=feature,
         template="seaborn",
         xaxis=dict(rangeslider=dict(visible=True)),
         showlegend = False
    )
    return fig


#------------------
# NEW FEATURES
#------------------

# ----------------------------
# 1. F-Test using f_regression on Cleaned Data
# ----------------------------

with open("dash_apps/clean_f_test_figure.json", "r") as f:
    fig_json = f.read()

fig_clean_f_test = pio.from_json(fig_json)

dash_app_clean_f_test = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/clean_f_test/",
    external_stylesheets=external_stylesheets,
)
dash_app_clean_f_test.layout = html.Div(
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[dcc.Graph(figure=fig_clean_f_test)]
)

# ----------------------------
# 2. Mutual Information on Cleaned Data
# ----------------------------

with open("dash_apps/clean_mi_figure.json", "r") as f:
    fig_json = f.read()

fig_clean_mi = pio.from_json(fig_json)

dash_app_clean_mi = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/clean_mutualinfo/",
    external_stylesheets=external_stylesheets,
)
dash_app_clean_mi.layout = html.Div(
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[dcc.Graph(figure=fig_clean_mi)]
)

# ----------------------------
# 3. RFE (Recursive Feature Elimination) on Cleaned Data
# ----------------------------
with open("dash_apps/clean_rfe_figure.json", "r") as f:
    fig_json = f.read()

fig_clean_rfe = pio.from_json(fig_json)

dash_app_clean_rfe = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/clean_rfe/",
    external_stylesheets=external_stylesheets,
)
dash_app_clean_rfe.layout = html.Div(
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[dcc.Graph(figure=fig_clean_rfe)]
)

# ----------------------------
# 4. Random Forest Feature Importance on Cleaned Data
# ----------------------------
with open("dash_apps/clean_rf_figure.json", "r") as f:
    fig_json = f.read()

fig_clean_rf = pio.from_json(fig_json)

dash_app_clean_rf = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/clean_rf/",
    external_stylesheets=external_stylesheets,
)
dash_app_clean_rf.layout = html.Div(
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[dcc.Graph(figure=fig_clean_rf)]
)

# -----------------------------
# Correlation for Cleaned Data
# -----------------------------

with open("dash_apps/clean_corr_figure.json", "r") as f:
    fig_json = f.read()

fig_corr_clean = pio.from_json(fig_json)
# Create a new Dash app for the cleaned correlation heatmap
dash_app_clean_corr = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/clean_corr/",
    external_stylesheets=external_stylesheets,
)

dash_app_clean_corr.layout = html.Div(
    className="dark-background",
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[
        dcc.Graph(
            figure=fig_corr_clean,
            style={"height": "80vh", "width": "100%"}
        )
    ]
)




# ------------------------
# DASH APP FOR VALIDATION
# ------------------------

with open("dash_apps/val_xgb_figure.json", "r") as f:
    fig_json = f.read()

fig_validation_xgb = pio.from_json(fig_json)

with open("dash_apps/val_nn_figure.json", "r") as f:
    fig_json = f.read()

fig_validation_nn = pio.from_json(fig_json)

# ------------------------
# Dash App for XGBoost 
# ------------------------
dash_app_validation_xgb = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/validation_xgb/",
    external_stylesheets=external_stylesheets,
)
dash_app_validation_xgb.layout = html.Div(
    className="dark-background",
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[dcc.Graph(figure=fig_validation_xgb)]
)

# ------------------------
# Dash App for NN
# ------------------------
dash_app_validation_nn = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dash/validation_nn/",
    external_stylesheets=external_stylesheets,
)
dash_app_validation_nn.layout = html.Div(
    className="dark-background",
    style={"padding": "20px", "fontFamily": "var(--default-font)"},
    children=[dcc.Graph(figure=fig_validation_nn)]
)



# -------------- 
# GEMINI PART
# --------------

GEMINI_API_KEY ="AIzaSyAKyIXA_j0EZZ9mxPovSVOfQ8ijDT7G0lY"
genai.configure(api_key=GEMINI_API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def resize_image(image_data, max_size=(1024, 1024)):
    """Resizes image data to a maximum size using LANCZOS resampling."""
    try:
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

@app.route('/ai_generate', methods=['POST'])
def ai_generate():
    print(f"google.generativeai version: {genai.__version__}")  # Print the version.
    data = request.get_json()
    model_type = data.get("model_type")

    if model_type == "XGBoost":
        prompt_text = (
            "Please analyze the following XGBoost model evaluation:\n\n"
            "Model: XGBoost (default settings).\n"
            "Metrics:\n"
            " - MAE: 16.76\n"
            " - MBE: 5.00\n"
            " - MSE: 1210.94\n"
            " - RMSE: 34.80\n"
            " - cvRMSE: 19.02%\n"
            " - NMBE: 2.73%\n"
            " - R²: 0.891\n\n"
            "Features: HR, solarRad_W/m2, Weekday, Hour_sin, Hour_cos, "
            "Power_kW_lag_1h, Power_kW_lag_2h, Power_kW_lag_24h, HDH, "
            "CDH_squared, CDH_Humidity, log_temp, holiday.\n\n"
            "Please analyze the performance graph shown below and return your analysis as beautifully formatted HTML, with clear headings and paragraphs. "
            "For the heading use h5. Don't show the plot of the figure just text."
            "Discuss any missing trends or peaks and suggest improvements."
        )
        image_path = "XGBoostGraph.png"

    elif model_type == "NN":
        prompt_text = (
            "Please analyze the following Neural Network model evaluation:\n\n"
            "Model Architecture: EnergyPredictor with layers:\n"
            " - fc1: Linear(13, 128)\n"
            " - bn1: BatchNorm1d(128)\n"
            " - fc2: Linear(128, 256)\n"
            " - bn2: BatchNorm1d(256)\n"
            " - fc3: Linear(256, 128)\n"
            " - bn3: BatchNorm1d(128)\n"
            " - fc4: Linear(128, 64)\n"
            " - fc5: Linear(64, 1)\n"
            " - dropout: Dropout(p=0.2)\n\n"
            "Metrics:\n"
            " - MAE: 15.36\n"
            " - MBE: 5.71\n"
            " - MSE: 1084.24\n"
            " - RMSE: 32.93\n"
            " - cvRMSE: 17.99%\n"
            " - NMBE: 3.12%\n"
            " - R²: 0.902\n\n"
            "Features: HR, solarRad_W/m2, Weekday, Hour_sin, Hour_cos, "
            "Power_kW_lag_1h, Power_kW_lag_2h, Power_kW_lag_24h, HDH, "
            "CDH_squared, CDH_Humidity, log_temp, holiday.\n\n"
            "Please analyze the performance graph shown below and return your analysis as beautifully formatted HTML, with clear headings and paragraphs. "
            "For the heading use h5. Don't show the plot of the figure just text."
            "Also, have in mind that was used batch normalization, dropout, l2 regularization, adam optimizer, batch size of 64, early stop and adaptive learning rate. There was no overfitting."
            "Explain any discrepancies or missing trends, and suggest improvements."
        )
        image_path = "NNGraph.png"
    else:
        return jsonify({"analysis": "Model type not recognized."}), 400

    # Encode and resize the image.
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    resized_image_data = resize_image(image_data)

    if resized_image_data:
        try:
            # Use the GenerativeModel with the multimodal payload.
            response = genai.GenerativeModel("models/gemini-2.0-flash").generate_content(
                contents=[prompt_text, {"mime_type": "image/png", "data": resized_image_data}]
            )
            result = response.text
            return jsonify({"analysis": result})
        except Exception as e:
            return jsonify({"analysis": f"Error contacting Gemini AI: {str(e)}"}), 500
    else:
        return jsonify({"analysis": "Image resizing failed."}), 500


if __name__ == "__main__":
    app.run(debug=True)
