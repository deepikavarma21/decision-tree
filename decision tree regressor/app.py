# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ======================================================
# SIMPLE DARK CSS
# ======================================================

st.markdown("""
<style>
.stApp{
    background-color:#0E1117;
    color:white;
}

h1,h2,h3,h4{
    color:#58A6FF;
}

[data-testid="stSidebar"]{
    background-color:#161B22;
}

.metric-box{
    background:#161B22;
    padding:20px;
    border-radius:10px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# INR CONVERSION
# ======================================================

EUR_TO_INR = 90.5

def to_inr(x):
    return x * EUR_TO_INR

# ======================================================
# LOAD + PREPROCESS DATA
# ======================================================

DATA_PATH = "laptop_price.csv"

@st.cache_data
def load_raw_df(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1")


def _extract_number(x, kind: str):
    """Extract numeric value from strings like '8GB', '1.37kg', '32GB SSD', etc."""
    if pd.isna(x):
        return np.nan

    s = str(x)
    # keep digits and dot
    if kind == "int":
        s = "".join(ch for ch in s if ch.isdigit())
        return int(s) if s else np.nan
    if kind == "float":
        allowed = set("0123456789.")
        s2 = "".join(ch for ch in s if ch in allowed)
        return float(s2) if s2 and s2 != "." else np.nan

    return np.nan


@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean Ram -> Ram_num
    df["Ram_num"] = df["Ram"].apply(lambda v: _extract_number(v, "int"))

    # Clean Weight -> Weight_num
    df["Weight_num"] = df["Weight"].apply(lambda v: _extract_number(v, "float"))

    # Normalize common typos/variants (keep minimal)
    df["OpSys"] = df["OpSys"].replace({
        "Mac OS X": "macOS",
        "macOS": "macOS",
        "No OS": "Linux",
        "Windows 10 S": "Windows 10",
    })

    # Keep only what our model uses
    keep_cols = [
        "Company",
        "TypeName",
        "Inches",
        "Weight_num",
        "Ram_num",
        "OpSys",
        "Price_euros",
    ]
    df = df[keep_cols]

    return df

# ======================================================
# END PREPROCESS
# ======================================================


# (old synthetic preprocess removed)


# ======================================================
# TRAIN MODEL
# ======================================================


@st.cache_resource
def train_model(df, max_depth, test_size):

    numeric_features = [
        "Inches",
        "Weight_num",
        "Ram_num"
    ]




    categorical_features = [
        "Company",
        "TypeName",
        "OpSys"
    ]

    X = df[numeric_features + categorical_features]
    y = df["Price_euros"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    preprocessor = ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_pred, mae, rmse, r2

# ======================================================
# LOAD DATA
# ======================================================

df_raw = load_raw_df()

df = preprocess(df_raw)

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.header("⚙️ Model Settings")

    max_depth = st.slider(
        "Max Depth",
        2,
        20,
        5
    )

    test_size = st.slider(
        "Test Size",
        0.1,
        0.4,
        0.2
    )

    st.header("💻 Predict Laptop")

    company = st.selectbox(
        "Company",
        ['Apple','Dell','HP','Lenovo','Asus','MSI','Acer']
    )

    typename = st.selectbox(
        "Type",
        ['Ultrabook','Notebook','Gaming','Workstation']
    )

    ram = st.selectbox(
        "RAM",
        ['4GB','8GB','16GB','32GB']
    )

    inches = st.slider(
        "Screen Size",
        13.0,
        18.0,
        15.6
    )

    weight = st.slider(
        "Weight",
        1.0,
        5.0,
        2.0
    )

    opsys = st.selectbox(
        "Operating System",
        ['Windows 10','macOS','Linux']
    )

# ======================================================
# TRAIN
# ======================================================

model, X_train, X_test, y_train, y_test, y_pred, mae, rmse, r2 = train_model(
    df,
    max_depth,
    test_size
)

# ======================================================
# TITLE
# ======================================================

st.title("💻 Laptop Price Predictor")

st.write(
    "Decision Tree Regression model for predicting laptop prices."
)

# ======================================================
# METRICS
# ======================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

with col2:
    st.metric(
        "MAE",
        f"₹{to_inr(mae):,.0f}"
    )

with col3:
    st.metric(
        "RMSE",
        f"₹{to_inr(rmse):,.0f}"
    )

# ======================================================
# PREDICTION
# ======================================================

input_df = pd.DataFrame([{
    "Inches": inches,
    "Weight_num": weight,
    "Ram_num": int(ram.replace("GB","")),
    "Company": company,
    "TypeName": typename,
    "OpSys": opsys
}])


pred = model.predict(input_df)[0]

st.subheader("🔮 Predicted Laptop Price")

st.success(f"Estimated Price: ₹ {to_inr(pred):,.0f}")

# ======================================================
# DATASET
# ======================================================

st.subheader("📊 Dataset")

st.dataframe(df.head(20), use_container_width=True)

# ======================================================
# ACTUAL VS PREDICTED
# ======================================================

st.subheader("📈 Actual vs Predicted")

fig1, ax1 = plt.subplots(figsize=(6,5))

actual = np.array(y_test) * EUR_TO_INR
predicted = y_pred * EUR_TO_INR

ax1.scatter(actual, predicted, alpha=0.5)

min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())

ax1.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")

st.pyplot(fig1)

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

st.subheader("🌳 Feature Importance")

regressor = model.named_steps["regressor"]

ohe_cols = (
    model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out([
        "Company",
        "TypeName",
        "OpSys"
    ])
)

all_features = [
    "Inches",
    "Weight_num",
    "Ram_num"
] + list(ohe_cols)


importance = regressor.feature_importances_

importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax2.invert_yaxis()

st.pyplot(fig2)

# ======================================================
# DECISION TREE
# ======================================================

st.subheader("🌲 Decision Tree")

X_small = pd.get_dummies(
    df[[
        "Company",
        "TypeName",
        "Inches",
        "Weight_num",
        "Ram_num"
    ]]

)

tree_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

tree_model.fit(X_small, df["Price_euros"])

fig3, ax3 = plt.subplots(figsize=(20,8))

plot_tree(
    tree_model,
    feature_names=X_small.columns,
    filled=True,
    fontsize=7
)

st.pyplot(fig3)

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.write("Built using Streamlit and Scikit-learn")