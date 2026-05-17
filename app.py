import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Iris Decision Tree",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
}

.hero {
    background: linear-gradient(90deg, #e94560, #f39c12, #1abc9c);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 20px;
}

.hero h1 {
    color: white;
    font-size: 42px;
}

.metric-card {
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    color: white;
    font-weight: bold;
    margin-bottom: 10px;
}

.green { background: linear-gradient(135deg,#11998e,#38ef7d); }
.orange { background: linear-gradient(135deg,#f7971e,#ffd200); }
.blue { background: linear-gradient(135deg,#4776e6,#8e54e9); }
.pink { background: linear-gradient(135deg,#e94560,#c0392b); }

.section {
    color: #f39c12;
    font-size: 24px;
    margin-top: 25px;
    margin-bottom: 15px;
    font-weight: bold;
}

.prediction {
    padding: 25px;
    border-radius: 14px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
}

.setosa {
    background: linear-gradient(135deg,#11998e,#38ef7d);
}

.versicolor {
    background: linear-gradient(135deg,#4776e6,#8e54e9);
}

.virginica {
    background: linear-gradient(135deg,#e94560,#c0392b);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = pd.Categorical.from_codes(
        iris.target,
        iris.target_names
    )

    return df, iris

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(test_size,
                max_depth,
                criterion,
                splitter,
                random_state,
                use_grid):

    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if use_grid:

        params = {
            "criterion": ["gini", "entropy"],
            "max_depth": [2,3,4,5,None],
            "splitter": ["best","random"]
        }

        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=random_state),
            params,
            cv=5
        )

        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        best_params = grid.best_params_

    else:

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth if max_depth > 0 else None,
            splitter=splitter,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        best_params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "splitter": splitter
        }

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        output_dict=True
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        acc,
        cm,
        report,
        best_params,
        iris
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.title("⚙️ Settings")

    use_grid = st.toggle(
        "Use GridSearchCV",
        value=False
    )

    test_size = st.slider(
        "Test Size",
        0.1,
        0.4,
        0.2,
        0.05
    )

    random_state = st.slider(
        "Random State",
        0,
        100,
        42
    )

    if not use_grid:

        criterion = st.selectbox(
            "Criterion",
            ["gini", "entropy", "log_loss"]
        )

        splitter = st.selectbox(
            "Splitter",
            ["best", "random"]
        )

        max_depth = st.slider(
            "Max Depth (0 = None)",
            0,
            20,
            3
        )

    else:

        criterion = "gini"
        splitter = "best"
        max_depth = 3

    st.markdown("---")

    st.subheader("🔮 Live Prediction")

    sl = st.slider(
        "Sepal Length",
        4.0,
        8.0,
        5.8,
        0.1
    )

    sw = st.slider(
        "Sepal Width",
        1.5,
        5.0,
        3.0,
        0.1
    )

    pl = st.slider(
        "Petal Length",
        1.0,
        7.0,
        3.7,
        0.1
    )

    pw = st.slider(
        "Petal Width",
        0.1,
        2.6,
        1.2,
        0.1
    )

# ─────────────────────────────────────────────
# LOAD + TRAIN
# ─────────────────────────────────────────────
df, iris_raw = load_data()

(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    acc,
    cm,
    report,
    best_params,
    iris_obj
) = train_model(
    test_size,
    max_depth,
    criterion,
    splitter,
    random_state,
    use_grid
)

class_names = iris_obj.target_names.tolist()
feature_names = iris_obj.feature_names

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
<h1>🌸 IRIS DECISION TREE CLASSIFIER</h1>
<p>Interactive Machine Learning Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
depth = model.get_depth()
leaves = model.get_n_leaves()
train_acc = accuracy_score(
    y_train,
    model.predict(X_train)
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card green">
    <h2>{acc*100:.1f}%</h2>
    Test Accuracy
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card orange">
    <h2>{train_acc*100:.1f}%</h2>
    Train Accuracy
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card blue">
    <h2>{depth}</h2>
    Tree Depth
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card pink">
    <h2>{leaves}</h2>
    Leaf Nodes
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section">📌 Model Settings</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

m1.info(f"Criterion: {model.criterion}")
m2.info(f"Max Depth: {depth}")
m3.info(f"Splitter: {model.splitter}")

# ─────────────────────────────────────────────
# OVERFITTING WARNING
# ─────────────────────────────────────────────
if depth > 8:
    st.warning("⚠️ High tree depth may cause overfitting.")
else:
    st.success("✅ Tree depth looks good.")

# ─────────────────────────────────────────────
# LIVE PREDICTION
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section">🔮 Live Prediction</div>',
    unsafe_allow_html=True
)

user_input = np.array([[sl, sw, pl, pw]])

pred_idx = model.predict(user_input)[0]

pred_name = class_names[pred_idx]

pred_proba = model.predict_proba(user_input)[0]

confidence = np.max(pred_proba)

if pred_name == "setosa":
    cls = "setosa"
    flower = "🌼"

elif pred_name == "versicolor":
    cls = "versicolor"
    flower = "🌸"

else:
    cls = "virginica"
    flower = "🌺"

st.markdown(f"""
<div class="prediction {cls}">
{flower} Iris {pred_name.capitalize()}
</div>
""", unsafe_allow_html=True)

st.info(
    f"Prediction Confidence: {confidence*100:.2f}%"
)

st.progress(float(acc))

if confidence > 0.95:
    st.balloons()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset",
    "🌳 Decision Tree",
    "📉 Confusion Matrix",
    "📋 Report",
    "🔬 Feature Analysis"
])

# ─────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────
with tab1:

    st.markdown(
        '<div class="section">Dataset</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Dataset",
        csv,
        "iris_dataset.csv",
        "text/csv"
    )

    fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(
        df.select_dtypes("number").corr(),
        annot=True,
        cmap="plasma",
        ax=ax
    )

    st.pyplot(fig)

# ─────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────
with tab2:

    st.markdown(
        '<div class="section">Decision Tree</div>',
        unsafe_allow_html=True
    )

    if use_grid:
        st.success(
            f"Best Parameters: {best_params}"
        )

    fig_tree, ax_tree = plt.subplots(
        figsize=(18,8)
    )

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9
    )

    st.pyplot(fig_tree)

    st.markdown(
        '<div class="section">Feature Importances</div>',
        unsafe_allow_html=True
    )

    importances = model.feature_importances_

    fig2, ax2 = plt.subplots(figsize=(7,3))

    ax2.barh(
        feature_names,
        importances
    )

    st.pyplot(fig2)

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    st.dataframe(
        fi_df.sort_values(
            by="Importance",
            ascending=False
        ),
        use_container_width=True
    )

# ─────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────
with tab3:

    st.markdown(
        '<div class="section">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    fig_cm, ax_cm = plt.subplots(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=class_names,
        yticklabels=class_names
    )

    st.pyplot(fig_cm)

# ─────────────────────────────────────────────
# TAB 4
# ─────────────────────────────────────────────
with tab4:

    st.markdown(
        '<div class="section">Classification Report</div>',
        unsafe_allow_html=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

# ─────────────────────────────────────────────
# TAB 5
# ─────────────────────────────────────────────
with tab5:

    st.markdown(
        '<div class="section">Feature Analysis</div>',
        unsafe_allow_html=True
    )

    feat_x = st.selectbox(
        "X-axis",
        feature_names,
        index=2
    )

    feat_y = st.selectbox(
        "Y-axis",
        feature_names,
        index=3
    )

    fig_sc, ax_sc = plt.subplots(figsize=(7,5))

    palette = {
        "setosa":"green",
        "versicolor":"blue",
        "virginica":"red"
    }

    for sp, color in palette.items():

        mask = df["species"] == sp

        ax_sc.scatter(
            df.loc[mask, feat_x],
            df.loc[mask, feat_y],
            label=sp,
            color=color
        )

    xi = feature_names.index(feat_x)
    yi = feature_names.index(feat_y)

    ax_sc.scatter(
        user_input[0, xi],
        user_input[0, yi],
        color="yellow",
        s=300,
        marker="*",
        label="Your Input"
    )

    ax_sc.legend()

    st.pyplot(fig_sc)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center'>
    🌸 Built with Streamlit & Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)