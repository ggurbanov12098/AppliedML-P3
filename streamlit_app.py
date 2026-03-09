"""
Course Project 3 — Resampling & Model Selection
Auto MPG Dataset  |  Streamlit Interactive UI
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from itertools import combinations
import io

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto MPG — ML Explorer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .hero-banner {
        background: linear-gradient(135deg, #1a1f35 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-banner h1 { color: #e2e8f0; font-size: 2.4rem; font-weight: 700; margin:0; }
    .hero-banner p  { color: #94a3b8; font-size: 1.05rem; margin-top: 0.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #1e2740, #16213e);
        border: 1px solid #2d3a5e;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-card .lbl { font-size: 0.82rem; color: #94a3b8; margin-top: 0.2rem; }

    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #1a2744);
        border-left: 4px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 0.7rem 1.2rem;
        margin: 1.5rem 0 1rem 0;
        color: #e2e8f0;
        font-size: 1.15rem;
        font-weight: 600;
    }

    .result-box {
        background: #1a2035;
        border: 1px solid #2d3a5e;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-top: 0.8rem;
    }
    .result-box p { color: #94a3b8; margin: 0.25rem 0; font-size: 0.93rem; }
    .result-box .highlight { color: #34d399; font-weight: 600; font-size: 1rem; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
    }
    div[data-testid="stSidebar"] .css-1d391kg { padding: 1rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2740;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #94a3b8;
        border: 1px solid #2d3a5e;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e40af, #1d4ed8);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }

    .info-pill {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ["mpg","cylinders","displacement","horsepower","weight",
            "acceleration","model_year","origin","car_name"]
    try:
        df = pd.read_csv(url, names=cols, sep=r"\s+", na_values="?")
    except Exception:
        # Fallback: generate synthetic dataset
        np.random.seed(42)
        n = 392
        df = pd.DataFrame({
            "mpg": np.random.normal(23.5, 7.8, n).clip(9, 47),
            "cylinders": np.random.choice([4,6,8], n, p=[0.5,0.28,0.22]),
            "displacement": np.random.normal(194, 104, n).clip(68, 455),
            "horsepower": np.random.normal(104, 38, n).clip(46, 230),
            "weight": np.random.normal(2978, 847, n).clip(1613, 5140),
            "acceleration": np.random.normal(15.5, 2.8, n).clip(8, 24.8),
            "model_year": np.random.randint(70, 83, n),
            "origin": np.random.choice([1,2,3], n, p=[0.62,0.18,0.20]),
            "car_name": ["car"] * n
        })
    df.drop("car_name", axis=1, inplace=True)
    df.dropna(inplace=True)
    df["horsepower"] = df["horsepower"].astype(float)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def prepare_data(df):
    feat_cols = ["cylinders","displacement","horsepower","weight",
                 "acceleration","model_year","origin"]
    X = df[feat_cols].values
    y = df["mpg"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feat_cols, scaler

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Auto MPG Explorer")
    st.markdown("---")
    st.markdown("**Course Project 3**")
    st.markdown("Resampling & Model Selection")
    st.markdown("---")

    page = st.radio("📂 Navigate", [
        "🏠 Overview & EDA",
        "🔄 Resampling Methods",
        "🎯 Subset Selection",
        "📉 Shrinkage (Ridge/Lasso)",
        "🔬 PCA & PLS",
        "🏆 Model Comparison",
        "🧪 Live Prediction Lab"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** UCI Auto MPG")
    st.markdown("**Target:** `mpg` (fuel efficiency)")
    st.markdown("**Task:** Regression")
    st.markdown("---")
    st.markdown("<span class='info-pill'>scikit-learn</span>"
                "<span class='info-pill'>statsmodels</span>"
                "<span class='info-pill'>Streamlit</span>", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = load_data()
X_scaled, y, feature_cols, scaler = prepare_data(df)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & EDA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & EDA":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🚗 Auto MPG — Resampling & Model Selection</h1>
        <p>Interactive exploration of resampling methods and model selection techniques on the UCI Auto MPG dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [len(df), len(feature_cols), f"{df['mpg'].mean():.1f}", f"{df['mpg'].std():.1f}"],
        ["Total Samples", "Features", "Mean MPG", "Std MPG"]
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{val}</div>
            <div class='lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📊 Dataset Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-header'>📈 Feature Distributions</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), facecolor='#0f1117')
        fig.patch.set_facecolor('#0f1117')
        all_cols = feature_cols + ['mpg']
        for i, col_name in enumerate(all_cols):
            ax = axes[i // 4][i % 4]
            ax.set_facecolor('#1a2035')
            ax.hist(df[col_name], bins=25, color='#3b82f6', edgecolor='#1e3a5f', alpha=0.85)
            ax.set_title(col_name, color='#e2e8f0', fontsize=9)
            ax.tick_params(colors='#94a3b8', labelsize=7)
            for spine in ax.spines.values(): spine.set_edgecolor('#2d3a5e')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='#0f1117')
        ax.set_facecolor('#1a2035')
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                    mask=mask, ax=ax, linewidths=0.5,
                    annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix', color='#e2e8f0', fontsize=12, pad=10)
        ax.tick_params(colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-header'>🔍 Scatter Plots: Feature vs MPG</div>", unsafe_allow_html=True)
    colors = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#8b5cf6','#06b6d4','#f97316']
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor='#0f1117')
    fig.patch.set_facecolor('#0f1117')
    for i, feat in enumerate(feature_cols):
        ax = axes[i // 4][i % 4]
        ax.set_facecolor('#1a2035')
        ax.scatter(df[feat], df['mpg'], alpha=0.4, color=colors[i], s=15, edgecolors='none')
        z = np.polyfit(df[feat], df['mpg'], 1)
        xl = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), 'w--', alpha=0.6, lw=1.5)
        ax.set_xlabel(feat, color='#94a3b8', fontsize=9)
        ax.set_ylabel('mpg', color='#94a3b8', fontsize=9)
        ax.set_title(f'mpg vs {feat}', color='#e2e8f0', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#94a3b8', labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor('#2d3a5e')
    axes[1][3].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RESAMPLING METHODS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Resampling Methods":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🔄 Resampling Methods</h1>
        <p>Cross-Validation (K=5, K=10, LOOCV) and Bootstrap — evaluating Polynomial Regression complexity</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("### ⚙️ Parameters")
        feat_choice = st.selectbox("Feature for Polynomial Regression",
                                   feature_cols, index=3,
                                   help="weight has highest negative correlation with mpg")
        max_deg = st.slider("Max Polynomial Degree", 3, 10, 8)
        n_boot  = st.slider("Bootstrap Samples (B)", 50, 500, 200, step=50)
        run_btn = st.button("▶️ Run Resampling Analysis", use_container_width=True)

    with col_r:
        st.markdown("### 📖 Method Overview")
        st.markdown("""
        <div class='result-box'>
        <p><b style='color:#60a5fa'>K-Fold CV:</b> Split data into K folds. Train on K-1, test on 1. Average MSE over all K iterations.</p>
        <p><b style='color:#34d399'>LOOCV:</b> K = n. Each sample is a test fold once. Low bias, high variance.</p>
        <p><b style='color:#f59e0b'>Bootstrap:</b> Sample with replacement. Out-of-bag samples as test set.</p>
        </div>""", unsafe_allow_html=True)

    if run_btn:
        with st.spinner("Running cross-validation..."):
            X_single = df[[feat_choice]].values
            degrees = list(range(1, max_deg + 1))
            kf5   = KFold(n_splits=5,  shuffle=True, random_state=42)
            kf10  = KFold(n_splits=10, shuffle=True, random_state=42)
            loocv = LeaveOneOut()

            cv_k5, cv_k10, cv_loocv = [], [], []
            for deg in degrees:
                pipe = Pipeline([
                    ('poly',   PolynomialFeatures(degree=deg, include_bias=False)),
                    ('scaler', StandardScaler()),
                    ('lr',     LinearRegression())
                ])
                cv_k5.append(  -cross_val_score(pipe, X_single, y, cv=kf5,   scoring='neg_mean_squared_error').mean())
                cv_k10.append( -cross_val_score(pipe, X_single, y, cv=kf10,  scoring='neg_mean_squared_error').mean())
                cv_loocv.append(-cross_val_score(pipe, X_single, y, cv=loocv, scoring='neg_mean_squared_error').mean())

        with st.spinner("Running bootstrap..."):
            def bootstrap_mse(X, y, degree, n_bootstraps=200, rs=42):
                rng = np.random.RandomState(rs)
                n = len(y)
                mses = []
                for _ in range(n_bootstraps):
                    idx_b = rng.choice(n, n, replace=True)
                    idx_o = list(set(range(n)) - set(idx_b))
                    if not idx_o: continue
                    pipe = Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                                     ('scaler', StandardScaler()), ('lr', LinearRegression())])
                    pipe.fit(X[idx_b], y[idx_b])
                    mses.append(mean_squared_error(y[idx_o], pipe.predict(X[idx_o])))
                return np.mean(mses), np.std(mses)

            boot_mse, boot_std = [], []
            for deg in degrees:
                m, s = bootstrap_mse(X_single, y, deg, n_boot)
                boot_mse.append(m); boot_std.append(s)

        st.session_state['resample_results'] = {
            'degrees': degrees, 'cv_k5': cv_k5, 'cv_k10': cv_k10,
            'cv_loocv': cv_loocv, 'boot_mse': boot_mse, 'boot_std': boot_std
        }

    if 'resample_results' not in st.session_state:
        st.markdown("""
        <div style="background:#1a2035; border:2px dashed #2d3a5e; border-radius:16px;
                    padding:3rem; text-align:center; margin-top:1rem;">
            <div style="font-size:3rem;">▶️</div>
            <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.8rem;">
                Set your parameters and press<br>
                <b style="color:#60a5fa;">Run Resampling Analysis</b> to see results
            </p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    degrees   = st.session_state['resample_results']['degrees']
    cv_k5     = st.session_state['resample_results']['cv_k5']
    cv_k10    = st.session_state['resample_results']['cv_k10']
    cv_loocv  = st.session_state['resample_results']['cv_loocv']
    boot_mse  = st.session_state['resample_results']['boot_mse']
    boot_std  = st.session_state['resample_results']['boot_std']

    best_k5 = degrees[np.argmin(cv_k5)]
    best_k10 = degrees[np.argmin(cv_k10)]
    best_loocv = degrees[np.argmin(cv_loocv)]
    best_boot = degrees[np.argmin(boot_mse)]

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip([c1,c2,c3,c4],
                              [best_k5, best_k10, best_loocv, best_boot],
                              ['Best Deg (K=5)', 'Best Deg (K=10)', 'Best Deg (LOOCV)', 'Best Deg (Bootstrap)']):
        col.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>",
                     unsafe_allow_html=True)

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f1117')
        fig.patch.set_facecolor('#0f1117')

        ax = axes[0]
        ax.set_facecolor('#1a2035')
        ax.plot(degrees, cv_k5,    'o-',  color='#ef4444', lw=2.5, ms=8, label='K=5 CV')
        ax.plot(degrees, cv_k10,   's-',  color='#3b82f6', lw=2.5, ms=8, label='K=10 CV')
        ax.plot(degrees, cv_loocv, '^--', color='#22c55e', lw=2.5, ms=8, label='LOOCV')
        for val, clr in [(best_k5,'#ef4444'),(best_k10,'#3b82f6'),(best_loocv,'#22c55e')]:
            ax.axvline(val, color=clr, ls=':', alpha=0.5, lw=1.5)
        ax.set_xlabel('Polynomial Degree', color='#94a3b8', fontsize=12)
        ax.set_ylabel('MSE', color='#94a3b8', fontsize=12)
        ax.set_title('Cross-Validation MSE vs Degree', color='#e2e8f0', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        ax.legend(fontsize=11, facecolor='#1a2035', labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')

        ax = axes[1]
        ax.set_facecolor('#1a2035')
        ax.errorbar(degrees, boot_mse, yerr=boot_std, fmt='D-',
                    color='#8b5cf6', lw=2.5, ms=8, capsize=5, capthick=2,
                    elinewidth=1.5, label='Bootstrap (OOB ±1σ)')
        ax.fill_between(degrees,
                        np.array(boot_mse)-np.array(boot_std),
                        np.array(boot_mse)+np.array(boot_std),
                        alpha=0.2, color='#8b5cf6')
        ax.axvline(best_boot, color='#8b5cf6', ls=':', lw=2, alpha=0.8,
                   label=f'Best degree = {best_boot}')
        ax.set_xlabel('Polynomial Degree', color='#94a3b8', fontsize=12)
        ax.set_ylabel('MSE', color='#94a3b8', fontsize=12)
        ax.set_title('Bootstrap MSE vs Degree', color='#e2e8f0', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        ax.legend(fontsize=11, facecolor='#1a2035', labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Data table
        st.markdown("<div class='section-header'>📋 Detailed MSE Table</div>", unsafe_allow_html=True)
        tbl = pd.DataFrame({
            'Degree': degrees,
            'K=5 CV MSE': [round(v,3) for v in cv_k5],
            'K=10 CV MSE': [round(v,3) for v in cv_k10],
            'LOOCV MSE': [round(v,3) for v in cv_loocv],
            'Bootstrap MSE': [round(v,3) for v in boot_mse],
            'Bootstrap Std': [round(v,3) for v in boot_std]
        })
        st.dataframe(tbl.style.highlight_min(subset=['K=5 CV MSE','K=10 CV MSE','LOOCV MSE','Bootstrap MSE'],
                                              color='#14532d'), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SUBSET SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Subset Selection":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🎯 Subset Selection Methods</h1>
        <p>Best Subset, Forward Stepwise, Backward Stepwise — with Cp, AIC, BIC, Adj-R² and CV</p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def run_subset_selection(X_data, y_data, feature_names):
        p = X_data.shape[1]

        def compute_metrics_local(X_cols, X_d, y_d):
            X_sub = sm.add_constant(X_d[:, X_cols])
            model = sm.OLS(y_d, X_sub).fit()
            n, k = len(y_d), len(X_cols)
            X_full = sm.add_constant(X_d)
            full_m = sm.OLS(y_d, X_full).fit()
            sigma2_full = np.sum(full_m.resid**2) / (n - X_d.shape[1] - 1)
            rss = np.sum(model.resid**2)
            return {
                'cp':    (rss + 2*k*sigma2_full) / n,
                'aic':   model.aic,
                'bic':   model.bic,
                'adjr2': model.rsquared_adj,
                'mse':   rss / n,
                'cols':  X_cols
            }

        # Best subset
        best_subset = []
        for k in range(1, p+1):
            best_k = None
            for subset in combinations(range(p), k):
                r = compute_metrics_local(list(subset), X_data, y_data)
                if best_k is None or r['mse'] < best_k['mse']:
                    best_k = r
            best_subset.append(best_k)

        # Forward
        fwd = []
        remaining = list(range(p)); selected = []
        while remaining:
            best = None; best_feat = None
            for feat in remaining:
                trial = selected + [feat]
                r = compute_metrics_local(trial, X_data, y_data)
                if best is None or r['aic'] < best['aic']:
                    best = r; best_feat = feat
            selected.append(best_feat); remaining.remove(best_feat)
            fwd.append(best)

        # Backward
        bwd = []
        selected = list(range(p))
        bwd.append(compute_metrics_local(selected, X_data, y_data))
        while len(selected) > 1:
            best = None; worst = None
            for feat in selected:
                trial = [f for f in selected if f != feat]
                r = compute_metrics_local(trial, X_data, y_data)
                if best is None or r['aic'] < best['aic']:
                    best = r; worst = feat
            selected.remove(worst); bwd.append(best)
        bwd = bwd[::-1]

        # CV for best subset
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_mses = []
        for res in best_subset:
            X_sub = X_data[:, res['cols']]
            cv_mses.append(-cross_val_score(LinearRegression(), X_sub, y_data, cv=kf,
                                            scoring='neg_mean_squared_error').mean())
        return best_subset, fwd, bwd, cv_mses

    subset_btn = st.button("▶️ Run Subset Selection", use_container_width=False)

    if subset_btn:
        with st.spinner("Running subset selection (this may take a moment)..."):
            best_subset, fwd_res, bwd_res, cv_subset_mses = run_subset_selection(X_scaled, y, feature_cols)
        st.session_state['subset_results'] = {
            'best_subset': best_subset, 'fwd_res': fwd_res,
            'bwd_res': bwd_res, 'cv_subset_mses': cv_subset_mses
        }

    if 'subset_results' not in st.session_state:
        st.markdown("""
        <div style="background:#1a2035; border:2px dashed #2d3a5e; border-radius:16px;
                    padding:3rem; text-align:center; margin-top:1rem;">
            <div style="font-size:3rem;">▶️</div>
            <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.8rem;">
                Press <b style="color:#60a5fa;">Run Subset Selection</b> to start the analysis
            </p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    best_subset    = st.session_state['subset_results']['best_subset']
    fwd_res        = st.session_state['subset_results']['fwd_res']
    bwd_res        = st.session_state['subset_results']['bwd_res']
    cv_subset_mses = st.session_state['subset_results']['cv_subset_mses']

    p = len(feature_cols)
    k_vals = list(range(1, p+1))

    # Metric plots
    metrics_info = [
        ('aic',   'AIC',          np.argmin, '#3b82f6'),
        ('bic',   'BIC',          np.argmin, '#ef4444'),
        ('cp',    'Mallow\'s Cp', np.argmin, '#22c55e'),
        ('adjr2', 'Adjusted R²',  np.argmax, '#f59e0b'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0f1117')
    fig.patch.set_facecolor('#0f1117')

    for i, (metric, label, bf, color) in enumerate(metrics_info):
        ax = axes[i//2][i%2]
        ax.set_facecolor('#1a2035')
        bv = [r[metric] for r in best_subset]
        fv = [r[metric] for r in fwd_res]
        wv = [r[metric] for r in bwd_res]

        ax.plot(k_vals, bv, 'o-',  color='#ef4444', lw=2, ms=8, label='Best Subset')
        ax.plot(k_vals, fv, 's--', color='#3b82f6', lw=2, ms=8, label='Forward')
        ax.plot(k_vals, wv, '^:',  color='#22c55e', lw=2, ms=8, label='Backward')

        best_k = bf(bv) + 1
        ax.axvline(best_k, color='white', ls=':', alpha=0.5, lw=1.5)
        ax.scatter([best_k], [bv[best_k-1]], color='yellow', s=200, zorder=6, marker='*',
                   label=f'Optimal k={best_k}')

        ax.set_xlabel('Number of Predictors', color='#94a3b8', fontsize=11)
        ax.set_ylabel(label, color='#94a3b8', fontsize=11)
        ax.set_title(f'{label} vs # Predictors', color='#e2e8f0', fontsize=12, fontweight='bold')
        ax.set_xticks(k_vals)
        ax.tick_params(colors='#94a3b8')
        ax.legend(fontsize=9, facecolor='#1a2035', labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')

    plt.suptitle('Subset Selection: Model Selection Criteria', color='#e2e8f0',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # CV plot
    st.markdown("<div class='section-header'>📊 Cross-Validation Error vs Number of Predictors</div>",
                unsafe_allow_html=True)
    best_cv_k = np.argmin(cv_subset_mses) + 1

    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0f1117')
    ax.set_facecolor('#1a2035')
    ax.plot(k_vals, cv_subset_mses, 'o-', color='#f59e0b', lw=2.5, ms=9)
    ax.scatter([best_cv_k], [cv_subset_mses[best_cv_k-1]], color='#ef4444', s=250,
               zorder=5, marker='*', label=f'Optimal k={best_cv_k}')
    ax.axvline(best_cv_k, color='#ef4444', ls=':', alpha=0.7, lw=2)
    ax.set_xlabel('Number of Predictors', color='#94a3b8', fontsize=12)
    ax.set_ylabel('10-Fold CV MSE', color='#94a3b8', fontsize=12)
    ax.set_title('Best Subset Selection: CV Error vs # Predictors', color='#e2e8f0',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(k_vals)
    ax.tick_params(colors='#94a3b8')
    ax.legend(fontsize=12, facecolor='#1a2035', labelcolor='#e2e8f0')
    ax.grid(True, alpha=0.2, color='#2d3a5e')
    for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Show optimal features
    st.markdown(f"<div class='result-box'><p class='highlight'>✅ Optimal subset (k={best_cv_k} by CV): "
                f"{[feature_cols[c] for c in best_subset[best_cv_k-1]['cols']]}</p></div>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: SHRINKAGE METHODS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Shrinkage (Ridge/Lasso)":
    st.markdown("""
    <div class='hero-banner'>
        <h1>📉 Shrinkage Methods</h1>
        <p>Ridge and Lasso Regression — Bias-Variance Tradeoff and Optimal Lambda Selection</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("### ⚙️ Parameters")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        alpha_min = st.number_input("λ min (log10)", -4.0, 0.0, -3.0, 0.5)
        alpha_max_r = st.number_input("λ max for Ridge (log10)", 1.0, 7.0, 5.0, 0.5)
        alpha_max_l = st.number_input("λ max for Lasso (log10)", 0.0, 3.0, 1.0, 0.5)
        shrink_btn = st.button("▶️ Run Shrinkage Analysis", use_container_width=True)

    with col_r:
        st.markdown("### 📖 Method Overview")
        st.markdown("""<div class='result-box'>
        <p><b style='color:#3b82f6'>Ridge (L2):</b> Adds λΣβ² penalty. Shrinks all coefficients toward zero but never exactly zero. Best when all features contribute.</p>
        <p><b style='color:#ef4444'>Lasso (L1):</b> Adds λΣ|β| penalty. Produces sparse solutions — zeroes out weak predictors. Built-in feature selection.</p>
        <p><b style='color:#22c55e'>Optimal λ:</b> Found via cross-validation — minimizes test error, balancing bias and variance.</p>
        </div>""", unsafe_allow_html=True)

    if shrink_btn:
        with st.spinner("Computing Ridge & Lasso paths..."):
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
            alphas_r = np.logspace(alpha_min, alpha_max_r, 200)
            alphas_l = np.logspace(alpha_min, alpha_max_l, 200)

            r_coefs, r_train, r_test = [], [], []
            for a in alphas_r:
                m = Ridge(alpha=a).fit(X_train, y_train)
                r_coefs.append(m.coef_)
                r_train.append(mean_squared_error(y_train, m.predict(X_train)))
                r_test.append( mean_squared_error(y_test,  m.predict(X_test)))

            l_coefs, l_train, l_test = [], [], []
            for a in alphas_l:
                m = Lasso(alpha=a, max_iter=10000).fit(X_train, y_train)
                l_coefs.append(m.coef_)
                l_train.append(mean_squared_error(y_train, m.predict(X_train)))
                l_test.append( mean_squared_error(y_test,  m.predict(X_test)))

            opt_r = RidgeCV(alphas=alphas_r, cv=10).fit(X_scaled, y).alpha_
            opt_l = LassoCV(alphas=alphas_l, cv=10, max_iter=10000, random_state=42).fit(X_scaled, y).alpha_

        st.session_state['shrink_results'] = {
            'alphas_r': alphas_r, 'alphas_l': alphas_l,
            'r_coefs': np.array(r_coefs), 'l_coefs': np.array(l_coefs),
            'r_train': r_train, 'r_test': r_test,
            'l_train': l_train, 'l_test': l_test,
            'opt_r': opt_r, 'opt_l': opt_l,
        }

    if 'shrink_results' not in st.session_state:
        st.markdown("""
        <div style="background:#1a2035; border:2px dashed #2d3a5e; border-radius:16px;
                    padding:3rem; text-align:center; margin-top:1rem;">
            <div style="font-size:3rem;">▶️</div>
            <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.8rem;">
                Set your parameters and press<br>
                <b style="color:#60a5fa;">Run Shrinkage Analysis</b> to see results
            </p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    alphas_r = st.session_state['shrink_results']['alphas_r']
    alphas_l = st.session_state['shrink_results']['alphas_l']
    r_coefs  = st.session_state['shrink_results']['r_coefs']
    l_coefs  = st.session_state['shrink_results']['l_coefs']
    r_train  = st.session_state['shrink_results']['r_train']
    r_test   = st.session_state['shrink_results']['r_test']
    l_train  = st.session_state['shrink_results']['l_train']
    l_test   = st.session_state['shrink_results']['l_test']
    opt_r    = st.session_state['shrink_results']['opt_r']
    opt_l    = st.session_state['shrink_results']['opt_l']

    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='metric-card'><div class='val'>{opt_r:.4f}</div><div class='lbl'>Optimal Ridge λ (CV)</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='val'>{opt_l:.4f}</div><div class='lbl'>Optimal Lasso λ (CV)</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Coefficient Paths", "⚖️ Bias-Variance Tradeoff", "🔢 Coefficient Comparison"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f1117')
        fig.patch.set_facecolor('#0f1117')
        colors_feat = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#8b5cf6','#06b6d4','#f97316']
        for ax, coefs, alphas, opt_a, title in zip(
            axes, [r_coefs, l_coefs], [alphas_r, alphas_l], [opt_r, opt_l], ['Ridge','Lasso']):
            ax.set_facecolor('#1a2035')
            for j, feat in enumerate(feature_cols):
                ax.semilogx(alphas, coefs[:, j], lw=2, color=colors_feat[j], label=feat)
            ax.axvline(opt_a, color='white', ls='--', lw=2, label=f'λ*={opt_a:.3f}')
            ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
            ax.set_xlabel('λ (log scale)', color='#94a3b8', fontsize=11)
            ax.set_ylabel('Coefficient', color='#94a3b8', fontsize=11)
            ax.set_title(f'{title} — Coefficient Paths', color='#e2e8f0', fontsize=12, fontweight='bold')
            ax.tick_params(colors='#94a3b8')
            ax.legend(fontsize=8, facecolor='#1a2035', labelcolor='#e2e8f0', loc='upper right')
            ax.grid(True, alpha=0.2, color='#2d3a5e')
            for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f1117')
        fig.patch.set_facecolor('#0f1117')
        for ax, tr, te, alphas, opt_a, title, c1c, c2c in zip(
            axes,
            [r_train, l_train], [r_test, l_test],
            [alphas_r, alphas_l], [opt_r, opt_l],
            ['Ridge','Lasso'],
            ['#3b82f6','#ef4444'], ['#22c55e','#f59e0b']):
            ax.set_facecolor('#1a2035')
            ax.semilogx(alphas, tr, '-',  color=c1c, lw=2.5, label='Train MSE')
            ax.semilogx(alphas, te, '--', color=c2c, lw=2.5, label='Test MSE')
            ax.axvline(opt_a, color='white', ls=':', lw=2, label=f'λ*={opt_a:.3f}')
            ax.fill_between(alphas, tr, te, alpha=0.1, color='gray', label='Variance region')
            ax.set_xlabel('λ (log scale)', color='#94a3b8', fontsize=11)
            ax.set_ylabel('MSE', color='#94a3b8', fontsize=11)
            ax.set_title(f'{title} — Bias-Variance Tradeoff', color='#e2e8f0', fontsize=12, fontweight='bold')
            ax.tick_params(colors='#94a3b8')
            ax.legend(fontsize=10, facecolor='#1a2035', labelcolor='#e2e8f0')
            ax.grid(True, alpha=0.2, color='#2d3a5e')
            for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
        plt.suptitle('Bias-Variance Tradeoff: Low λ = High Variance | High λ = High Bias',
                     color='#94a3b8', fontsize=11, y=1.01)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        ols    = LinearRegression().fit(X_scaled, y)
        ridge_opt = Ridge(alpha=opt_r).fit(X_scaled, y)
        lasso_opt = Lasso(alpha=opt_l, max_iter=10000).fit(X_scaled, y)

        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0f1117')
        ax.set_facecolor('#1a2035')
        x = np.arange(len(feature_cols))
        w = 0.25
        ax.bar(x-w, ols.coef_,       w, label='OLS',   color='#3b82f6', alpha=0.85)
        ax.bar(x,   ridge_opt.coef_,  w, label='Ridge', color='#ef4444', alpha=0.85)
        ax.bar(x+w, lasso_opt.coef_,  w, label='Lasso', color='#22c55e', alpha=0.85)
        ax.axhline(0, color='white', lw=0.8, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(feature_cols, rotation=20, color='#94a3b8')
        ax.set_ylabel('Coefficient Value', color='#94a3b8', fontsize=12)
        ax.set_title('OLS vs Ridge vs Lasso Coefficients at Optimal λ',
                     color='#e2e8f0', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        ax.legend(fontsize=11, facecolor='#1a2035', labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, axis='y', color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'OLS':     np.round(ols.coef_, 4),
            'Ridge':   np.round(ridge_opt.coef_, 4),
            'Lasso':   np.round(lasso_opt.coef_, 4)
        })
        st.dataframe(coef_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: PCA & PLS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 PCA & PLS":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🔬 PCA & Partial Least Squares</h1>
        <p>Dimensionality Reduction — Principal Component Regression vs PLS vs OLS</p>
    </div>
    """, unsafe_allow_html=True)

    pca_full = PCA(n_components=7).fit(X_scaled)
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Variance plots
    col_l, col_r = st.columns(2)
    with col_l:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0f1117')
        ax.set_facecolor('#1a2035')
        clrs = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#8b5cf6','#06b6d4','#f97316']
        bars = ax.bar(range(1,8), explained*100, color=clrs, alpha=0.85, edgecolor='#0f1117', lw=0.5)
        for bar, val in zip(bars, explained*100):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', color='#e2e8f0', fontsize=8)
        ax.set_xlabel('Principal Component', color='#94a3b8', fontsize=11)
        ax.set_ylabel('Variance Explained (%)', color='#94a3b8', fontsize=11)
        ax.set_title('PCA: Variance per Component', color='#e2e8f0', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.2, axis='y', color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0f1117')
        ax.set_facecolor('#1a2035')
        ax.plot(range(1,8), cumulative*100, 'o-', color='#ef4444', lw=2.5, ms=9)
        ax.fill_between(range(1,8), cumulative*100, alpha=0.2, color='#ef4444')
        ax.axhline(95, color='#22c55e', ls='--', lw=1.5, label='95% threshold')
        ax.set_xlabel('# Components', color='#94a3b8', fontsize=11)
        ax.set_ylabel('Cumulative Variance (%)', color='#94a3b8', fontsize=11)
        ax.set_title('PCA: Cumulative Variance', color='#e2e8f0', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        ax.legend(fontsize=10, facecolor='#1a2035', labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # PCR vs PLS
    st.markdown("<div class='section-header'>📊 PCR vs PLS: MSE vs Number of Components</div>",
                unsafe_allow_html=True)

    pca_btn = st.button("▶️ Run PCR vs PLS Cross-Validation", use_container_width=False)

    if pca_btn:
        with st.spinner("Running 10-fold CV for PCR and PLS..."):
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            n_comps = range(1, 8)
            pcr_mses, pls_mses = [], []
            for nc in n_comps:
                pcr_pipe = Pipeline([('pca', PCA(n_components=nc)), ('lr', LinearRegression())])
                pcr_mses.append(-cross_val_score(pcr_pipe, X_scaled, y, cv=kf, scoring='neg_mean_squared_error').mean())
                pls_mses.append(-cross_val_score(PLSRegression(n_components=nc), X_scaled, y, cv=kf, scoring='neg_mean_squared_error').mean())
            ols_mse = -cross_val_score(LinearRegression(), X_scaled, y, cv=kf, scoring='neg_mean_squared_error').mean()
        st.session_state['pca_results'] = {
            'pcr_mses': pcr_mses, 'pls_mses': pls_mses, 'ols_mse': ols_mse
        }

    if 'pca_results' not in st.session_state:
        st.markdown("""
        <div style="background:#1a2035; border:2px dashed #2d3a5e; border-radius:16px;
                    padding:2rem; text-align:center; margin-top:0.5rem;">
            <div style="font-size:2.5rem;">▶️</div>
            <p style="color:#94a3b8; font-size:1.05rem; margin-top:0.6rem;">
                Press <b style="color:#60a5fa;">Run PCR vs PLS Cross-Validation</b> to compute results
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        pcr_mses = st.session_state['pca_results']['pcr_mses']
        pls_mses = st.session_state['pca_results']['pls_mses']
        ols_mse  = st.session_state['pca_results']['ols_mse']
        n_comps  = range(1, 8)

        best_pcr = np.argmin(pcr_mses) + 1
        best_pls = np.argmin(pls_mses) + 1

        fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0f1117')
        ax.set_facecolor('#1a2035')
        ax.plot(list(n_comps), pcr_mses, 'o-',  color='#3b82f6', lw=2.5, ms=9, label='PCR')
        ax.plot(list(n_comps), pls_mses, 's--', color='#ef4444', lw=2.5, ms=9, label='PLS')
        ax.axhline(ols_mse, color='#22c55e', ls=':', lw=2.5, label=f'OLS (MSE={ols_mse:.2f})')
        ax.scatter([best_pcr], [pcr_mses[best_pcr-1]], color='#3b82f6', s=250, zorder=6, marker='*',
                   label=f'Best PCR: {best_pcr} comp')
        ax.scatter([best_pls], [pls_mses[best_pls-1]],  color='#ef4444', s=250, zorder=6, marker='*',
                   label=f'Best PLS: {best_pls} comp')
        ax.set_xlabel('Number of Components', color='#94a3b8', fontsize=12)
        ax.set_ylabel('10-Fold CV MSE', color='#94a3b8', fontsize=12)
        ax.set_title('PCR vs PLS vs OLS: MSE vs Components', color='#e2e8f0', fontsize=13, fontweight='bold')
        ax.set_xticks(list(n_comps))
        ax.tick_params(colors='#94a3b8')
        ax.legend(fontsize=11, facecolor='#1a2035', labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Results table
        tbl = pd.DataFrame({
            '# Components': list(n_comps),
            'PCR CV MSE': [round(v,3) for v in pcr_mses],
            'PLS CV MSE': [round(v,3) for v in pls_mses],
        })
        tbl['OLS CV MSE'] = round(ols_mse, 3)
        st.dataframe(tbl.style.highlight_min(subset=['PCR CV MSE','PLS CV MSE'], color='#14532d'),
                     use_container_width=True)

    # PCA Biplot (no CV needed — always show)
    st.markdown("<div class='section-header'>🗺️ PCA Biplot (PC1 vs PC2, colored by MPG)</div>",
                unsafe_allow_html=True)
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0f1117')
    ax.set_facecolor('#1a2035')
    sc = ax.scatter(X_pca2[:,0], X_pca2[:,1], c=y, cmap='RdYlGn', alpha=0.7, s=30, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('MPG', color='#94a3b8')
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')
    scale = 3
    colors_feat = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#8b5cf6','#06b6d4','#f97316']
    for j, feat in enumerate(feature_cols):
        ax.arrow(0, 0, pca2.components_[0,j]*scale, pca2.components_[1,j]*scale,
                 head_width=0.07, head_length=0.05, fc=colors_feat[j], ec=colors_feat[j], lw=1.5)
        ax.text(pca2.components_[0,j]*scale*1.2, pca2.components_[1,j]*scale*1.2,
                feat, fontsize=10, color=colors_feat[j], fontweight='bold')
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', color='#94a3b8', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', color='#94a3b8', fontsize=12)
    ax.set_title('PCA Biplot — Feature Loadings & MPG', color='#e2e8f0', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    ax.axhline(0, color='gray', lw=0.6, alpha=0.5); ax.axvline(0, color='gray', lw=0.6, alpha=0.5)
    ax.grid(True, alpha=0.2, color='#2d3a5e')
    for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🏆 Final Model Comparison</h1>
        <p>Head-to-head comparison of all models — RMSE, R², and insights</p>
    </div>
    """, unsafe_allow_html=True)

    compare_btn = st.button("▶️ Run Model Comparison", use_container_width=False)

    if compare_btn:
        with st.spinner("Computing all models with 10-fold CV. This may take 15–30 seconds..."):
            kf = KFold(n_splits=10, shuffle=True, random_state=42)

            alphas_r = np.logspace(-3, 5, 100)
            alphas_l = np.logspace(-3, 1, 100)
            opt_r = RidgeCV(alphas=alphas_r, cv=10).fit(X_scaled, y).alpha_
            opt_l = LassoCV(alphas=alphas_l, cv=10, max_iter=10000, random_state=42).fit(X_scaled, y).alpha_

            p = X_scaled.shape[1]
            best_4_cols = None
            best_mse = np.inf
            for subset in combinations(range(p), 4):
                lr = LinearRegression()
                mse = -cross_val_score(lr, X_scaled[:, list(subset)], y, cv=kf,
                                       scoring='neg_mean_squared_error').mean()
                if mse < best_mse:
                    best_mse = mse; best_4_cols = list(subset)

            pcr_mses = [-cross_val_score(
                Pipeline([('pca', PCA(n_components=nc)), ('lr', LinearRegression())]),
                X_scaled, y, cv=kf, scoring='neg_mean_squared_error').mean()
                        for nc in range(1, 8)]
            pls_mses = [-cross_val_score(PLSRegression(n_components=nc), X_scaled, y, cv=kf,
                                          scoring='neg_mean_squared_error').mean()
                        for nc in range(1, 8)]
            best_pcr_nc = np.argmin(pcr_mses) + 1
            best_pls_nc = np.argmin(pls_mses) + 1

            models_cfg = [
                ('OLS (all features)',        LinearRegression(),                                                            X_scaled),
                ('OLS (best 4 features)',     LinearRegression(),                                                            X_scaled[:, best_4_cols]),
                (f'Ridge (λ={opt_r:.3f})',    Ridge(alpha=opt_r),                                                           X_scaled),
                (f'Lasso (λ={opt_l:.3f})',    Lasso(alpha=opt_l, max_iter=10000),                                           X_scaled),
                (f'PCR ({best_pcr_nc} comp)', Pipeline([('pca', PCA(n_components=best_pcr_nc)), ('lr', LinearRegression())]), X_scaled),
                (f'PLS ({best_pls_nc} comp)', PLSRegression(n_components=best_pls_nc),                                       X_scaled),
            ]

            summary = []
            for name, model, X_use in models_cfg:
                mse = -cross_val_score(model, X_use, y, cv=kf, scoring='neg_mean_squared_error').mean()
                r2  =  cross_val_score(model, X_use, y, cv=kf, scoring='r2').mean()
                summary.append({'Model': name, 'CV MSE': round(mse, 3),
                                 'CV RMSE': round(np.sqrt(mse), 3), 'CV R²': round(r2, 4)})

        st.session_state['compare_results'] = {
            'summary': summary, 'models_cfg': models_cfg
        }

    if 'compare_results' not in st.session_state:
        st.markdown("""
        <div style="background:#1a2035; border:2px dashed #2d3a5e; border-radius:16px;
                    padding:3rem; text-align:center; margin-top:1rem;">
            <div style="font-size:3rem;">▶️</div>
            <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.8rem;">
                Press <b style="color:#60a5fa;">Run Model Comparison</b> to evaluate all models
            </p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    summary     = st.session_state['compare_results']['summary']
    models_cfg  = st.session_state['compare_results']['models_cfg']
    summary_df  = pd.DataFrame(summary)
    best_model  = summary_df.loc[summary_df['CV RMSE'].idxmin(), 'Model']

    # KPI row
    cols = st.columns(3)
    cols[0].markdown(f"<div class='metric-card'><div class='val'>{summary_df['CV RMSE'].min():.3f}</div><div class='lbl'>Best RMSE</div></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='metric-card'><div class='val'>{summary_df['CV R²'].max():.4f}</div><div class='lbl'>Best R²</div></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='metric-card'><div class='val'>6</div><div class='lbl'>Models Compared</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='result-box'><p class='highlight'>🏆 Best model: {best_model}</p></div>",
                unsafe_allow_html=True)

    # Bar charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f1117')
    fig.patch.set_facecolor('#0f1117')
    names = [r['Model'] for r in summary]
    clrs  = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#8b5cf6','#06b6d4']

    for ax, metric, label in zip(axes, ['CV RMSE','CV R²'], ['CV RMSE','CV R²']):
        ax.set_facecolor('#1a2035')
        vals = [r[metric] for r in summary]
        bars = ax.barh(names, vals, color=clrs, alpha=0.85, edgecolor='#0f1117', lw=0.5)
        for bar, val in zip(bars, vals):
            ax.text(val * (1.01 if metric == 'CV RMSE' else 1.001),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', color='#e2e8f0', fontsize=9)
        ax.set_xlabel(label, color='#94a3b8', fontsize=12)
        ax.set_title(f'Model Comparison: {label}', color='#e2e8f0', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.2, axis='x', color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Summary table
    st.markdown("<div class='section-header'>📋 Full Results Table</div>", unsafe_allow_html=True)
    st.dataframe(
        summary_df.style
            .highlight_min(subset=['CV MSE','CV RMSE'], color='#14532d')
            .highlight_max(subset=['CV R²'], color='#14532d'),
        use_container_width=True
    )

    # Actual vs Predicted for best model
    st.markdown("<div class='section-header'>🎯 Best Model: Actual vs Predicted</div>",
                unsafe_allow_html=True)
    best_cfg = models_cfg[summary_df['CV RMSE'].idxmin()]
    _, X_t, _, y_t = train_test_split(best_cfg[2], y, test_size=0.2, random_state=42)
    best_cfg[1].fit(best_cfg[2][:int(len(y)*0.8)], y[:int(len(y)*0.8)])
    preds = best_cfg[1].predict(X_t)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0f1117')
    ax.set_facecolor('#1a2035')
    ax.scatter(y_t, preds, alpha=0.6, color='#3b82f6', s=35, edgecolors='none')
    mn, mx = min(y_t.min(), preds.min()), max(y_t.max(), preds.max())
    ax.plot([mn,mx],[mn,mx], 'w--', lw=2, alpha=0.7, label='Perfect prediction')
    ax.set_xlabel('Actual MPG', color='#94a3b8', fontsize=12)
    ax.set_ylabel('Predicted MPG', color='#94a3b8', fontsize=12)
    ax.set_title(f'{best_cfg[0]}: Actual vs Predicted', color='#e2e8f0', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    ax.legend(fontsize=10, facecolor='#1a2035', labelcolor='#e2e8f0')
    ax.grid(True, alpha=0.2, color='#2d3a5e')
    for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7: LIVE PREDICTION LAB
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Live Prediction Lab":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🧪 Live Prediction Lab</h1>
        <p>Set your car's specifications, choose an algorithm, tune its parameters — and predict MPG in real time</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Helpers ──────────────────────────────────────────────────────────────
    @st.cache_data
    def get_trained_models(X_data, y_data):
        """Train all models on full dataset and return fitted objects + metadata."""
        alphas_r = np.logspace(-3, 5, 150)
        alphas_l = np.logspace(-3, 1, 150)
        opt_r = RidgeCV(alphas=alphas_r, cv=10).fit(X_data, y_data).alpha_
        opt_l = LassoCV(alphas=alphas_l, cv=10, max_iter=10000, random_state=42).fit(X_data, y_data).alpha_

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        models_info = {}

        # OLS
        m = LinearRegression().fit(X_data, y_data)
        cv_mse = -cross_val_score(LinearRegression(), X_data, y_data, cv=kf,
                                   scoring='neg_mean_squared_error').mean()
        models_info['OLS (Linear Regression)'] = {
            'model': m, 'cv_rmse': np.sqrt(cv_mse),
            'cv_r2': cross_val_score(LinearRegression(), X_data, y_data, cv=kf, scoring='r2').mean(),
            'type': 'ols', 'uses_all_features': True,
            'description': 'Ordinary Least Squares — fits coefficients by minimizing residual sum of squares. No regularization.',
            'coefs': dict(zip(feature_cols, m.coef_)), 'intercept': m.intercept_
        }

        # Ridge
        m = Ridge(alpha=opt_r).fit(X_data, y_data)
        cv_mse = -cross_val_score(Ridge(alpha=opt_r), X_data, y_data, cv=kf,
                                   scoring='neg_mean_squared_error').mean()
        models_info[f'Ridge (λ={opt_r:.3f})'] = {
            'model': m, 'cv_rmse': np.sqrt(cv_mse),
            'cv_r2': cross_val_score(Ridge(alpha=opt_r), X_data, y_data, cv=kf, scoring='r2').mean(),
            'type': 'ridge', 'uses_all_features': True,
            'description': f'Ridge Regression (L2) with optimal λ={opt_r:.4f} found by 10-fold CV. Shrinks all coefficients.',
            'coefs': dict(zip(feature_cols, m.coef_)), 'intercept': m.intercept_,
            'lambda': opt_r
        }

        # Lasso
        m = Lasso(alpha=opt_l, max_iter=10000).fit(X_data, y_data)
        active = [f for f, c in zip(feature_cols, m.coef_) if abs(c) > 1e-6]
        cv_mse = -cross_val_score(Lasso(alpha=opt_l, max_iter=10000), X_data, y_data, cv=kf,
                                   scoring='neg_mean_squared_error').mean()
        models_info[f'Lasso (λ={opt_l:.3f})'] = {
            'model': m, 'cv_rmse': np.sqrt(cv_mse),
            'cv_r2': cross_val_score(Lasso(alpha=opt_l, max_iter=10000), X_data, y_data, cv=kf, scoring='r2').mean(),
            'type': 'lasso', 'uses_all_features': True,
            'description': f'Lasso Regression (L1) with optimal λ={opt_l:.4f}. Zeroes out weak predictors → active: {active}.',
            'coefs': dict(zip(feature_cols, m.coef_)), 'intercept': m.intercept_,
            'lambda': opt_l, 'active_features': active
        }

        # PCR
        best_nc_pcr = 1
        best_mse_pcr = np.inf
        for nc in range(1, 8):
            pipe = Pipeline([('pca', PCA(n_components=nc)), ('lr', LinearRegression())])
            mse = -cross_val_score(pipe, X_data, y_data, cv=kf, scoring='neg_mean_squared_error').mean()
            if mse < best_mse_pcr:
                best_mse_pcr = mse; best_nc_pcr = nc
        pipe_pcr = Pipeline([('pca', PCA(n_components=best_nc_pcr)), ('lr', LinearRegression())])
        pipe_pcr.fit(X_data, y_data)
        models_info[f'PCR ({best_nc_pcr} components)'] = {
            'model': pipe_pcr, 'cv_rmse': np.sqrt(best_mse_pcr),
            'cv_r2': cross_val_score(pipe_pcr, X_data, y_data, cv=kf, scoring='r2').mean(),
            'type': 'pcr', 'uses_all_features': True,
            'description': f'Principal Component Regression with {best_nc_pcr} PCs (chosen by CV). Unsupervised dimensionality reduction.',
            'n_components': best_nc_pcr
        }

        # PLS
        best_nc_pls = 1
        best_mse_pls = np.inf
        for nc in range(1, 8):
            pls = PLSRegression(n_components=nc)
            mse = -cross_val_score(pls, X_data, y_data, cv=kf, scoring='neg_mean_squared_error').mean()
            if mse < best_mse_pls:
                best_mse_pls = mse; best_nc_pls = nc
        pls_m = PLSRegression(n_components=best_nc_pls).fit(X_data, y_data)
        models_info[f'PLS ({best_nc_pls} components)'] = {
            'model': pls_m, 'cv_rmse': np.sqrt(best_mse_pls),
            'cv_r2': cross_val_score(pls_m, X_data, y_data, cv=kf, scoring='r2').mean(),
            'type': 'pls', 'uses_all_features': True,
            'description': f'Partial Least Squares with {best_nc_pls} components (chosen by CV). Supervised dimensionality reduction.',
            'n_components': best_nc_pls
        }

        return models_info, opt_r, opt_l

    with st.spinner("Training all models on full dataset..."):
        trained_models, opt_r_global, opt_l_global = get_trained_models(X_scaled, y)

    # ── Layout ───────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        # ── Algorithm Selector ────────────────────────────────────────────
        st.markdown("<div class='section-header'>🤖 Choose Algorithm</div>", unsafe_allow_html=True)
        algo_name = st.selectbox("Algorithm", list(trained_models.keys()), label_visibility="collapsed")
        algo_info = trained_models[algo_name]

        st.markdown(f"""
        <div class='result-box'>
            <p style='color:#60a5fa; font-weight:600; font-size:0.95rem;'>ℹ️ {algo_name}</p>
            <p>{algo_info['description']}</p>
        </div>""", unsafe_allow_html=True)

        # Model performance badges
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='metric-card'><div class='val'>{algo_info['cv_rmse']:.3f}</div><div class='lbl'>CV RMSE (mpg)</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='val'>{algo_info['cv_r2']:.4f}</div><div class='lbl'>CV R²</div></div>", unsafe_allow_html=True)

        # ── Custom Lambda Tuning for Ridge/Lasso ──────────────────────────
        custom_model = None
        if algo_info['type'] in ('ridge', 'lasso'):
            st.markdown("<div class='section-header'>⚙️ Tune Regularization</div>", unsafe_allow_html=True)
            use_custom_lambda = st.toggle("Override optimal λ", value=False)
            if use_custom_lambda:
                if algo_info['type'] == 'ridge':
                    custom_lambda = st.slider("Ridge λ", 0.001, 5000.0,
                                              float(algo_info['lambda']), step=0.5, format="%.3f")
                    custom_model = Ridge(alpha=custom_lambda).fit(X_scaled, y)
                else:
                    custom_lambda = st.slider("Lasso λ", 0.001, 5.0,
                                              float(algo_info['lambda']), step=0.01, format="%.3f")
                    custom_model = Lasso(alpha=custom_lambda, max_iter=10000).fit(X_scaled, y)
                st.caption(f"Optimal λ (CV) = {algo_info['lambda']:.4f}")

        # ── Car Specs Input ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>🚗 Car Specifications</div>", unsafe_allow_html=True)

        # Preset example cars
        presets = {
            "Custom": None,
            "🇺🇸 70s V8 Muscle":   [8, 350, 165, 3693, 11.5, 70, 1],
            "🇯🇵 80s Economy":      [4,  97,  75, 2265, 18.2, 82, 3],
            "🇪🇺 Euro Compact":     [4, 121,  97, 2300, 14.7, 78, 2],
            "🇺🇸 6-cyl Sedan":      [6, 250,  88, 3139, 14.5, 76, 1],
            "🇯🇵 4-cyl Efficient":  [4,  85,  65, 1975, 19.4, 81, 3],
        }
        preset = st.selectbox("Load a preset car", list(presets.keys()))

        pv = presets[preset]  # None or list of values

        def _v(idx, default, lo, hi, label, step=1, fmt=None):
            val = float(pv[idx]) if pv else default
            kw = dict(min_value=float(lo), max_value=float(hi), value=float(val), step=float(step))
            if fmt: kw['format'] = fmt
            return st.slider(label, **kw)

        cylinders    = _v(0, 4,    3,   8,   "🔧 Cylinders",          1)
        displacement = _v(1, 150,  68,  455, "💨 Displacement (cu.in)", 5.0)
        horsepower   = _v(2, 100,  46,  230, "⚡ Horsepower",          5.0)
        weight       = _v(3, 2800, 1600, 5200, "⚖️ Weight (lbs)",      50.0)
        acceleration = _v(4, 15.0, 8.0, 25.0, "🏁 Acceleration (0-60s)", 0.5, "%.1f")
        model_year   = int(_v(5, 76, 70, 82, "📅 Model Year (mod 100)", 1))
        origin       = int(st.select_slider("🌍 Origin",
                                            options=[1, 2, 3],
                                            value=int(pv[6]) if pv else 1,
                                            format_func=lambda x: {1:"🇺🇸 USA", 2:"🇪🇺 Europe", 3:"🇯🇵 Japan"}[x]))

        predict_btn = st.button("Predict MPG", use_container_width=True)

    # ── Trigger: store results in session state only on button press ─────
    if predict_btn:
        raw_input    = np.array([[cylinders, displacement, horsepower, weight,
                                   acceleration, model_year, origin]])
        input_scaled = scaler.transform(raw_input)
        model_to_use = custom_model if custom_model is not None else algo_info['model']
        prediction   = float(model_to_use.predict(input_scaled).ravel()[0])
        st.session_state['pred_result'] = {
            'prediction':   max(5.0, min(55.0, prediction)),
            'input_scaled': input_scaled,
            'raw_input':    [cylinders, displacement, horsepower, weight,
                             acceleration, model_year, origin],
            'algo_name':    algo_name,
            'algo_info':    algo_info,
            'custom_model': custom_model,
        }

    # ── Right Column: Results ─────────────────────────────────────────────
    with right_col:
        if 'pred_result' not in st.session_state:
            st.markdown("""
            <div style="background:#1a2035; border:2px dashed #2d3a5e; border-radius:16px;
                        padding:3rem; text-align:center; margin-top:1rem;">
                <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.8rem;">
                    Set your car specs and press<br>
                    <b style="color:#60a5fa;">Predict MPG</b> to see results
                </p>
            </div>""", unsafe_allow_html=True)
            st.stop()

        res          = st.session_state['pred_result']
        prediction   = res['prediction']
        input_scaled = res['input_scaled']
        algo_name    = res['algo_name']
        algo_info    = res['algo_info']
        custom_model = res['custom_model']
        raw_vals     = res['raw_input']
        cylinders, displacement, horsepower, weight, acceleration, model_year, origin = raw_vals

        # MPG gauge-style display
        mpg_pct = (prediction - 5) / (50) * 100
        if prediction < 18:
            gauge_color, eff_label, eff_emoji = "#ef4444", "Low Efficiency", "🔴"
        elif prediction < 28:
            gauge_color, eff_label, eff_emoji = "#f59e0b", "Moderate Efficiency", "🟡"
        elif prediction < 35:
            gauge_color, eff_label, eff_emoji = "#22c55e", "Good Efficiency", "🟢"
        else:
            gauge_color, eff_label, eff_emoji = "#06b6d4", "Excellent Efficiency", "🔵"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a2035, #16213e);
                    border: 2px solid {gauge_color}33;
                    border-radius: 16px; padding: 2rem; text-align: center; margin-bottom: 1rem;">
            <p style="color:#94a3b8; font-size:0.9rem; margin:0 0 0.3rem 0;">PREDICTED FUEL EFFICIENCY</p>
            <div style="font-size: 5rem; font-weight: 800; color: {gauge_color};
                        text-shadow: 0 0 30px {gauge_color}66; line-height:1.1;">
                {prediction:.1f}
            </div>
            <div style="color:#e2e8f0; font-size:1.3rem; font-weight:600;">miles per gallon</div>
            <div style="margin-top:0.8rem;">
                <span style="background:{gauge_color}22; color:{gauge_color}; border:1px solid {gauge_color}55;
                             border-radius:20px; padding:4px 16px; font-size:0.9rem; font-weight:600;">
                    {eff_emoji} {eff_label}
                </span>
            </div>
            <div style="margin-top:1.2rem; background:#0f1117; border-radius:8px; height:14px; overflow:hidden;">
                <div style="width:{mpg_pct:.0f}%; height:100%;
                            background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e, #06b6d4);
                            border-radius:8px; transition: width 0.5s ease;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; color:#4b5563; font-size:0.75rem; margin-top:3px;">
                <span>5 mpg</span><span>55 mpg</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── All Models Comparison ────────────────────────────────────────
        st.markdown("<div class='section-header'>📊 All Models — Side-by-Side Prediction</div>",
                    unsafe_allow_html=True)

        all_preds = {}
        for mname, minfo in trained_models.items():
            try:
                p = float(minfo['model'].predict(input_scaled).ravel()[0])
                all_preds[mname] = max(5.0, min(55.0, p))
            except Exception:
                all_preds[mname] = None

        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f1117')
        ax.set_facecolor('#1a2035')
        names_p = list(all_preds.keys())
        vals_p  = [all_preds[n] for n in names_p]
        clrs_p  = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#8b5cf6'][:len(names_p)]
        bars = ax.barh(names_p, vals_p, color=clrs_p, alpha=0.85, edgecolor='#0f1117', height=0.55)
        for bar, val in zip(bars, vals_p):
            ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f} mpg', va='center', color='#e2e8f0', fontsize=10, fontweight='600')
        ax.axvline(df['mpg'].mean(), color='white', ls='--', lw=1.5, alpha=0.5,
                   label=f'Dataset mean ({df["mpg"].mean():.1f} mpg)')
        ax.set_xlabel('Predicted MPG', color='#94a3b8', fontsize=11)
        ax.set_title('All Algorithms — Predicted MPG for This Car', color='#e2e8f0',
                     fontsize=12, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        ax.legend(fontsize=9, facecolor='#1a2035', labelcolor='#e2e8f0')
        ax.set_xlim(0, max(vals_p) * 1.2)
        ax.grid(True, alpha=0.2, axis='x', color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Coefficient / Feature Impact Table ───────────────────────────
        if algo_info['type'] in ('ols', 'ridge', 'lasso'):
            st.markdown("<div class='section-header'>🔢 Feature Contributions to Prediction</div>",
                        unsafe_allow_html=True)

            m_use = custom_model if custom_model is not None else algo_info['model']
            coefs  = m_use.coef_
            intercept = float(m_use.intercept_)
            contrib = coefs * input_scaled[0]

            contrib_df = pd.DataFrame({
                'Feature':       feature_cols,
                'Input Value':   [cylinders, displacement, horsepower, weight,
                                   acceleration, model_year, origin],
                'Scaled Value':  np.round(input_scaled[0], 3),
                'Coefficient':   np.round(coefs, 4),
                'Contribution':  np.round(contrib, 4),
            }).sort_values('Contribution', key=abs, ascending=False)

            # Contribution bar chart
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f1117')
            ax.set_facecolor('#1a2035')
            c_vals  = contrib_df['Contribution'].values
            c_names = contrib_df['Feature'].values
            c_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in c_vals]
            bars = ax.barh(c_names, c_vals, color=c_colors, alpha=0.85, edgecolor='#0f1117', height=0.55)
            for bar, val in zip(bars, c_vals):
                xpos = val + 0.05 if val >= 0 else val - 0.05
                ha   = 'left' if val >= 0 else 'right'
                ax.text(xpos, bar.get_y() + bar.get_height()/2,
                        f'{val:+.3f}', va='center', ha=ha, color='#e2e8f0', fontsize=9)
            ax.axvline(0, color='white', lw=1, alpha=0.6)
            ax.set_xlabel('Contribution to MPG Prediction', color='#94a3b8', fontsize=11)
            ax.set_title(f'Feature Contributions — {algo_name}', color='#e2e8f0',
                         fontsize=12, fontweight='bold')
            ax.tick_params(colors='#94a3b8')
            ax.grid(True, alpha=0.2, axis='x', color='#2d3a5e')
            for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Numeric table
            st.dataframe(contrib_df.reset_index(drop=True), use_container_width=True)
            st.markdown(f"""
            <div class='result-box'>
                <p>Intercept: <span class='highlight'>{intercept:.4f}</span></p>
                <p>Sum of contributions: <span class='highlight'>{contrib.sum():.4f}</span></p>
                <p>Final prediction = intercept + contributions = 
                   <span class='highlight'>{intercept + contrib.sum():.4f} mpg</span></p>
            </div>""", unsafe_allow_html=True)

        elif algo_info['type'] in ('pcr', 'pls'):
            st.markdown("<div class='section-header'>🔬 Projection to Component Space</div>",
                        unsafe_allow_html=True)
            nc = algo_info['n_components']
            if algo_info['type'] == 'pcr':
                pca_step = algo_info['model'].named_steps['pca']
                X_proj   = pca_step.transform(input_scaled)[0]
                exp_var  = pca_step.explained_variance_ratio_
                comp_labels = [f'PC{i+1} ({exp_var[i]*100:.1f}%)' for i in range(nc)]
            else:
                X_proj_full = algo_info['model'].transform(input_scaled)[0]
                X_proj = X_proj_full[:nc]
                comp_labels = [f'LV{i+1}' for i in range(nc)]

            fig, ax = plt.subplots(figsize=(8, 3.5), facecolor='#0f1117')
            ax.set_facecolor('#1a2035')
            clrs_c = ['#3b82f6','#ef4444','#22c55e','#f59e0b','#8b5cf6','#06b6d4','#f97316'][:nc]
            ax.bar(comp_labels, X_proj, color=clrs_c, alpha=0.85, edgecolor='#0f1117')
            ax.axhline(0, color='white', lw=0.8, alpha=0.5)
            ax.set_ylabel('Score', color='#94a3b8', fontsize=11)
            ax.set_title(f'Input projected onto {nc} components', color='#e2e8f0',
                         fontsize=12, fontweight='bold')
            ax.tick_params(colors='#94a3b8', rotation=15)
            ax.grid(True, alpha=0.2, axis='y', color='#2d3a5e')
            for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Where does this car sit in the dataset? ───────────────────────
        st.markdown("<div class='section-header'>📍 How Does This Car Compare to the Dataset?</div>",
                    unsafe_allow_html=True)

        feat_compare = st.selectbox("Compare along feature", feature_cols + ['mpg'],
                                    index=3, key='compare_feat')
        input_feat_vals = dict(zip(feature_cols, [cylinders, displacement, horsepower,
                                                    weight, acceleration, model_year, origin]))
        if feat_compare == 'mpg':
            compare_val = prediction
            dataset_vals = df['mpg'].values
        else:
            compare_val  = input_feat_vals[feat_compare]
            dataset_vals = df[feat_compare].values

        pct = float(np.mean(dataset_vals <= compare_val) * 100)

        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#0f1117')
        ax.set_facecolor('#1a2035')
        ax.hist(dataset_vals, bins=35, color='#3b82f6', alpha=0.6, edgecolor='#0f1117', label='Dataset')
        ax.axvline(compare_val, color='#f59e0b', lw=3, ls='--',
                   label=f'Your car: {compare_val:.1f}  (p{pct:.0f})')
        ax.fill_between([dataset_vals.min(), compare_val],
                        0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 30,
                        alpha=0.07, color='#f59e0b')
        ax.set_xlabel(feat_compare, color='#94a3b8', fontsize=11)
        ax.set_ylabel('Count', color='#94a3b8', fontsize=11)
        ax.set_title(f'Your car vs dataset distribution — {feat_compare}',
                     color='#e2e8f0', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        ax.legend(fontsize=10, facecolor='#1a2035', labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, axis='y', color='#2d3a5e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3a5e')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class='result-box'>
            <p>Your car's <b style='color:#60a5fa'>{feat_compare}</b> value of 
               <span class='highlight'>{compare_val:.2f}</span> is higher than 
               <span class='highlight'>{pct:.1f}%</span> of cars in the dataset.</p>
        </div>""", unsafe_allow_html=True)