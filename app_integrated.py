import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================
# KONFIGURASI HALAMAN
# ============================
st.set_page_config(
    page_title="Dashboard Predictive Maintenance - Clustering & Classification",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS - DARK THEME
# ============================
st.markdown("""
<style>
    /* Force dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Main header with gradient */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sub headers - readable on dark */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Metric cards - dark background with light text */
    .metric-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #fafafa;
    }
    
    .metric-card h3 {
        color: #667eea !important;
        margin: 0;
    }
    
    .metric-card h1 {
        color: #fafafa !important;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        color: #b0b0b0 !important;
        margin: 0;
    }
    
    /* Success box - dark green */
    .success-box {
        padding: 1rem;
        background-color: #1a3d2e;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
        color: #d4edda;
    }
    
    .success-box h4 {
        color: #4caf50 !important;
    }
    
    /* Warning box - dark yellow */
    .warning-box {
        padding: 1rem;
        background-color: #3d3420;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
        color: #fff3cd;
    }
    
    .warning-box h4 {
        color: #ffc107 !important;
    }
    
    /* Danger box - dark red */
    .danger-box {
        padding: 1rem;
        background-color: #3d2022;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        margin: 1rem 0;
        color: #f8d7da;
    }
    
    .danger-box h4 {
        color: #ff6b6b !important;
    }
    
    /* Info box - dark blue */
    .info-box {
        padding: 1rem;
        background-color: #1a2d3d;
        border-left: 4px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
        color: #d1ecf1;
    }
    
    .info-box strong {
        color: #17a2b8 !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #764ba2;
        border: none;
    }
    
    /* Ensure all text is readable */
    p, li, span, div {
        color: #e0e0e0;
    }
    
    strong {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# FUNGSI LOAD MODEL
# ============================
@st.cache_resource
def load_models():
    """Load semua model yang diperlukan"""
    try:
        # Load Clustering Models
        kmeans = joblib.load('kmeans_model.pkl')
        scaler_cluster = joblib.load('scaler.pkl')
        
        # Load Classification Models
        xgb_model = joblib.load('xgb_classification_model.pkl')
        scaler_classifier = joblib.load('classifier_scaler.pkl')
        label_encoder = joblib.load('label_encoder_failure_type.pkl')
        
        return {
            'kmeans': kmeans,
            'scaler_cluster': scaler_cluster,
            'xgb_model': xgb_model,
            'scaler_classifier': scaler_classifier,
            'label_encoder': label_encoder
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load dataset"""
    try:
        df_cluster = pd.read_csv('clustered_data.csv')
        df_classification = pd.read_csv('classification_training_data.csv')
        return df_cluster, df_classification
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

# ============================
# FUNGSI HELPER UNTUK PLOTLY DARK THEME
# ============================
def get_dark_plotly_layout():
    """Return standardized dark theme layout for plotly charts"""
    return dict(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#e0e0e0', size=12),
        xaxis=dict(
            gridcolor='#2d2d2d',
            linecolor='#2d2d2d',
            title_font=dict(size=14, color='#e0e0e0')
        ),
        yaxis=dict(
            gridcolor='#2d2d2d',
            linecolor='#2d2d2d',
            title_font=dict(size=14, color='#e0e0e0')
        ),
        legend=dict(
            bgcolor='rgba(30, 30, 30, 0.8)',
            bordercolor='#2d2d2d',
            borderwidth=1,
            font=dict(color='#e0e0e0')
        )
    )

def get_dark_plotly_3d_layout():
    """Return dark theme layout for 3D plotly charts"""
    return dict(
        scene=dict(
            bgcolor='#0e1117',
            xaxis=dict(
                backgroundcolor='#1e1e1e',
                gridcolor='#2d2d2d',
                showbackground=True,
                title_font=dict(size=14, color='#e0e0e0'),
                tickfont=dict(color='#e0e0e0')
            ),
            yaxis=dict(
                backgroundcolor='#1e1e1e',
                gridcolor='#2d2d2d',
                showbackground=True,
                title_font=dict(size=14, color='#e0e0e0'),
                tickfont=dict(color='#e0e0e0')
            ),
            zaxis=dict(
                backgroundcolor='#1e1e1e',
                gridcolor='#2d2d2d',
                showbackground=True,
                title_font=dict(size=14, color='#e0e0e0'),
                tickfont=dict(color='#e0e0e0')
            )
        ),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#e0e0e0'),
        legend=dict(
            bgcolor='rgba(30, 30, 30, 0.9)',
            bordercolor='#2d2d2d',
            borderwidth=1,
            font=dict(color='#e0e0e0', size=12),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

# ============================
# LOAD MODELS & DATA
# ============================
models = load_models()
df_cluster, df_classification = load_data()

if models is None or df_cluster is None:
    st.stop()

# Extract models
kmeans = models['kmeans']
scaler_cluster = models['scaler_cluster']
xgb_model = models['xgb_model']
scaler_classifier = models['scaler_classifier']
label_encoder = models['label_encoder']

# ============================
# SIDEBAR
# ============================
st.sidebar.markdown("# ⚙️ Predictive Maintenance")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "📊 Menu Navigasi",
    [
        "🏠 Overview",
        "🔵 Clustering Analysis",
        "🔴 Failure Classification",
        "🔗 Integrated Analysis",
        "📊 3D Visualization",
        "🎯 Prediction Tool"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Developer Team")
st.sidebar.markdown("""
**Kelompok 9 (LC41)**
- Louis Hartanto Utomo
- Raymond Christopher Sofian
- Gelfand Hanli Lim
- Karlina Gunawan
""")

# ============================
# HALAMAN: OVERVIEW
# ============================
if menu == "🏠 Overview":
    st.markdown('<p class="main-header">🏠 Dashboard Overview - Predictive Maintenance</p>', unsafe_allow_html=True)
    
    st.markdown("### 📈 Dataset Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Total Data</h3>
            <h1>{}</h1>
            <p>Data Points</p>
        </div>
        """.format(len(df_cluster)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔵 Clusters</h3>
            <h1>{}</h1>
            <p>K-Means Clusters</p>
        </div>
        """.format(df_cluster['Cluster'].nunique()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔴 Failure Types</h3>
            <h1>{}</h1>
            <p>Classification Classes</p>
        </div>
        """.format(len(label_encoder.classes_)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📋 Features</h3>
            <h1>5</h1>
            <p>Input Variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Distribusi Cluster
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">🔵 Cluster Distribution</p>', unsafe_allow_html=True)
        cluster_dist = df_cluster['Cluster'].value_counts().sort_index()
        
        fig = px.bar(
            x=cluster_dist.index,
            y=cluster_dist.values,
            labels={'x': 'Cluster', 'y': 'Count'},
            color=cluster_dist.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            **get_dark_plotly_layout(),
            showlegend=False,
            height=400,
            xaxis_title="Cluster",
            yaxis_title="Number of Data Points"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">🔴 Failure Type Distribution</p>', unsafe_allow_html=True)
        
        # Info box
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Note:</strong> Failure type distribution akan ditampilkan setelah prediksi dilakukan pada dataset
        </div>
        """, unsafe_allow_html=True)
        
        # Show failure types
        st.markdown("**Available Failure Types:**")
        for i, failure_type in enumerate(label_encoder.classes_, 1):
            icon = "✅" if failure_type == "No Failure" else "⚠️"
            st.markdown(f"{icon} **{i}.** {failure_type}")
    
    # Feature Statistics
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 Feature Statistics</p>', unsafe_allow_html=True)
    
    features_cluster = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
    
    stats_df = df_cluster[features_cluster].describe().T
    stats_df = stats_df.round(2)
    
    st.dataframe(stats_df, use_container_width=True)

# ============================
# HALAMAN: CLUSTERING ANALYSIS
# ============================
elif menu == "🔵 Clustering Analysis":
    st.markdown('<p class="main-header">🔵 Clustering Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Cluster Selection")
    selected_cluster = st.selectbox(
        "Pilih cluster untuk analisis detail:",
        options=sorted(df_cluster['Cluster'].unique()),
        format_func=lambda x: f"Cluster {x}"
    )
    
    # Filter data untuk cluster terpilih
    cluster_data = df_cluster[df_cluster['Cluster'] == selected_cluster]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Data Points", len(cluster_data))
    
    with col2:
        percentage = (len(cluster_data) / len(df_cluster)) * 100
        st.metric("📈 Percentage", f"{percentage:.2f}%")
    
    with col3:
        features_cluster = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
        avg_torque = cluster_data['Torque [Nm]'].mean()
        st.metric("⚡ Avg Torque", f"{avg_torque:.2f} Nm")
    
    with col4:
        avg_temp = cluster_data['Process temperature [K]'].mean()
        st.metric("🌡️ Avg Temperature", f"{avg_temp:.2f} K")
    
    st.markdown("---")
    
    # Statistics Table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">📊 Cluster Statistics</p>', unsafe_allow_html=True)
        cluster_stats = cluster_data[features_cluster].describe().T
        cluster_stats = cluster_stats.round(2)
        st.dataframe(cluster_stats, use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">📈 Feature Distribution</p>', unsafe_allow_html=True)
        
        feature_to_plot = st.selectbox(
            "Select feature:",
            features_cluster
        )
        
        fig = px.histogram(
            cluster_data,
            x=feature_to_plot,
            nbins=30,
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            **get_dark_plotly_layout(),
            showlegend=False,
            height=300,
            xaxis_title=feature_to_plot,
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison with overall data
    st.markdown("---")
    st.markdown('<p class="sub-header">🔄 Comparison with Overall Data</p>', unsafe_allow_html=True)
    
    comparison_data = []
    for feature in features_cluster:
        comparison_data.append({
            'Feature': feature,
            f'Cluster {selected_cluster} Mean': cluster_data[feature].mean(),
            'Overall Mean': df_cluster[feature].mean(),
            'Difference': cluster_data[feature].mean() - df_cluster[feature].mean()
        })
    
    comparison_df = pd.DataFrame(comparison_data).round(2)
    st.dataframe(comparison_df, use_container_width=True)

# ============================
# HALAMAN: FAILURE CLASSIFICATION
# ============================
elif menu == "🔴 Failure Classification":
    st.markdown('<p class="main-header">🔴 Failure Type Classification</p>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Classification Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 Model Type</h4>
            <h3>XGBoost</h3>
            <p>Classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Classes</h4>
            <h3>{}</h3>
            <p>Failure Types</p>
        </div>
        """.format(len(label_encoder.classes_)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Features</h4>
            <h3>5</h3>
            <p>Input Variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Failure Types Info
    st.markdown('<p class="sub-header">🔍 Failure Types Information</p>', unsafe_allow_html=True)
    
    failure_info = {
        'Heat Dissipation Failure': {
            'icon': '🔥',
            'severity': 'High',
            'description': 'Kegagalan sistem pendinginan menyebabkan overheat',
            'action': 'Periksa sistem cooling, bersihkan heat sink'
        },
        'No Failure': {
            'icon': '✅',
            'severity': 'None',
            'description': 'Mesin beroperasi normal tanpa masalah',
            'action': 'Lanjutkan monitoring rutin'
        },
        'Overstrain Failure': {
            'icon': '💪',
            'severity': 'High',
            'description': 'Beban kerja melebihi kapasitas mesin',
            'action': 'Kurangi beban, periksa torque limit'
        },
        'Power Failure': {
            'icon': '⚡',
            'severity': 'Critical',
            'description': 'Masalah pada sistem kelistrikan mesin',
            'action': 'Periksa power supply dan electrical system'
        },
        'Random Failures': {
            'icon': '🎲',
            'severity': 'Medium',
            'description': 'Kegagalan acak yang tidak terprediksi',
            'action': 'Analisis log dan pattern lebih detail'
        },
        'Tool Wear Failure': {
            'icon': '🔧',
            'severity': 'Medium',
            'description': 'Alat sudah aus dan perlu diganti',
            'action': 'Ganti tool, lakukan preventive maintenance'
        }
    }
    
    cols = st.columns(2)
    for idx, (failure_type, info) in enumerate(failure_info.items()):
        with cols[idx % 2]:
            severity_color = {
                'None': 'success',
                'Medium': 'warning',
                'High': 'danger',
                'Critical': 'danger'
            }[info['severity']]
            
            st.markdown(f"""
            <div class="{severity_color}-box">
                <h4>{info['icon']} {failure_type}</h4>
                <p><strong>Severity:</strong> {info['severity']}</p>
                <p><strong>Description:</strong> {info['description']}</p>
                <p><strong>Action:</strong> {info['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature Importance (if available)
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 Feature Importance</p>', unsafe_allow_html=True)
    
    if hasattr(xgb_model, 'feature_importances_'):
        feature_names = ['Air temperature [K]', 'Process temperature [K]', 
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            **get_dark_plotly_layout(),
            showlegend=False,
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance tidak tersedia untuk model ini")

# ============================
# HALAMAN: INTEGRATED ANALYSIS
# ============================
elif menu == "🔗 Integrated Analysis":
    st.markdown('<p class="main-header">🔗 Integrated Analysis - Clustering & Classification</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Info:</strong> Analisis ini menggabungkan hasil clustering dengan prediksi failure type 
        untuk mendapatkan insight yang lebih komprehensif tentang kondisi mesin.
    </div>
    """, unsafe_allow_html=True)
    
    # Predict failure types for all data
    st.markdown("### 🔄 Generating Predictions...")
    
    # Prepare features for classification
    classification_features = ['Air temperature [K]', 'Process temperature [K]', 
                              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Check if all features exist in df_cluster
    missing_features = [f for f in classification_features if f not in df_cluster.columns]
    
    if missing_features:
        st.warning(f"⚠️ Missing features in clustering data: {missing_features}")
        st.info("Menggunakan data klasifikasi terpisah untuk analisis...")
        
        # Use classification data instead
        X_classification = df_classification[classification_features]
        X_scaled = scaler_classifier.transform(X_classification)
        predictions = xgb_model.predict(X_scaled)
        predicted_failures = label_encoder.inverse_transform(predictions)
        
        # Create integrated dataframe
        df_integrated = df_classification.copy()
        df_integrated['Predicted_Failure'] = predicted_failures
        
        st.success("✅ Predictions generated!")
        
    else:
        X_classification = df_cluster[classification_features]
        X_scaled = scaler_classifier.transform(X_classification)
        predictions = xgb_model.predict(X_scaled)
        predicted_failures = label_encoder.inverse_transform(predictions)
        
        df_integrated = df_cluster.copy()
        df_integrated['Predicted_Failure'] = predicted_failures
        
        st.success("✅ Predictions generated!")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_failures = (df_integrated['Predicted_Failure'] != 'No Failure').sum()
        st.metric("⚠️ Total Failures", total_failures)
    
    with col2:
        no_failure = (df_integrated['Predicted_Failure'] == 'No Failure').sum()
        st.metric("✅ Normal Operation", no_failure)
    
    with col3:
        failure_rate = (total_failures / len(df_integrated)) * 100
        st.metric("📊 Failure Rate", f"{failure_rate:.2f}%")
    
    st.markdown("---")
    
    # Failure Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">🔴 Failure Type Distribution</p>', unsafe_allow_html=True)
        
        failure_dist = df_integrated['Predicted_Failure'].value_counts()
        
        fig = px.pie(
            values=failure_dist.values,
            names=failure_dist.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            **get_dark_plotly_layout(),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">📊 Failure Count by Type</p>', unsafe_allow_html=True)
        
        fig = px.bar(
            x=failure_dist.index,
            y=failure_dist.values,
            color=failure_dist.values,
            color_continuous_scale='reds',
            labels={'x': 'Failure Type', 'y': 'Count'}
        )
        fig.update_layout(
            **get_dark_plotly_layout(),
            showlegend=False,
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster-Failure Cross Analysis (if cluster column exists)
    if 'Cluster' in df_integrated.columns:
        st.markdown("---")
        st.markdown('<p class="sub-header">🔗 Cluster vs Failure Type Analysis</p>', unsafe_allow_html=True)
        
        # Create crosstab
        crosstab = pd.crosstab(
            df_integrated['Cluster'],
            df_integrated['Predicted_Failure'],
            normalize='index'
        ) * 100
        
        # Heatmap
        fig = px.imshow(
            crosstab.values,
            labels=dict(x="Failure Type", y="Cluster", color="Percentage (%)"),
            x=crosstab.columns,
            y=[f"Cluster {i}" for i in crosstab.index],
            color_continuous_scale='RdYlGn_r',
            aspect="auto"
        )
        fig.update_layout(
            **get_dark_plotly_layout(),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table view
        st.markdown("**Detailed Breakdown:**")
        crosstab_count = pd.crosstab(df_integrated['Cluster'], df_integrated['Predicted_Failure'])
        st.dataframe(crosstab_count, use_container_width=True)

# ============================
# HALAMAN: 3D VISUALIZATION
# ============================
elif menu == "📊 3D Visualization":
    st.markdown('<p class="main-header">📊 3D Interactive Visualization</p>', unsafe_allow_html=True)
    
    # Visualization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        viz_type = st.selectbox(
            "Visualization Type:",
            ["Clustering View", "Feature Exploration"]
        )
    
    with col2:
        point_size = st.slider("Point Size:", 1, 10, 4)
    
    with col3:
        opacity = st.slider("Opacity:", 0.1, 1.0, 0.6, 0.1)
    
    st.markdown("---")
    
    if viz_type == "Clustering View":
        st.markdown('<p class="sub-header">🔵 3D Clustering Visualization</p>', unsafe_allow_html=True)
        
        # Features for 3D plot
        features_cluster = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
        
        # Define color palette for clusters
        colors = px.colors.qualitative.Set3
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Plot each cluster
        for idx, cluster in enumerate(sorted(df_cluster['Cluster'].unique())):
            cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_data[features_cluster[0]],
                y=cluster_data[features_cluster[1]],
                z=cluster_data[features_cluster[2]],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(
                    size=point_size,
                    color=colors[idx % len(colors)],
                    opacity=opacity,
                    line=dict(width=0.5, color='rgba(255,255,255,0.3)')
                ),
                text=[f'Cluster: {cluster}<br>' + 
                      f'{features_cluster[0]}: {row[features_cluster[0]]:.2f}<br>' +
                      f'{features_cluster[1]}: {row[features_cluster[1]]:.2f}<br>' +
                      f'{features_cluster[2]}: {row[features_cluster[2]]:.2f}'
                      for _, row in cluster_data.iterrows()],
                hoverinfo='text',
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Add centroids
        show_centroids = st.checkbox("Show Centroids", value=True)
        if show_centroids:
            centroids = kmeans.cluster_centers_
            fig.add_trace(go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers',
                name='Centroids',
                marker=dict(
                    size=12,
                    color='#ff0000',
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                text=[f'Centroid {i}' for i in range(len(centroids))],
                hoverinfo='text',
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Apply dark theme layout for 3D
        fig.update_layout(
            **get_dark_plotly_3d_layout(),
            scene=dict(
                xaxis_title=features_cluster[0],
                yaxis_title=features_cluster[1],
                zaxis_title=features_cluster[2],
                xaxis=dict(
                    backgroundcolor='#1e1e1e',
                    gridcolor='#2d2d2d',
                    showbackground=True,
                    title_font=dict(size=14, color='#e0e0e0'),
                    tickfont=dict(color='#e0e0e0')
                ),
                yaxis=dict(
                    backgroundcolor='#1e1e1e',
                    gridcolor='#2d2d2d',
                    showbackground=True,
                    title_font=dict(size=14, color='#e0e0e0'),
                    tickfont=dict(color='#e0e0e0')
                ),
                zaxis=dict(
                    backgroundcolor='#1e1e1e',
                    gridcolor='#2d2d2d',
                    showbackground=True,
                    title_font=dict(size=14, color='#e0e0e0'),
                    tickfont=dict(color='#e0e0e0')
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=700,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Instructions
        st.markdown("""
        <div class="info-box">
            <strong>💡 Cara Interaksi:</strong>
            <ul>
                <li><strong>Rotate:</strong> Click dan drag untuk memutar grafik</li>
                <li><strong>Zoom:</strong> Scroll mouse untuk zoom in/out</li>
                <li><strong>Pan:</strong> Shift + drag untuk menggeser view</li>
                <li><strong>Hover:</strong> Arahkan mouse ke titik untuk melihat detail</li>
                <li><strong>Legend:</strong> Click nama cluster di legend untuk hide/show</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:  # Feature Exploration
        st.markdown('<p class="sub-header">📈 Feature Exploration (3D)</p>', unsafe_allow_html=True)
        
        available_features = ['Air temperature [K]', 'Process temperature [K]', 
                             'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        
        # Filter available features
        available_features = [f for f in available_features if f in df_cluster.columns]
        
        if len(available_features) >= 3:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_feature = st.selectbox("X-Axis:", available_features, index=0)
            with col2:
                y_feature = st.selectbox("Y-Axis:", available_features, index=min(1, len(available_features)-1))
            with col3:
                z_feature = st.selectbox("Z-Axis:", available_features, index=min(2, len(available_features)-1))
            
            # Color palette
            colors = px.colors.qualitative.Set3
            
            # Create 3D scatter
            fig = go.Figure()
            
            for idx, cluster in enumerate(sorted(df_cluster['Cluster'].unique())):
                cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
                
                fig.add_trace(go.Scatter3d(
                    x=cluster_data[x_feature],
                    y=cluster_data[y_feature],
                    z=cluster_data[z_feature],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(
                        size=point_size,
                        color=colors[idx % len(colors)],
                        opacity=opacity,
                        line=dict(width=0.5, color='rgba(255,255,255,0.3)')
                    ),
                    text=[f'Cluster: {cluster}<br>' +
                          f'{x_feature}: {row[x_feature]:.2f}<br>' +
                          f'{y_feature}: {row[y_feature]:.2f}<br>' +
                          f'{z_feature}: {row[z_feature]:.2f}'
                          for _, row in cluster_data.iterrows()],
                    hoverinfo='text',
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ))
            
            fig.update_layout(
                **get_dark_plotly_3d_layout(),
                scene=dict(
                    xaxis_title=x_feature,
                    yaxis_title=y_feature,
                    zaxis_title=z_feature,
                    xaxis=dict(
                        backgroundcolor='#1e1e1e',
                        gridcolor='#2d2d2d',
                        showbackground=True,
                        title_font=dict(size=14, color='#e0e0e0'),
                        tickfont=dict(color='#e0e0e0')
                    ),
                    yaxis=dict(
                        backgroundcolor='#1e1e1e',
                        gridcolor='#2d2d2d',
                        showbackground=True,
                        title_font=dict(size=14, color='#e0e0e0'),
                        tickfont=dict(color='#e0e0e0')
                    ),
                    zaxis=dict(
                        backgroundcolor='#1e1e1e',
                        gridcolor='#2d2d2d',
                        showbackground=True,
                        title_font=dict(size=14, color='#e0e0e0'),
                        tickfont=dict(color='#e0e0e0')
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.3)
                    )
                ),
                height=700,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak cukup fitur tersedia untuk visualisasi 3D")

# ============================
# HALAMAN: PREDICTION TOOL
# ============================
elif menu == "🎯 Prediction Tool":
    st.markdown('<p class="main-header">🎯 Real-time Prediction Tool</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Petunjuk:</strong> Masukkan nilai parameter mesin untuk mendapatkan prediksi 
        cluster dan jenis failure secara real-time.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input form
    st.markdown('<p class="sub-header">📝 Input Parameters</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        air_temp = st.number_input(
            "🌡️ Air Temperature (K)",
            min_value=290.0,
            max_value=310.0,
            value=300.0,
            step=0.1,
            help="Suhu udara sekitar dalam Kelvin"
        )
        
        process_temp = st.number_input(
            "🔥 Process Temperature (K)",
            min_value=300.0,
            max_value=320.0,
            value=310.0,
            step=0.1,
            help="Suhu proses mesin dalam Kelvin"
        )
        
        rotational_speed = st.number_input(
            "🔄 Rotational Speed (rpm)",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=10,
            help="Kecepatan rotasi mesin"
        )
    
    with col2:
        torque = st.number_input(
            "⚡ Torque (Nm)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1,
            help="Torsi mesin dalam Newton-meter"
        )
        
        tool_wear = st.number_input(
            "🔧 Tool Wear (min)",
            min_value=0,
            max_value=300,
            value=100,
            step=1,
            help="Waktu penggunaan tool dalam menit"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🚀 Predict Now", use_container_width=True):
        
        # Prepare input data
        input_data_cluster = np.array([[torque, process_temp, tool_wear]])
        input_data_classification = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear]])
        
        # Scale input
        input_scaled_cluster = scaler_cluster.transform(input_data_cluster)
        input_scaled_classification = scaler_classifier.transform(input_data_classification)
        
        # Predict cluster
        cluster_pred = kmeans.predict(input_scaled_cluster)[0]
        
        # Calculate distance to centroid
        distances = np.linalg.norm(kmeans.cluster_centers_ - input_scaled_cluster, axis=1)
        confidence_cluster = 100 * (1 - distances[cluster_pred] / distances.sum())
        
        # Predict failure type
        failure_pred_encoded = xgb_model.predict(input_scaled_classification)[0]
        failure_pred = label_encoder.inverse_transform([failure_pred_encoded])[0]
        
        # Get prediction probability
        failure_proba = xgb_model.predict_proba(input_scaled_classification)[0]
        confidence_failure = failure_proba.max() * 100
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="sub-header">🎯 Prediction Results</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔵 Cluster Prediction</h3>
                <h1>Cluster {cluster_pred}</h1>
                <p>Confidence: {confidence_cluster:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get cluster characteristics
            cluster_data = df_cluster[df_cluster['Cluster'] == cluster_pred]
            st.markdown("**Cluster Characteristics:**")
            st.write(f"• Avg Torque: {cluster_data['Torque [Nm]'].mean():.2f} Nm")
            st.write(f"• Avg Process Temp: {cluster_data['Process temperature [K]'].mean():.2f} K")
            st.write(f"• Avg Tool Wear: {cluster_data['Tool wear [min]'].mean():.2f} min")
        
        with col2:
            # Determine severity
            if failure_pred == "No Failure":
                box_class = "success-box"
                icon = "✅"
            elif failure_pred in ["Power Failure"]:
                box_class = "danger-box"
                icon = "🔴"
            else:
                box_class = "warning-box"
                icon = "⚠️"
            
            st.markdown(f"""
            <div class="{box_class}">
                <h3>{icon} Failure Prediction</h3>
                <h2>{failure_pred}</h2>
                <p>Confidence: {confidence_failure:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all probabilities
            st.markdown("**All Failure Probabilities:**")
            proba_df = pd.DataFrame({
                'Failure Type': label_encoder.classes_,
                'Probability (%)': failure_proba * 100
            }).sort_values('Probability (%)', ascending=False)
            
            for idx, row in proba_df.iterrows():
                st.write(f"• {row['Failure Type']}: {row['Probability (%)']:.2f}%")
        
        # Visualization
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Prediction Visualization</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster visualization
            features_cluster = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
            colors = px.colors.qualitative.Set3
            
            fig = go.Figure()
            
            # Plot existing data
            for idx, cluster in enumerate(sorted(df_cluster['Cluster'].unique())):
                cluster_data_viz = df_cluster[df_cluster['Cluster'] == cluster]
                fig.add_trace(go.Scatter3d(
                    x=cluster_data_viz[features_cluster[0]],
                    y=cluster_data_viz[features_cluster[1]],
                    z=cluster_data_viz[features_cluster[2]],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(
                        size=3, 
                        opacity=0.3,
                        color=colors[idx % len(colors)]
                    )
                ))
            
            # Plot input point
            fig.add_trace(go.Scatter3d(
                x=[torque],
                y=[process_temp],
                z=[tool_wear],
                mode='markers',
                name='Your Input',
                marker=dict(
                    size=15,
                    color='#ff0000',
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                text=[f'Your Input<br>Predicted: Cluster {cluster_pred}'],
                hoverinfo='text'
            ))
            
            fig.update_layout(
                **get_dark_plotly_3d_layout(),
                scene=dict(
                    xaxis_title=features_cluster[0],
                    yaxis_title=features_cluster[1],
                    zaxis_title=features_cluster[2],
                    xaxis=dict(
                        backgroundcolor='#1e1e1e',
                        gridcolor='#2d2d2d',
                        showbackground=True,
                        title_font=dict(size=12, color='#e0e0e0'),
                        tickfont=dict(color='#e0e0e0')
                    ),
                    yaxis=dict(
                        backgroundcolor='#1e1e1e',
                        gridcolor='#2d2d2d',
                        showbackground=True,
                        title_font=dict(size=12, color='#e0e0e0'),
                        tickfont=dict(color='#e0e0e0')
                    ),
                    zaxis=dict(
                        backgroundcolor='#1e1e1e',
                        gridcolor='#2d2d2d',
                        showbackground=True,
                        title_font=dict(size=12, color='#e0e0e0'),
                        tickfont=dict(color='#e0e0e0')
                    )
                ),
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Failure probability chart
            fig = px.bar(
                proba_df,
                x='Probability (%)',
                y='Failure Type',
                orientation='h',
                color='Probability (%)',
                color_continuous_scale='reds'
            )
            fig.update_layout(
                **get_dark_plotly_layout(),
                showlegend=False,
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown('<p class="sub-header">💡 Recommendations</p>', unsafe_allow_html=True)
        
        if failure_pred == "No Failure":
            st.markdown("""
            <div class="success-box">
                <h4>✅ Status: Normal Operation</h4>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Continue regular monitoring</li>
                    <li>Maintain current operational parameters</li>
                    <li>Schedule routine maintenance as planned</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif failure_pred == "Heat Dissipation Failure":
            st.markdown("""
            <div class="danger-box">
                <h4>🔥 Status: Heat Dissipation Issue Detected</h4>
                <p><strong>Immediate Actions Required:</strong></p>
                <ul>
                    <li>Check cooling system immediately</li>
                    <li>Clean heat sinks and ventilation</li>
                    <li>Verify coolant levels and circulation</li>
                    <li>Reduce operational load if temperature persists</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif failure_pred == "Power Failure":
            st.markdown("""
            <div class="danger-box">
                <h4>⚡ Status: Power System Issue Detected</h4>
                <p><strong>Immediate Actions Required:</strong></p>
                <ul>
                    <li>Inspect electrical connections</li>
                    <li>Check power supply stability</li>
                    <li>Test voltage and current levels</li>
                    <li>Examine motor and drive components</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif failure_pred == "Tool Wear Failure":
            st.markdown("""
            <div class="warning-box">
                <h4>🔧 Status: Tool Wear Detected</h4>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Schedule tool replacement</li>
                    <li>Inspect tool condition visually</li>
                    <li>Prepare replacement parts</li>
                    <li>Plan maintenance window</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif failure_pred == "Overstrain Failure":
            st.markdown("""
            <div class="warning-box">
                <h4>💪 Status: Overstrain Detected</h4>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Reduce operational load immediately</li>
                    <li>Check torque limits</li>
                    <li>Verify machine capacity specifications</li>
                    <li>Consider load distribution adjustment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:  # Random Failures
            st.markdown("""
            <div class="warning-box">
                <h4>🎲 Status: Random Failure Pattern</h4>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Conduct comprehensive system check</li>
                    <li>Review recent operational logs</li>
                    <li>Monitor for pattern changes</li>
                    <li>Increase inspection frequency</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #b0b0b0; padding: 1rem;">
    <p><strong>Dashboard Predictive Maintenance v2.0</strong></p>
    <p>Developed by Kelompok 9 (LC41) | BINUS University</p>
    <p>© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
