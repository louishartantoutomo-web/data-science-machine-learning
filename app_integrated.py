"""
Dashboard Predictive Maintenance - Clustering & Classification
Aplikasi Streamlit untuk visualisasi dan prediksi cluster kondisi mesin + klasifikasi failure types
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
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
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    .metric-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #fafafa;
    }
    
    .cluster-info {
        background-color: #1a2d3d;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
        color: #d1ecf1;
    }
    
    .program-box {
        background-color: #1a3d2e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        color: #d4edda;
    }
    
    .success-box {
        padding: 1rem;
        background-color: #1a3d2e;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
        color: #d4edda;
    }
    
    .warning-box {
        padding: 1rem;
        background-color: #3d3420;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
        color: #fff3cd;
    }
    
    .danger-box {
        padding: 1rem;
        background-color: #3d2022;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        margin: 1rem 0;
        color: #f8d7da;
    }
    
    .info-box {
        padding: 1rem;
        background-color: #1a2d3d;
        border-left: 4px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
        color: #d1ecf1;
    }
    
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
    
    p, li, span, div {
        color: #e0e0e0;
    }
    
    strong {
        color: #ffffff;
    }
    
    h1, h2, h3, h4 {
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# DATA INTERPRETASI CLUSTER
# ============================
CLUSTER_INTERPRETATIONS = {
    0: {
        "label": "Beban Tinggi Optimal",
        "torque": "40.5 - 66.7",
        "temp": "305.9 - 309.9",
        "wear": "0 - 117",
        "programs": [
            "Optimasi Jadwal Perawatan Prediktif: Tim perawatan akan menjalankan predictive maintenance dengan analisis getaran dan pengambilan sampel oli setiap 300 jam operasi pada bearing dan gearbox untuk mendeteksi pola kerusakan dini dan mencegah downtime tidak terencana.",
            "Program Penyeimbangan Beban Dinamis: Perencana produksi akan menggunakan algoritma penjadwalan untuk menjaga torsi tetap di kisaran 45–60 Nm dengan cara mendistribusikan ulang pesanan kerja setiap shift, sehingga umur komponen kritis lebih panjang dan efisiensi energi meningkat 15–20%.",
            "Peningkatan Material Komponen Beban Tinggi: Manajer perawatan akan mengganti bearing dan coupling pada shaft utama dengan baja keras dalam 2 minggu, karena torsi tinggi membutuhkan ketahanan lelah lebih baik untuk mencapai MTBF 50% lebih tinggi."
        ]
    },
    1: {
        "label": "Suhu Tinggi Ringan",
        "torque": "20.9 - 42.4",
        "temp": "309.8 - 313.7",
        "wear": "0 - 114",
        "programs": [
            "Perbaikan Sistem Pendingin Mendesak: Tim pendingin akan melakukan uji tekanan, pembersihan heat exchanger, dan penggantian coolant dalam 3 hari, karena suhu tinggi pada beban rendah menunjukkan efisiensi pendinginan menurun drastis.",
            "Pemasangan Ventilasi Tambahan: Tim fasilitas akan memasang 3 unit exhaust fan industri dengan kapasitas besar dan saluran udara optimal dalam 2 minggu untuk menurunkan suhu ruangan 5–8°C sehingga suhu proses kembali normal <310K.",
            "Pemantauan Suhu Rutin: Tim perawatan akan melakukan inspeksi pencitraan termal setiap Senin pagi sebelum shift produksi untuk mendeteksi titik panas >85°C pada motor, bearing, dan panel listrik."
        ]
    },
    2: {
        "label": "Kondisi Kritis",
        "torque": "38.9 - 66.8",
        "temp": "310 - 313.8",
        "wear": "96 - 253",
        "programs": [
            "Penghentian Darurat dan Overhaul Besar: Manajer perawatan harus menghentikan operasi mesin dalam 24 jam dan melakukan pembongkaran total selama 2 minggu dengan penggantian 80% komponen aus karena risiko kegagalan besar.",
            "Analisis Akar Masalah Komprehensif: Tim keandalan akan melakukan analisis kerusakan dengan uji metalurgi, pemeriksaan pola keausan, dan perhitungan tegangan panas untuk menemukan penyebab utama degradasi.",
            "Validasi Komisioning Bertahap: Tim kontrol kualitas akan melakukan uji jalan 72 jam dengan pemantauan lebih dari 100 parameter (torsi, suhu, getaran, keausan alat) menggunakan peningkatan beban bertahap 25–50–75–100%."
        ]
    },
    3: {
        "label": "Aus Parah",
        "torque": "20.5 - 42.8",
        "temp": "305.7 - 310.1",
        "wear": "108 - 253",
        "programs": [
            "Penggantian Alat Mendesak: Tim perawatan akan mengganti semua alat potong, insert, dan toolholder dalam 48 jam dengan produk berkualitas tinggi untuk menghindari kegagalan mendadak.",
            "Audit Program Perawatan Alat: Tim keandalan akan meninjau log perawatan dan jadwal penggantian alat, kemudian menetapkan interval maksimal 150 jam operasi untuk penggantian preventif.",
            "Pemantauan Keausan Real-time: Tim kontrol akan memasang sensor laser untuk mengukur dimensi alat setiap 2 jam dan menghentikan mesin otomatis jika keausan >0.3 mm untuk mencegah produk cacat."
        ]
    },
    4: {
        "label": "Kondisi Optimal",
        "torque": "20.3 - 42.6",
        "temp": "305.7 - 309.9",
        "wear": "0 - 105",
        "programs": [
            "Pemeliharaan Preventif Rutin: Tim perawatan akan melanjutkan jadwal pemeliharaan preventif setiap 500 jam operasi dengan checklist menyeluruh dan dokumentasi digital untuk mempertahankan kinerja optimal.",
            "Benchmark Best Practice: Tim operasi akan mendokumentasikan parameter optimal dan membuat SOP standar yang dapat diadopsi oleh shift lain untuk menjaga konsistensi operasional.",
            "Program Pemantauan Berkelanjutan: Tim kontrol kualitas akan melakukan inspeksi visual harian dan pencatatan parameter operasi untuk mendeteksi deviasi <5% sehingga kondisi optimal tetap terjaga."
        ]
    },
    5: {
        "label": "Suhu Tinggi Kritis",
        "torque": "38.9 - 66.7",
        "temp": "310 - 313.8",
        "wear": "0 - 95",
        "programs": [
            "Shutdown Darurat untuk Investigasi Termal: Manajer perawatan harus menghentikan operasi dalam 4 jam dan melakukan inspeksi termal dengan kamera inframerah untuk mengidentifikasi overheating >90°C pada motor, bearing, atau sistem hidrolik.",
            "Overhaul Sistem Pendingin Menyeluruh: Tim pendingin akan melakukan penggantian komponen cooling (pompa, radiator, thermostat) dan flush sistem dalam 1 minggu dengan tekanan uji 150% untuk memastikan kapasitas pendinginan optimal.",
            "Desain Ulang Aliran Pendingin: Tim engineering akan melakukan simulasi CFD dan memodifikasi layout saluran pendingin dalam 2 minggu untuk meningkatkan heat dissipation 30–40% di area kritis."
        ]
    },
    6: {
        "label": "Beban Tinggi Kritis",
        "torque": "38.8 - 66.8",
        "temp": "305.9 - 310.1",
        "wear": "107 - 253",
        "programs": [
            "Revisi Total Beban Operasi: Manajer produksi harus segera menurunkan target produksi 30–40% untuk mengurangi torsi rata-rata menjadi <50 Nm sambil merencanakan upgrade kapasitas mesin dalam 3 bulan.",
            "Penggantian Komponen Transmisi: Tim perawatan akan mengganti gearbox, bearing, dan shaft dalam 2 minggu dengan spesifikasi heavy-duty untuk menangani beban tinggi tanpa kerusakan cepat.",
            "Studi Kelayakan Penambahan Mesin: Tim engineering akan melakukan analisis biaya-manfaat untuk menambah 1 unit mesin paralel dalam 6 bulan sehingga beban terdistribusi dan torsi per mesin turun 40–50%."
        ]
    },
    7: {
        "label": "Kondisi Sangat Kritis",
        "torque": "38.9 - 66.8",
        "temp": "310 - 313.8",
        "wear": "107 - 253",
        "programs": [
            "Shutdown Darurat Segera: Manajer perawatan harus menghentikan operasi dalam 1 jam karena kombinasi torsi tinggi, suhu tinggi, dan keausan parah menunjukkan risiko kegagalan katastrofik >80%.",
            "Overhaul Total dan Root Cause Analysis: Tim keandalan akan melakukan pembongkaran lengkap, analisis kerusakan metalurgi, dan perhitungan thermal-mechanical stress dalam 3 minggu untuk menemukan akar masalah sistemik.",
            "Redesign Sistem dan Upgrade Kapasitas: Tim engineering akan merancang ulang sistem transmisi, pendinginan, dan kontrol beban dengan simulasi FEA dalam 2 bulan, lalu melakukan retrofit dengan komponen berkualitas tinggi untuk mencegah pengulangan kondisi kritis."
        ]
    }
}

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

# Prepare df for original menus
df = df_cluster.copy()

# ============================
# SIDEBAR - MENU SECTIONS
# ============================
st.sidebar.markdown("# ⚙️ Predictive Maintenance")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Menu Navigation")

# Section 1: Clustering Model (Original menus - without prediction)
st.sidebar.markdown("**🔵 Clustering Model**")
clustering_menu = st.sidebar.radio(
    "Clustering Analysis:",
    ["🏠 Overview", "📈 Visualisasi 3D", "📋 Analisis Cluster"],
    key="clustering"
)

st.sidebar.markdown("---")

# Section 2: Classification Model (Info only)
st.sidebar.markdown("**🔴 Classification Model**")
classification_menu = st.sidebar.radio(
    "Failure Classification:",
    ["📊 Failure Info"],
    key="classification"
)

st.sidebar.markdown("---")

# Section 3: Integrated Analysis (with combined prediction)
st.sidebar.markdown("**🔗 Integrated Model**")
integrated_menu = st.sidebar.radio(
    "Combined Analysis:",
    ["🎯 Prediksi Terintegrasi", "🔗 Integrated Dashboard"],
    key="integrated"
)

st.sidebar.markdown("---")

# Model Info
st.sidebar.markdown("### ℹ️ Model Information")
st.sidebar.info(f"""
**🔵 Clustering:**  
- K-Means: {kmeans.n_clusters} clusters
- Features: 3 variables

**🔴 Classification:**  
- XGBoost Classifier
- Classes: {len(label_encoder.classes_)}
- Features: 5 variables
""")

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
# DETERMINE ACTIVE MENU
# ============================
# Check which section was last clicked
if 'last_menu' not in st.session_state:
    st.session_state.last_menu = 'clustering'

# Detect which radio button was clicked
if clustering_menu and clustering_menu != st.session_state.get('prev_clustering'):
    st.session_state.last_menu = 'clustering'
    st.session_state.prev_clustering = clustering_menu

if classification_menu and classification_menu != st.session_state.get('prev_classification'):
    st.session_state.last_menu = 'classification'
    st.session_state.prev_classification = classification_menu

if integrated_menu and integrated_menu != st.session_state.get('prev_integrated'):
    st.session_state.last_menu = 'integrated'
    st.session_state.prev_integrated = integrated_menu

# Set active menu
if st.session_state.last_menu == 'clustering':
    menu = clustering_menu
elif st.session_state.last_menu == 'classification':
    menu = classification_menu
else:
    menu = integrated_menu

# ============================
# MENU CLUSTERING: 🏠 OVERVIEW
# ============================
if menu == "🏠 Overview":
    st.markdown('<p class="main-header">🏠 Overview Dashboard - Clustering Model</p>', unsafe_allow_html=True)
    
    st.header("📊 Ringkasan Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Cluster", kmeans.n_clusters)
    with col3:
        st.metric("Fitur Clustering", "3")
    with col4:
        if 'Target' in df.columns:
            st.metric("Target Failure", f"{df['Target'].sum():,}")
        else:
            st.metric("Data Points", f"{len(df):,}")
    
    st.markdown("---")
    
    # Distribusi cluster
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Data per Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Jumlah Data'},
            title='Jumlah Data di Setiap Cluster',
            color=cluster_counts.values,
            color_continuous_scale='Blues'
        )
        layout = get_dark_plotly_layout()
        layout['showlegend'] = False
        layout['height'] = 400
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Proporsi Cluster")
        fig = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title='Proporsi Data di Setiap Cluster',
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        layout = get_dark_plotly_layout()
        layout['height'] = 400
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif Variabel Clustering")
    features = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
    stats_df = df[features].describe().T
    stats_df = stats_df.round(2)
    st.dataframe(stats_df, use_container_width=True)

# ============================
# MENU CLUSTERING: 📈 VISUALISASI 3D
# ============================
elif menu == "📈 Visualisasi 3D":
    st.markdown('<p class="main-header">📈 Visualisasi 3D Interaktif - Clustering</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Cara Interaksi:</strong>
        <ul>
            <li><strong>Rotate:</strong> Click dan drag untuk memutar grafik</li>
            <li><strong>Zoom:</strong> Scroll mouse untuk zoom in/out</li>
            <li><strong>Pan:</strong> Shift + drag untuk menggeser view</li>
            <li><strong>Hover:</strong> Arahkan mouse ke titik untuk melihat detail</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Control options
    col1, col2, col3 = st.columns(3)
    with col1:
        point_size = st.slider("Point Size:", 1, 10, 4)
    with col2:
        opacity = st.slider("Opacity:", 0.1, 1.0, 0.6, 0.1)
    with col3:
        show_centroids = st.checkbox("Show Centroids", value=True)
    
    st.markdown("---")
    
    # Features for 3D plot
    features = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
    
    # Color palette
    colors = px.colors.qualitative.Set3
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Plot each cluster
    for idx, cluster in enumerate(sorted(df['Cluster'].unique())):
        cluster_data = df[df['Cluster'] == cluster]
        
        # Get cluster info
        cluster_info = CLUSTER_INTERPRETATIONS.get(cluster, {"label": f"Cluster {cluster}"})
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data[features[0]],
            y=cluster_data[features[1]],
            z=cluster_data[features[2]],
            mode='markers',
            name=f'Cluster {cluster}: {cluster_info["label"]}',
            marker=dict(
                size=point_size,
                color=colors[idx % len(colors)],
                opacity=opacity,
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            text=[f'Cluster: {cluster}<br>' +
                  f'{cluster_info["label"]}<br>' +
                  f'{features[0]}: {row[features[0]]:.2f}<br>' +
                  f'{features[1]}: {row[features[1]]:.2f}<br>' +
                  f'{features[2]}: {row[features[2]]:.2f}'
                  for _, row in cluster_data.iterrows()],
            hoverinfo='text',
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    # Add centroids
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
    
    # Apply dark theme layout
    layout_3d = get_dark_plotly_3d_layout()
    layout_3d['scene'].update({
        'xaxis_title': features[0],
        'yaxis_title': features[1],
        'zaxis_title': features[2],
        'yaxis': {
            'range': [300, 320]
        },
        'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.3}}
    })
    layout_3d['height'] = 700
    layout_3d['showlegend'] = True
    fig.update_layout(layout_3d)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================
# MENU CLUSTERING: 🔮 PREDIKSI CLUSTER
# ============================
elif menu == "📋 Analisis Cluster":
    st.markdown('<p class="main-header">📋 Analisis Detail Cluster</p>', unsafe_allow_html=True)
    
    # Cluster selection
    st.subheader("🎯 Pilih Cluster untuk Analisis")
    selected_cluster = st.selectbox(
        "Cluster:",
        options=sorted(df['Cluster'].unique()),
        format_func=lambda x: f"Cluster {x}: {CLUSTER_INTERPRETATIONS[x]['label']}"
    )
    
    cluster_info = CLUSTER_INTERPRETATIONS[selected_cluster]
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    # Cluster info box
    st.markdown(f"""
    <div class="cluster-info">
        <h3>Cluster {selected_cluster}: {cluster_info['label']}</h3>
        <p><strong>Torque Range:</strong> {cluster_info['torque']} Nm</p>
        <p><strong>Temperature Range:</strong> {cluster_info['temp']} K</p>
        <p><strong>Tool Wear Range:</strong> {cluster_info['wear']} min</p>
        <p><strong>Jumlah Data:</strong> {len(cluster_data):,} ({len(cluster_data)/len(df)*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Statistik Cluster")
        features = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
        cluster_stats = cluster_data[features].describe().T
        cluster_stats = cluster_stats.round(2)
        st.dataframe(cluster_stats, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribusi Variabel")
        feature_to_plot = st.selectbox("Pilih Variabel:", features)
        
        fig = px.histogram(
            cluster_data,
            x=feature_to_plot,
            nbins=30,
            title=f'Distribusi {feature_to_plot}',
            color_discrete_sequence=['#667eea']
        )
        layout = get_dark_plotly_layout()
        layout['showlegend'] = False
        layout['height'] = 300
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Program Perawatan yang Disarankan")
    for i, program in enumerate(cluster_info['programs'], 1):
        st.markdown(f"""
        <div class="program-box">
            <strong>{i}. {program.split(':')[0]}:</strong><br>
            {':'.join(program.split(':')[1:])}
        </div>
        """, unsafe_allow_html=True)

# ============================
# MENU CLASSIFICATION: 📊 FAILURE INFO
# ============================
elif menu == "📊 Failure Info":
    st.markdown('<p class="main-header">📊 Failure Classification Information</p>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Classification Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #667eea;">🤖 Model Type</h4>
            <h3>XGBoost</h3>
            <p>Classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #667eea;">🎯 Classes</h4>
            <h3>{len(label_encoder.classes_)}</h3>
            <p>Failure Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #667eea;">📊 Features</h4>
            <h3>5</h3>
            <p>Input Variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Failure Types Info
    st.markdown("### 🔍 Failure Types Information")
    
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
    
    # Feature Importance
    st.markdown("---")
    st.markdown("### 📊 Feature Importance")
    
    try:
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
                color_continuous_scale='viridis',
                title='Feature Importance for Failure Classification'
            )
            layout = get_dark_plotly_layout()
            layout['showlegend'] = False
            layout['height'] = 400
            layout['yaxis'] = {'categoryorder': 'total ascending'}
            fig.update_layout(layout)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance tidak tersedia untuk model ini")
    except Exception as e:
        st.warning(f"⚠️ Could not display feature importance: {str(e)}")

# ============================
# MENU CLASSIFICATION: 🎯 PREDIKSI FAILURE
# ============================
# ============================
# MENU INTEGRATED: 🎯 PREDIKSI TERINTEGRASI
# ============================
elif menu == "🎯 Prediksi Terintegrasi":
    st.markdown('<p class="main-header">🎯 Prediksi Terintegrasi - Cluster & Failure Type</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Petunjuk:</strong> Masukkan 5 parameter mesin untuk mendapatkan prediksi komprehensif:
        <ul>
            <li><strong>Clustering:</strong> Kondisi operasional mesin (Cluster 0-7)</li>
            <li><strong>Classification:</strong> Jenis failure yang mungkin terjadi</li>
            <li><strong>Recommendations:</strong> Program perawatan dari cluster + actions untuk failure</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input form
    st.subheader("📝 Input Parameter Mesin")
    
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
    if st.button("🚀 Prediksi Sekarang", use_container_width=True):
        
        # ============================
        # CLUSTERING PREDICTION
        # ============================
        # Prepare input data for clustering (3 features)
        input_data_cluster = np.array([[torque, process_temp, tool_wear]])
        
        # Scale input
        input_scaled_cluster = scaler_cluster.transform(input_data_cluster)
        
        # Predict cluster
        cluster_pred = kmeans.predict(input_scaled_cluster)[0]
        
        # Calculate distance to centroid
        distances = np.linalg.norm(kmeans.cluster_centers_ - input_scaled_cluster, axis=1)
        confidence_cluster = 100 * (1 - distances[cluster_pred] / distances.sum())
        
        # Get cluster info
        cluster_info = CLUSTER_INTERPRETATIONS[cluster_pred]
        
        # ============================
        # CLASSIFICATION PREDICTION
        # ============================
        # Prepare input data for classification (5 features)
        input_data_classification = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear]])
        
        # Scale input
        input_scaled_classification = scaler_classifier.transform(input_data_classification)
        
        # Predict failure type
        failure_pred_encoded = xgb_model.predict(input_scaled_classification)[0]
        failure_pred = label_encoder.inverse_transform([failure_pred_encoded])[0]
        
        # Get prediction probability
        failure_proba = xgb_model.predict_proba(input_scaled_classification)[0]
        confidence_failure = failure_proba.max() * 100
        
        # ============================
        # DISPLAY INTEGRATED RESULTS
        # ============================
        st.markdown("---")
        st.markdown('<p class="sub-header">🎯 Hasil Prediksi Terintegrasi</p>', unsafe_allow_html=True)
        
        # Summary boxes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea;">🔵 Clustering Prediction</h3>
                <h1 style="color: #fafafa;">Cluster {cluster_pred}</h1>
                <h3 style="color: #17a2b8;">{cluster_info['label']}</h3>
                <p>Confidence: {confidence_cluster:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**📊 Karakteristik Cluster:**")
            st.write(f"• Torque Range: {cluster_info['torque']} Nm")
            st.write(f"• Temperature Range: {cluster_info['temp']} K")
            st.write(f"• Tool Wear Range: {cluster_info['wear']} min")
        
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
                <h3>{icon} 🔴 Failure Prediction</h3>
                <h2>{failure_pred}</h2>
                <p>Confidence: {confidence_failure:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all failure probabilities
            st.markdown("**📈 All Failure Probabilities:**")
            proba_df = pd.DataFrame({
                'Failure Type': label_encoder.classes_,
                'Probability (%)': failure_proba * 100
            }).sort_values('Probability (%)', ascending=False)
            
            for idx, row in proba_df.iterrows():
                st.write(f"• {row['Failure Type']}: {row['Probability (%)']:.2f}%")
        
        # ============================
        # VISUALIZATIONS
        # ============================
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Visualisasi Prediksi</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔵 Posisi Cluster dalam 3D Space**")
            
            # 3D plot for clustering
            features = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
            colors = px.colors.qualitative.Set3
            
            fig = go.Figure()
            
            # Plot existing data (sample)
            for idx, cluster in enumerate(sorted(df['Cluster'].unique())):
                cluster_data = df[df['Cluster'] == cluster].sample(n=min(50, len(df[df['Cluster'] == cluster])))
                fig.add_trace(go.Scatter3d(
                    x=cluster_data[features[0]],
                    y=cluster_data[features[1]],
                    z=cluster_data[features[2]],
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
                text=[f'Your Input<br>Cluster: {cluster_pred}<br>Failure: {failure_pred}'],
                hoverinfo='text'
            ))
            
            layout_3d = get_dark_plotly_3d_layout()
            layout_3d['scene'].update({
                'xaxis_title': features[0],
                'yaxis_title': features[1],
                'zaxis_title': features[2]
            })
            layout_3d['height'] = 500
            layout_3d['showlegend'] = True
            fig.update_layout(layout_3d)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**🔴 Distribusi Probabilitas Failure**")
            
            # Failure probability chart
            fig = px.bar(
                proba_df,
                x='Probability (%)',
                y='Failure Type',
                orientation='h',
                color='Probability (%)',
                color_continuous_scale='reds',
                title='Probabilitas Semua Jenis Failure'
            )
            layout = get_dark_plotly_layout()
            layout['showlegend'] = False
            layout['height'] = 500
            layout['yaxis'] = {'categoryorder': 'total ascending'}
            fig.update_layout(layout)
            st.plotly_chart(fig, use_container_width=True)
        
        # ============================
        # COMBINED RECOMMENDATIONS
        # ============================
        st.markdown("---")
        st.markdown('<p class="sub-header">💡 Rekomendasi Terintegrasi</p>', unsafe_allow_html=True)
        
        # Cluster-based recommendations
        st.markdown(f"""
        <div class="cluster-info">
            <h4>🔵 Rekomendasi Berdasarkan Cluster {cluster_pred}: {cluster_info['label']}</h4>
            <p>Berikut adalah program perawatan yang disarankan untuk kondisi operasional mesin Anda:</p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, program in enumerate(cluster_info['programs'], 1):
            st.markdown(f"""
            <div class="program-box">
                <strong>{i}. {program.split(':')[0]}:</strong><br>
                {':'.join(program.split(':')[1:])}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Failure-based recommendations
        st.markdown(f"""
        <div class="cluster-info">
            <h4>🔴 Rekomendasi Berdasarkan Prediksi Failure: {failure_pred}</h4>
            <p>Berikut adalah tindakan yang disarankan berdasarkan jenis failure yang diprediksi:</p>
        </div>
        """, unsafe_allow_html=True)
        
        if failure_pred == "No Failure":
            st.markdown("""
            <div class="success-box">
                <h4>✅ Status: Normal Operation</h4>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Continue regular monitoring</li>
                    <li>Maintain current operational parameters</li>
                    <li>Schedule routine maintenance as planned</li>
                    <li>Document current settings for future reference</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif failure_pred == "Heat Dissipation Failure":
            st.markdown("""
            <div class="danger-box">
                <h4>🔥 Status: Heat Dissipation Issue Detected</h4>
                <p><strong>Immediate Actions Required:</strong></p>
                <ul>
                    <li><strong>Priority 1:</strong> Check cooling system immediately (dalam 4 jam)</li>
                    <li><strong>Priority 2:</strong> Clean heat sinks and verify ventilation (dalam 24 jam)</li>
                    <li><strong>Priority 3:</strong> Verify coolant levels and circulation</li>
                    <li><strong>Priority 4:</strong> Reduce operational load if temperature persists above 312K</li>
                    <li><strong>Long-term:</strong> Consider upgrading cooling capacity atau desain ulang aliran pendingin</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif failure_pred == "Power Failure":
            st.markdown("""
            <div class="danger-box">
                <h4>⚡ Status: Power System Issue Detected</h4>
                <p><strong>Immediate Actions Required:</strong></p>
                <ul>
                    <li><strong>Priority 1:</strong> Inspect electrical connections (dalam 2 jam)</li>
                    <li><strong>Priority 2:</strong> Check power supply stability dengan multimeter</li>
                    <li><strong>Priority 3:</strong> Test voltage and current levels di semua phase</li>
                    <li><strong>Priority 4:</strong> Examine motor windings dan drive components untuk signs of damage</li>
                    <li><strong>Long-term:</strong> Install power quality monitor untuk continuous tracking</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif failure_pred == "Tool Wear Failure":
            st.markdown("""
            <div class="warning-box">
                <h4>🔧 Status: Tool Wear Detected</h4>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li><strong>Priority 1:</strong> Schedule tool replacement dalam 48-72 jam</li>
                    <li><strong>Priority 2:</strong> Inspect tool condition visually untuk keausan >0.3mm</li>
                    <li><strong>Priority 3:</strong> Prepare replacement parts (insert, toolholder)</li>
                    <li><strong>Priority 4:</strong> Plan maintenance window untuk minimal disruption</li>
                    <li><strong>Long-term:</strong> Review tool replacement interval, consider reducing dari current schedule</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif failure_pred == "Overstrain Failure":
            st.markdown("""
            <div class="warning-box">
                <h4>💪 Status: Overstrain Detected</h4>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li><strong>Priority 1:</strong> Reduce operational load immediately 30-40%</li>
                    <li><strong>Priority 2:</strong> Check torque limits dan pastikan tidak exceed rated capacity</li>
                    <li><strong>Priority 3:</strong> Verify machine capacity specifications vs actual load</li>
                    <li><strong>Priority 4:</strong> Consider load distribution adjustment atau production scheduling</li>
                    <li><strong>Long-term:</strong> Evaluate need for additional machine capacity atau upgrade</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:  # Random Failures
            st.markdown("""
            <div class="warning-box">
                <h4>🎲 Status: Random Failure Pattern</h4>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li><strong>Priority 1:</strong> Conduct comprehensive system check dalam 24 jam</li>
                    <li><strong>Priority 2:</strong> Review recent operational logs untuk pattern identification</li>
                    <li><strong>Priority 3:</strong> Monitor for pattern changes di next 48-72 jam operasi</li>
                    <li><strong>Priority 4:</strong> Increase inspection frequency (daily → per shift)</li>
                    <li><strong>Long-term:</strong> Implement condition monitoring sensors untuk early detection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # ============================
        # INTEGRATED INSIGHTS
        # ============================
        st.markdown("---")
        st.markdown('<p class="sub-header">🔍 Integrated Insights</p>', unsafe_allow_html=True)
        
        # Risk level assessment
        risk_score = 0
        if cluster_pred in [2, 7]:  # Kondisi Kritis / Sangat Kritis
            risk_score += 40
        elif cluster_pred in [1, 5, 6]:  # Suhu Tinggi / Beban Tinggi
            risk_score += 25
        
        if failure_pred == "Power Failure":
            risk_score += 40
        elif failure_pred in ["Heat Dissipation Failure", "Overstrain Failure"]:
            risk_score += 30
        elif failure_pred in ["Tool Wear Failure", "Random Failures"]:
            risk_score += 15
        
        # Display risk level
        if risk_score >= 60:
            risk_level = "CRITICAL"
            risk_box = "danger-box"
            risk_icon = "🔴"
            risk_action = "SHUTDOWN & IMMEDIATE MAINTENANCE REQUIRED"
        elif risk_score >= 40:
            risk_level = "HIGH"
            risk_box = "warning-box"
            risk_icon = "⚠️"
            risk_action = "Urgent attention needed within 24 hours"
        elif risk_score >= 20:
            risk_level = "MEDIUM"
            risk_box = "warning-box"
            risk_icon = "🟡"
            risk_action = "Schedule maintenance within 1 week"
        else:
            risk_level = "LOW"
            risk_box = "success-box"
            risk_icon = "✅"
            risk_action = "Continue normal operation with routine monitoring"
        
        st.markdown(f"""
        <div class="{risk_box}">
            <h3>{risk_icon} Overall Risk Assessment: {risk_level}</h3>
            <p><strong>Risk Score:</strong> {risk_score}/100</p>
            <p><strong>Cluster Contribution:</strong> Cluster {cluster_pred} ({cluster_info['label']})</p>
            <p><strong>Failure Contribution:</strong> {failure_pred} ({confidence_failure:.1f}% confidence)</p>
            <p><strong>Recommended Action:</strong> {risk_action}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Priority matrix
        st.markdown("### 🎯 Action Priority Matrix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Immediate Actions (0-24 hours):**")
            immediate_actions = []
            
            if failure_pred in ["Power Failure", "Heat Dissipation Failure"]:
                immediate_actions.append(f"• Address {failure_pred} (CRITICAL)")
            
            if cluster_pred in [2, 7]:
                immediate_actions.append(f"• Review Cluster {cluster_pred} conditions (HIGH)")
            
            if risk_score >= 60:
                immediate_actions.append("• Prepare for potential shutdown")
            
            if not immediate_actions:
                immediate_actions.append("• Continue monitoring")
            
            for action in immediate_actions:
                st.write(action)
        
        with col2:
            st.markdown("**Short-term Actions (1-7 days):**")
            short_term_actions = []
            
            if failure_pred == "Tool Wear Failure":
                short_term_actions.append("• Schedule tool replacement")
            
            if failure_pred == "Overstrain Failure":
                short_term_actions.append("• Optimize load distribution")
            
            if cluster_pred in [1, 5, 6]:
                short_term_actions.append(f"• Implement Cluster {cluster_pred} maintenance programs")
            
            if not short_term_actions:
                short_term_actions.append("• Execute routine maintenance schedule")
            
            for action in short_term_actions:
                st.write(action)


elif menu == "🔗 Integrated Dashboard":
    st.markdown('<p class="main-header">🔗 Integrated Analysis - Clustering & Classification</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Info:</strong> Dashboard ini menggabungkan hasil clustering (kondisi mesin) dengan 
        klasifikasi failure type untuk memberikan analisis komprehensif.
    </div>
    """, unsafe_allow_html=True)
    
    # Predict failure types for clustering data
    st.markdown("### 🔄 Generating Integrated Predictions...")
    
    classification_features = ['Air temperature [K]', 'Process temperature [K]', 
                              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Check if all features exist in df_cluster
    missing_features = [f for f in classification_features if f not in df_cluster.columns]
    
    if missing_features:
        st.warning(f"⚠️ Missing features in clustering data: {missing_features}")
        st.info("Menggunakan data klasifikasi terpisah untuk analisis...")
        
        # Use classification data
        X_classification = df_classification[classification_features]
        X_scaled = scaler_classifier.transform(X_classification)
        predictions = xgb_model.predict(X_scaled)
        predicted_failures = label_encoder.inverse_transform(predictions)
        
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
        st.subheader("🔴 Failure Type Distribution")
        
        failure_dist = df_integrated['Predicted_Failure'].value_counts()
        
        fig = px.pie(
            values=failure_dist.values,
            names=failure_dist.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
            title='Distribution of Failure Types'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        layout = get_dark_plotly_layout()
        layout['height'] = 400
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Failure Count by Type")
        
        fig = px.bar(
            x=failure_dist.index,
            y=failure_dist.values,
            color=failure_dist.values,
            color_continuous_scale='reds',
            labels={'x': 'Failure Type', 'y': 'Count'},
            title='Count of Each Failure Type'
        )
        layout = get_dark_plotly_layout()
        layout['showlegend'] = False
        layout['height'] = 400
        layout['xaxis_tickangle'] = -45
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster-Failure Cross Analysis (if cluster column exists)
    if 'Cluster' in df_integrated.columns:
        st.markdown("---")
        st.subheader("🔗 Cluster vs Failure Type Analysis")
        
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
            y=[f"Cluster {i}: {CLUSTER_INTERPRETATIONS.get(i, {}).get('label', f'Cluster {i}')}" 
               for i in crosstab.index],
            color_continuous_scale='RdYlGn_r',
            aspect="auto",
            title='Heatmap: Cluster vs Failure Type'
        )
        layout = get_dark_plotly_layout()
        layout['height'] = 500
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
        
        # Table view
        st.markdown("**📋 Detailed Breakdown (Count):**")
        crosstab_count = pd.crosstab(df_integrated['Cluster'], df_integrated['Predicted_Failure'])
        st.dataframe(crosstab_count, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        # Find cluster with highest failure rate
        failure_by_cluster = df_integrated.groupby('Cluster').apply(
            lambda x: (x['Predicted_Failure'] != 'No Failure').sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        worst_cluster = failure_by_cluster.index[0]
        worst_rate = failure_by_cluster.values[0]
        
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ Highest Risk Cluster</h4>
            <p><strong>Cluster {worst_cluster}: {CLUSTER_INTERPRETATIONS.get(worst_cluster, {}).get('label', f'Cluster {worst_cluster}')}</strong></p>
            <p>Failure Rate: {worst_rate:.1f}%</p>
            <p>This cluster requires immediate attention and prioritized maintenance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Find most common failure in each cluster
        st.markdown("**🔍 Dominant Failure Type per Cluster:**")
        for cluster in sorted(df_integrated['Cluster'].unique()):
            cluster_failures = df_integrated[df_integrated['Cluster'] == cluster]['Predicted_Failure']
            most_common = cluster_failures.value_counts().index[0]
            count = cluster_failures.value_counts().values[0]
            percentage = (count / len(cluster_failures)) * 100
            
            st.write(f"• **Cluster {cluster} ({CLUSTER_INTERPRETATIONS.get(cluster, {}).get('label', '')}):** "
                    f"{most_common} ({percentage:.1f}%)")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #b0b0b0; padding: 1rem;">
    <p><strong>Dashboard Predictive Maintenance v2.5</strong></p>
    <p>Clustering Model + Classification Model + Integrated Analysis</p>
    <p>Developed by Kelompok 9 (LC41) | BINUS University</p>
    <p>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
