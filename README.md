# 🎯 Dashboard Predictive Maintenance - Integrated Clustering & Classification

Dashboard interaktif komprehensif berbasis Streamlit yang mengintegrasikan **Machine Learning Clustering (K-Means)** dan **Classification (XGBoost)** untuk analisis dan prediksi kondisi mesin secara real-time.

---

## ✨ Fitur Utama

### 🔵 Clustering Analysis (K-Means)
- Pengelompokan kondisi mesin menjadi 8 cluster
- Visualisasi 3D interaktif
- Analisis karakteristik setiap cluster
- Perbandingan statistik antar cluster

### 🔴 Failure Classification (XGBoost)
- Prediksi 6 jenis failure mesin:
  - ✅ No Failure
  - 🔥 Heat Dissipation Failure
  - ⚡ Power Failure
  - 🔧 Tool Wear Failure
  - 💪 Overstrain Failure
  - 🎲 Random Failures
- Confidence score untuk setiap prediksi
- Feature importance analysis

### 🔗 Integrated Analysis
- Analisis gabungan cluster dan failure type
- Cross-tabulation cluster vs failure
- Heatmap distribusi failure per cluster
- Insight komprehensif kondisi mesin

### 🎯 Real-time Prediction Tool
- Input 5 parameter mesin
- Prediksi cluster dan failure type secara bersamaan
- Visualisasi posisi data dalam 3D space
- Rekomendasi actionable berdasarkan prediksi

---

## 📋 Requirement

### Python Version
- Python 3.8 atau lebih baru

### Dependencies
```
streamlit>=1.31.0
pandas>=2.1.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.4.0
joblib>=1.3.0
xgboost>=2.0.0
```

---

## 🚀 Instalasi & Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements_integrated.txt
```

### 2. Struktur File

Pastikan Anda memiliki file-file berikut:

```
project/
│
├── app_integrated.py                 # Aplikasi Streamlit utama
├── requirements_integrated.txt       # Dependencies
│
├── Model Files (Clustering)
│   ├── kmeans_model.pkl             # Model K-Means
│   ├── scaler.pkl                   # Scaler untuk clustering
│   └── clustered_data.csv           # Data dengan label cluster
│
├── Model Files (Classification)
│   ├── xgb_classification_model.pkl # Model XGBoost
│   ├── classifier_scaler.pkl        # Scaler untuk classification
│   ├── label_encoder_failure_type.pkl # Label encoder
│   └── classification_training_data.csv # Data training classification
│
└── README_INTEGRATED.md             # Dokumentasi ini
```

### 3. Jalankan Dashboard

```bash
streamlit run app_integrated.py
```

Dashboard akan otomatis terbuka di browser (default: `http://localhost:8501`)

---

## 🎨 Menu & Fitur Dashboard

### 1. 🏠 Overview
**Fungsi:** Ringkasan umum dataset dan distribusi

**Fitur:**
- Total data points
- Jumlah cluster
- Jumlah tipe failure
- Distribusi cluster (bar chart)
- Daftar failure types
- Statistik deskriptif features

**Kapan digunakan:** 
Untuk mendapatkan overview cepat tentang dataset dan distribusinya

---

### 2. 🔵 Clustering Analysis
**Fungsi:** Analisis mendalam setiap cluster

**Fitur:**
- Pemilihan cluster untuk analisis detail
- Statistik cluster (mean, std, min, max, dll)
- Distribusi feature dalam cluster
- Perbandingan cluster vs overall data
- Histogram distribusi feature

**Cara Menggunakan:**
1. Pilih cluster dari dropdown
2. Lihat metrics: total data points, percentage, avg torque, avg temperature
3. Analisis statistik cluster
4. Pilih feature untuk melihat distribusinya
5. Bandingkan dengan data keseluruhan

**Insight yang didapat:**
- Karakteristik unik setiap cluster
- Perbedaan antar cluster
- Feature dominan di setiap cluster

---

### 3. 🔴 Failure Classification
**Fungsi:** Informasi tentang model klasifikasi dan jenis-jenis failure

**Fitur:**
- Informasi model (XGBoost, 6 classes, 5 features)
- Detail setiap failure type:
  - Icon dan severity level
  - Deskripsi masalah
  - Recommended actions
- Feature importance chart

**Failure Types Explained:**

| Failure Type | Severity | Description | Action |
|-------------|----------|-------------|---------|
| 🔥 Heat Dissipation | High | Overheat karena cooling gagal | Periksa sistem cooling |
| ✅ No Failure | None | Operasi normal | Monitoring rutin |
| 💪 Overstrain | High | Beban melebihi kapasitas | Kurangi beban |
| ⚡ Power Failure | Critical | Masalah kelistrikan | Periksa power supply |
| 🎲 Random Failures | Medium | Kegagalan acak | Analisis pattern detail |
| 🔧 Tool Wear | Medium | Tool aus | Ganti tool |

**Kapan digunakan:**
Untuk memahami jenis-jenis failure dan severitynya sebelum melakukan prediksi

---

### 4. 🔗 Integrated Analysis
**Fungsi:** Analisis gabungan clustering dan classification

**Fitur:**
- Auto-prediction failure type untuk semua data
- Total failures vs normal operation
- Failure rate percentage
- Pie chart distribusi failure types
- Bar chart failure count
- **Heatmap cluster vs failure type** (KEY FEATURE!)
- Cross-tabulation table

**Cara Menggunakan:**
1. Dashboard otomatis generate prediksi
2. Lihat metrics: total failures, normal operation, failure rate
3. Analisis distribusi failure dari pie chart
4. **Fokus pada heatmap:** melihat cluster mana yang prone ke failure tertentu
5. Gunakan table untuk detail breakdown

**Insight yang didapat:**
- Cluster mana yang paling banyak mengalami failure
- Jenis failure apa yang dominan di setiap cluster
- Pattern korelasi antara cluster dan failure type
- Risk assessment per cluster

**Example Insight:**
```
"Cluster 3 memiliki 80% probability Heat Dissipation Failure 
→ Indikasi: mesin di cluster ini beroperasi pada suhu tinggi
→ Action: Prioritas pengecekan cooling system untuk cluster 3"
```

---

### 5. 📊 3D Visualization
**Fungsi:** Visualisasi interaktif data dalam ruang 3D

**Mode 1: Clustering View**
- Scatter 3D semua data points berdasarkan cluster
- Feature: Torque, Process Temperature, Tool Wear
- Show/hide centroids
- Interactive rotation, zoom, hover

**Mode 2: Feature Exploration**
- Pilih sendiri 3 feature untuk axis X, Y, Z
- Eksplorasi hubungan antar feature
- Clustering overlay

**Cara Interaksi:**
- **Drag:** Rotate grafik
- **Scroll:** Zoom in/out
- **Hover:** Lihat detail data point
- **Legend click:** Show/hide cluster tertentu

**Customize:**
- Point size (1-10)
- Opacity (0.1-1.0)

**Kapan digunakan:**
- Memvisualisasikan separasi cluster
- Mencari outlier
- Memahami hubungan 3 feature sekaligus
- Presentasi visual

---

### 6. 🎯 Prediction Tool
**Fungsi:** Prediksi real-time untuk data baru

**Input Parameters:**
1. 🌡️ **Air Temperature (K)** [290-310]
   - Suhu udara sekitar
   
2. 🔥 **Process Temperature (K)** [300-320]
   - Suhu proses mesin
   
3. 🔄 **Rotational Speed (rpm)** [1000-3000]
   - Kecepatan rotasi
   
4. ⚡ **Torque (Nm)** [0-100]
   - Momen gaya mesin
   
5. 🔧 **Tool Wear (min)** [0-300]
   - Waktu penggunaan tool

**Output:**

**A. Cluster Prediction**
- Predicted cluster number
- Confidence score
- Cluster characteristics (avg torque, temp, tool wear)

**B. Failure Prediction**
- Predicted failure type
- Confidence score
- Probability untuk semua failure types
- Color-coded severity:
  - 🟢 Green: No Failure
  - 🟡 Yellow: Medium severity
  - 🔴 Red: High/Critical severity

**C. Visualizations**
- 3D plot dengan posisi input (red diamond)
- Horizontal bar chart probability semua failure types

**D. Recommendations**
- Specific actions berdasarkan predicted failure
- Prioritized checklist
- Maintenance suggestions

**Cara Menggunakan:**
1. Input semua 5 parameter
2. Klik tombol "🚀 Predict Now"
3. Lihat hasil prediksi cluster dan failure
4. Perhatikan confidence score (>80% = high confidence)
5. Analisis probability distribution
6. Baca dan terapkan recommendations

**Use Cases:**
- **Real-time Monitoring:** Input sensor data aktual
- **What-if Analysis:** Ubah parameter, lihat impact
- **Preventive Maintenance:** Prediksi sebelum failure
- **Decision Support:** Data-driven maintenance planning

---

## 📊 Data Requirements

### Clustering Data (clustered_data.csv)
Minimal columns:
- `Torque [Nm]`
- `Process temperature [K]`
- `Tool wear [min]`
- `Cluster` (hasil clustering)

Optional columns (untuk feature exploration):
- `Air temperature [K]`
- `Rotational speed [rpm]`

### Classification Data (classification_training_data.csv)
Required columns:
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

---

## ⚙️ Model Information

### K-Means Clustering
- **Algorithm:** K-Means
- **Number of Clusters:** 8 (optimal dari elbow method)
- **Features:** 3 (Torque, Process Temp, Tool Wear)
- **Scaling:** StandardScaler

### XGBoost Classification
- **Algorithm:** XGBoost Classifier
- **Number of Classes:** 6 failure types
- **Features:** 5 (Air Temp, Process Temp, Speed, Torque, Tool Wear)
- **Scaling:** StandardScaler
- **Output:** Multi-class classification with probability

---

## 🎨 Customization

### Mengubah Warna Cluster
Edit section `fig.add_trace()` dalam 3D visualization:
```python
marker=dict(
    size=point_size,
    color='your_color_here',  # Ubah warna
    opacity=opacity
)
```

### Menambah Failure Type Baru
1. Retrain model dengan class baru
2. Update `failure_info` dictionary di menu Failure Classification
3. Update recommendations di Prediction Tool

### Mengubah Threshold Severity
Edit section severity classification:
```python
if confidence_score > 90:  # High confidence
    severity = "Critical"
elif confidence_score > 70:  # Medium confidence
    severity = "High"
```

---

## 🔧 Troubleshooting

### Error: ModuleNotFoundError: No module named 'xgboost'
**Solusi:**
```bash
pip install xgboost --upgrade
```

### Error: Model file not found
**Solusi:**
Pastikan semua file model ada:
- kmeans_model.pkl
- scaler.pkl
- xgb_classification_model.pkl
- classifier_scaler.pkl
- label_encoder_failure_type.pkl

### Error: KeyError in features
**Solusi:**
Cek kolom di CSV:
```python
import pandas as pd
df = pd.read_csv('clustered_data.csv')
print(df.columns)  # Pastikan nama kolom sesuai
```

### Dashboard lambat / lag
**Solusi:**
1. Kurangi point size di 3D visualization
2. Kurangi opacity
3. Tutup menu yang tidak digunakan
4. Gunakan sample data untuk testing:
   ```python
   df_sample = df.sample(n=1000)  # Ambil 1000 data saja
   ```

### Port sudah digunakan
**Solusi:**
```bash
streamlit run app_integrated.py --server.port 8502
```

---

## 📈 Performance Tips

### Untuk Dataset Besar (>10,000 rows)
1. **Use sampling untuk visualization:**
   ```python
   df_viz = df.sample(n=5000, random_state=42)
   ```

2. **Enable caching:**
   ```python
   @st.cache_data(ttl=3600)  # Cache 1 jam
   def load_data():
       return pd.read_csv('data.csv')
   ```

3. **Optimize 3D plot:**
   - Reduce point size
   - Lower opacity
   - Hide less important clusters

---

## 🎓 Tutorial Penggunaan

### Scenario 1: Analisis Kondisi Mesin Existing
**Tujuan:** Memahami distribusi kondisi mesin saat ini

**Steps:**
1. Buka menu **🏠 Overview**
   - Lihat total data dan distribusi cluster
   
2. Buka menu **🔗 Integrated Analysis**
   - Lihat failure rate
   - Analisis heatmap cluster vs failure
   - Identifikasi cluster berisiko tinggi
   
3. Buka menu **🔵 Clustering Analysis**
   - Analisis detail cluster berisiko tinggi
   - Bandingkan dengan cluster normal
   
4. **Output:** 
   - Cluster mana yang perlu prioritas maintenance
   - Pattern failure di setiap cluster

---

### Scenario 2: Prediksi untuk Mesin Baru
**Tujuan:** Prediksi kondisi mesin dengan sensor data baru

**Steps:**
1. Buka menu **🎯 Prediction Tool**

2. Input sensor data:
   - Air Temp: 298 K
   - Process Temp: 310 K
   - Speed: 1500 rpm
   - Torque: 45 Nm
   - Tool Wear: 180 min

3. Klik "🚀 Predict Now"

4. Analisis hasil:
   - Cluster prediction → Bandingkan dengan cluster normal
   - Failure prediction → Check severity
   - Probability → Lihat risk alternatives
   
5. Baca recommendations

6. **Output:**
   - Prediksi failure type
   - Action items prioritas
   - Risk level

---

### Scenario 3: What-if Analysis
**Tujuan:** Memahami impact perubahan parameter

**Steps:**
1. Buka **🎯 Prediction Tool**

2. Set baseline:
   - Input kondisi "normal" mesin

3. Ubah 1 parameter (misal: Torque)
   - Torque 30 → Prediction A
   - Torque 50 → Prediction B
   - Torque 70 → Prediction C

4. Bandingkan hasil:
   - Cluster berubah?
   - Failure type berubah?
   - Confidence score?

5. **Output:**
   - Sensitivity analysis
   - Parameter threshold
   - Operational boundaries

---

### Scenario 4: Preventive Maintenance Planning
**Tujuan:** Tentukan prioritas maintenance

**Steps:**
1. Menu **🔗 Integrated Analysis**
   - Lihat cluster dengan highest failure rate
   - Note failure types yang dominan

2. Menu **🔵 Clustering Analysis**
   - Analisis karakteristik cluster high-risk
   - Tentukan threshold parameter

3. Menu **🔴 Failure Classification**
   - Review recommended actions per failure type
   - Prepare checklist

4. Menu **🎯 Prediction Tool**
   - Simulasi kondisi existing machines
   - Prioritize berdasarkan confidence score

5. **Output:**
   - Maintenance priority list
   - Resource allocation
   - Timeline planning

---

## 💡 Best Practices

### 1. Data Quality
- Pastikan data sensor akurat
- Handle missing values sebelum input
- Outlier detection penting

### 2. Model Interpretation
- High confidence (>85%): Trust prediction
- Medium confidence (70-85%): Consider alternatives
- Low confidence (<70%): Manual inspection needed

### 3. Action Priority
```
Critical Severity (Power Failure) → Immediate action
High Severity (Heat, Overstrain) → Within 24 hours
Medium Severity (Tool Wear, Random) → Within 1 week
No Failure → Continue monitoring
```

### 4. Regular Model Update
- Retrain quarterly dengan data baru
- Monitor model drift
- Update thresholds based on real outcomes

---

## 🔒 Security Notes

### Data Privacy
- Dashboard berjalan locally
- Tidak ada data dikirim ke external server
- Model tersimpan lokal

### Production Deployment
Untuk production, tambahkan:
1. **Authentication:**
   ```python
   import streamlit_authenticator as stauth
   ```

2. **Logging:**
   ```python
   import logging
   logging.basicConfig(filename='app.log', level=logging.INFO)
   ```

3. **Error handling:**
   ```python
   try:
       # prediction code
   except Exception as e:
       st.error(f"Error: {e}")
       logging.error(f"Prediction error: {e}")
   ```

---

## 📞 Support & Contact

### Developer Team
**Kelompok 9 (LC41) - BINUS University**

- **Louis Hartanto Utomo** - 2702285744
- **Raymond Christopher Sofian** - 2702320482
- **Gelfand Hanli Lim** - 2702322071
- **Karlina Gunawan** - 2702252973

### Reporting Issues
Jika menemukan bug atau punya saran:
1. Dokumentasikan error message
2. Screenshot jika perlu
3. Describe steps to reproduce
4. Contact developer team

---

## 📚 References

### Documentation
- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Docs](https://plotly.com/python/)
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [Scikit-learn Docs](https://scikit-learn.org/stable/)

### Related Papers
- K-Means Clustering for Predictive Maintenance
- XGBoost for Failure Classification
- Feature Importance in Machine Failure Prediction

---

## 🎉 Changelog

### Version 2.0 (Current)
- ✅ Integrated XGBoost classification
- ✅ 6 failure types prediction
- ✅ Integrated analysis heatmap
- ✅ Enhanced prediction tool
- ✅ Comprehensive recommendations
- ✅ Improved UI/UX

### Version 1.0
- ✅ K-Means clustering
- ✅ 3D visualization
- ✅ Basic cluster analysis

---

## 📄 License

© 2024 Kelompok 9 (LC41) - BINUS University
All Rights Reserved

---

## 🌟 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] All model files (.pkl) present
- [ ] Data files (.csv) present
- [ ] Run `streamlit run app_integrated.py`
- [ ] Dashboard opens successfully
- [ ] Test prediction tool
- [ ] Explore all menus

---

**Selamat menggunakan Dashboard Predictive Maintenance Integrated! 🚀**

For questions: Contact developer team
Last Updated: 2026
