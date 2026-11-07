
# ============================================================
# DASHBOARD K-MEANS CLUSTERING GEMPA BUMI GLOBAL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. KONFIGURASI DASHBOARD
# ============================================================

st.set_page_config(
    page_title="K-Means Clustering Gempa Bumi",
    page_icon="🌍",
    layout="wide"
)

sns.set_style("whitegrid")


# ============================================================
# 2. JUDUL DAN DESKRIPSI
# ============================================================

st.title("🌍 Dashboard K-Means Clustering Gempa Bumi Global")

st.markdown("""
Dashboard ini digunakan untuk melakukan analisis pengelompokan
data gempa bumi global menggunakan algoritma **K-Means Clustering**.

Analisis meliputi eksplorasi data, preprocessing, penentuan
jumlah cluster, evaluasi model, visualisasi, dan interpretasi
karakteristik setiap cluster.
""")

st.divider()


# ============================================================
# 3. MEMUAT DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("global_earthquakes_seismic.csv")


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "Dataset tidak ditemukan! "
        "Pastikan file CSV berada dalam folder yang sama dengan app.py."
    )
    st.stop()


# ============================================================
# 4. IDENTIFIKASI FITUR NUMERIK
# ============================================================

default_features = [
    "latitude",
    "longitude",
    "depth",
    "mag",
    "sig",
    "nst",
    "dmin",
    "rms",
    "gap",
    "confidence_score",
    "tsunami"
]

excluded_columns = [
    "id",
    "time"
]

numeric_features = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in excluded_columns
]

selected_default = [
    col for col in default_features
    if col in numeric_features
]


# ============================================================
# 5. SIDEBAR - PENGATURAN INTERAKTIF
# ============================================================

st.sidebar.title("⚙️ Pengaturan Dashboard")

st.sidebar.markdown("### Pilihan Fitur")

selected_features = st.sidebar.multiselect(
    "Pilih fitur untuk clustering:",
    options=numeric_features,
    default=selected_default
)

if len(selected_features) < 2:
    st.warning("Pilih minimal 2 fitur numerik untuk clustering.")
    st.stop()


st.sidebar.markdown("### Jumlah Cluster")

K = st.sidebar.slider(
    "Pilih jumlah cluster (K):",
    min_value=2,
    max_value=10,
    value=4,
    step=1
)

st.sidebar.markdown("### Visualisasi")

feature_x = st.sidebar.selectbox(
    "Fitur sumbu X:",
    options=selected_features,
    index=0
)

feature_y = st.sidebar.selectbox(
    "Fitur sumbu Y:",
    options=selected_features,
    index=1
)

st.sidebar.info(
    "Perubahan pilihan fitur dan jumlah cluster "
    "akan memperbarui hasil analisis."
)


# ============================================================
# 6. PREPROCESSING DATA
# ============================================================

# Menghapus duplikasi
df_clean = df.drop_duplicates().copy()

# Mengambil fitur terpilih
X = df_clean[selected_features].copy()

# Memastikan data numerik
X = X.apply(pd.to_numeric, errors="coerce")

# Mengganti nilai tak hingga
X = X.replace([np.inf, -np.inf], np.nan)

# Menghapus kolom kosong
X = X.dropna(axis=1, how="all")

if X.shape[1] < 2:
    st.error("Fitur numerik valid kurang dari dua.")
    st.stop()

# Mengisi missing values dengan median
X = X.fillna(X.median())

# Menangani outlier menggunakan IQR
for col in X.columns:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    X[col] = X[col].clip(
        lower=batas_bawah,
        upper=batas_atas
    )


# ============================================================
# 7. STANDARISASI DATA
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)


# ============================================================
# 8. MODEL K-MEANS
# ============================================================

if len(X) <= K:
    st.error("Jumlah data tidak cukup untuk jumlah cluster yang dipilih.")
    st.stop()

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

# Menambahkan label cluster
df_result = df_clean.loc[X.index].copy()

df_result["cluster"] = cluster_labels


# ============================================================
# 9. EVALUASI MODEL
# ============================================================

inertia_value = kmeans.inertia_

silhouette_avg = silhouette_score(
    X_scaled,
    cluster_labels
)


# ============================================================
# 10. RINGKASAN DATASET
# ============================================================

st.header("1. Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Jumlah Data",
    f"{len(df):,}"
)

col2.metric(
    "Jumlah Kolom",
    len(df.columns)
)

col3.metric(
    "Fitur Clustering",
    len(X.columns)
)

col4.metric(
    "Jumlah Cluster",
    K
)

st.markdown("### Informasi Dataset")

st.dataframe(
    pd.DataFrame({
        "Nama Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("### Statistik Deskriptif")

st.dataframe(
    df[selected_features].describe().T,
    use_container_width=True
)


# ============================================================
# 11. EKSPLORASI DATA (EDA)
# ============================================================

st.divider()

st.header("2. Eksplorasi Data (EDA)")

st.subheader("A. Distribusi Fitur")

feature_distribution = st.selectbox(
    "Pilih fitur untuk melihat distribusi:",
    options=selected_features,
    key="distribution"
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(
    df[feature_distribution].dropna(),
    bins=30,
    kde=True,
    ax=ax
)

ax.set_title(
    f"Distribusi Fitur {feature_distribution}"
)

ax.set_xlabel(feature_distribution)
ax.set_ylabel("Frekuensi")

st.pyplot(fig)
plt.close(fig)

st.markdown(f"""
**Analisis Distribusi:**

Histogram menunjukkan persebaran nilai pada fitur
**{feature_distribution}**.

- Puncak histogram menunjukkan rentang nilai yang paling sering muncul.
- Distribusi simetris menunjukkan persebaran relatif seimbang.
- Distribusi yang menceng menunjukkan adanya ketidakseimbangan persebaran.
- Ekor panjang dapat menunjukkan adanya nilai ekstrem.
""")


# ------------------------------------------------------------
# B. BOXPLOT
# ------------------------------------------------------------

st.subheader("B. Boxplot")

fig, ax = plt.subplots(figsize=(10, 4))

sns.boxplot(
    x=df[feature_distribution],
    ax=ax
)

ax.set_title(
    f"Boxplot {feature_distribution}"
)

st.pyplot(fig)
plt.close(fig)

st.markdown("""
**Analisis Boxplot:**

Boxplot memperlihatkan median, kuartil, dan persebaran
nilai fitur. Titik yang berada di luar whisker merupakan
kandidat outlier yang perlu diperiksa.
""")


# ------------------------------------------------------------
# C. CORRELATION HEATMAP
# ------------------------------------------------------------

st.subheader("C. Korelasi Antarfitur")

correlation = X.corr()

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0,
    ax=ax
)

ax.set_title("Correlation Heatmap")

st.pyplot(fig)
plt.close(fig)

st.markdown("""
**Analisis Korelasi:**

Heatmap digunakan untuk mengetahui hubungan antarfitur.

- Nilai mendekati +1 menunjukkan korelasi positif kuat.
- Nilai mendekati -1 menunjukkan korelasi negatif kuat.
- Nilai mendekati 0 menunjukkan hubungan linear yang lemah.

Korelasi dapat membantu memahami hubungan antarfitur,
tetapi tidak secara langsung menentukan kontribusi fitur
terhadap pembentukan cluster.
""")


# ============================================================
# 12. ELBOW METHOD
# ============================================================

st.divider()

st.header("3. Penentuan Jumlah Cluster")

st.subheader("A. Elbow Method")

K_range = range(2, 11)

inertia_values = []

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(model.inertia_)


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    list(K_range),
    inertia_values,
    marker="o",
    linestyle="--"
)

ax.axvline(
    K,
    linestyle=":",
    label=f"K dipilih = {K}"
)

ax.set_title("Elbow Method")
ax.set_xlabel("Jumlah Cluster (K)")
ax.set_ylabel("Inertia")
ax.legend()

st.pyplot(fig)
plt.close(fig)

st.markdown("""
**Analisis Elbow Method:**

Elbow Method digunakan untuk menentukan jumlah cluster
yang sesuai berdasarkan nilai inertia.

Semakin banyak cluster, nilai inertia biasanya semakin kecil.
Jumlah cluster dipilih pada titik siku ketika penurunan inertia
mulai melambat.
""")


# ============================================================
# 13. SILHOUETTE SCORE
# ============================================================

st.subheader("B. Silhouette Score")

silhouette_values = []

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(score)


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    list(K_range),
    silhouette_values,
    marker="o",
    color="green"
)

ax.axvline(
    K,
    linestyle=":",
    label=f"K dipilih = {K}"
)

ax.set_title("Silhouette Score")
ax.set_xlabel("Jumlah Cluster (K)")
ax.set_ylabel("Silhouette Score")
ax.legend()

st.pyplot(fig)
plt.close(fig)

st.markdown("""
**Analisis Silhouette Score:**

Silhouette Score digunakan untuk mengukur kekompakan
dan keterpisahan cluster.

- Mendekati 1: cluster semakin kompak dan terpisah.
- Mendekati 0: data berada dekat batas antarcluster.
- Nilai negatif: sebagian data mungkin lebih dekat ke cluster lain.
""")


# ============================================================
# 14. HASIL EVALUASI
# ============================================================

st.subheader("C. Hasil Evaluasi K-Means")

col1, col2 = st.columns(2)

col1.metric(
    "Inertia",
    f"{inertia_value:,.2f}"
)

col2.metric(
    "Silhouette Score",
    f"{silhouette_avg:.4f}"
)

st.markdown(f"""
**Analisis Evaluasi:**

Model K-Means dengan jumlah cluster **K = {K}**
menghasilkan nilai inertia sebesar **{inertia_value:,.2f}**
dan Silhouette Score sebesar **{silhouette_avg:.4f}**.

Nilai tersebut digunakan untuk mengevaluasi kekompakan
dan keterpisahan cluster. Penilaian jumlah K juga perlu
mempertimbangkan grafik Elbow Method dan karakteristik data.
""")


# ============================================================
# 15. VISUALISASI HASIL CLUSTERING
# ============================================================

st.divider()

st.header("4. Visualisasi Hasil Clustering")

st.subheader("A. Scatter Plot Cluster")

plot_data = X.copy()

plot_data["cluster"] = cluster_labels.astype(str)

fig, ax = plt.subplots(figsize=(11, 6))

sns.scatterplot(
    data=plot_data,
    x=feature_x,
    y=feature_y,
    hue="cluster",
    palette="viridis",
    alpha=0.7,
    s=40,
    ax=ax
)

ax.set_title(
    f"Clustering {feature_x} vs {feature_y}"
)

ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)

ax.legend(title="Cluster")

st.pyplot(fig)
plt.close(fig)

st.markdown(f"""
**Analisis Scatter Plot:**

Grafik menunjukkan hasil pengelompokan data gempa
berdasarkan fitur **{feature_x}** dan **{feature_y}**.

Setiap warna menunjukkan kelompok cluster yang berbeda.
Titik yang berdekatan memiliki nilai fitur yang relatif mirip.

Perlu diperhatikan bahwa K-Means menggunakan seluruh
fitur terpilih, bukan hanya dua fitur pada scatter plot.
""")


# ============================================================
# 16. JUMLAH DATA SETIAP CLUSTER
# ============================================================

st.subheader("B. Jumlah Data Setiap Cluster")

cluster_count = (
    pd.Series(cluster_labels)
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    x=cluster_count.index.astype(str),
    y=cluster_count.values,
    ax=ax
)

ax.set_title("Jumlah Anggota Setiap Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Jumlah Data")

st.pyplot(fig)
plt.close(fig)

st.markdown("""
**Analisis Jumlah Cluster:**

Grafik menunjukkan jumlah data yang masuk ke setiap cluster.
Cluster dengan jumlah anggota lebih banyak menunjukkan
kelompok yang memiliki lebih banyak data dibandingkan
cluster lainnya.
""")


# ============================================================
# 17. KARAKTERISTIK SETIAP CLUSTER
# ============================================================

st.divider()

st.header("5. Karakteristik Setiap Cluster")

cluster_data = X.copy()

cluster_data["cluster"] = cluster_labels

# Rata-rata
cluster_mean = (
    cluster_data
    .groupby("cluster")
    .mean(numeric_only=True)
)

# Median
cluster_median = (
    cluster_data
    .groupby("cluster")
    .median(numeric_only=True)
)

st.subheader("A. Rata-rata Fitur Setiap Cluster")

st.dataframe(
    cluster_mean.round(3),
    use_container_width=True
)

st.subheader("B. Median Fitur Setiap Cluster")

st.dataframe(
    cluster_median.round(3),
    use_container_width=True
)


# ------------------------------------------------------------
# C. BAR CHART KARAKTERISTIK
# ------------------------------------------------------------

st.subheader("C. Perbandingan Rata-rata Fitur")

feature_profile = st.selectbox(
    "Pilih fitur untuk membandingkan cluster:",
    options=selected_features,
    key="profile"
)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    x=cluster_mean.index,
    y=cluster_mean[feature_profile],
    ax=ax
)

ax.set_title(
    f"Rata-rata {feature_profile} pada Setiap Cluster"
)

ax.set_xlabel("Cluster")
ax.set_ylabel(f"Rata-rata {feature_profile}")

st.pyplot(fig)
plt.close(fig)

st.markdown(f"""
**Analisis Karakteristik Cluster:**

Grafik memperlihatkan perbandingan nilai rata-rata
fitur **{feature_profile}** pada setiap cluster.

Perbedaan nilai rata-rata dapat digunakan untuk
mengidentifikasi karakteristik yang membedakan
masing-masing kelompok.
""")


# ============================================================
# 18. TABEL HASIL CLUSTERING
# ============================================================

st.divider()

st.header("6. Dataset Hasil Clustering")

st.dataframe(
    df_result.head(100),
    use_container_width=True
)

# Download hasil clustering
csv_result = df_result.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Hasil Clustering CSV",
    data=csv_result,
    file_name="hasil_clustering_kmeans.csv",
    mime="text/csv"
)


# ============================================================
# 19. KESIMPULAN DASHBOARD
# ============================================================

st.divider()

st.header("7. Kesimpulan Analisis")

st.markdown(f"""
Berdasarkan proses clustering menggunakan algoritma
K-Means, diperoleh hasil pengelompokan sebanyak
**{K} cluster** dari **{len(X):,} data gempa bumi**.

Hasil evaluasi model menunjukkan:

- Inertia: **{inertia_value:,.2f}**
- Silhouette Score: **{silhouette_avg:.4f}**
- Jumlah data: **{len(X):,}**
- Jumlah fitur: **{len(X.columns)}**

Karakteristik setiap cluster dapat diketahui melalui
perbandingan nilai rata-rata dan median fitur.

Hasil clustering menggambarkan pengelompokan data
berdasarkan kemiripan karakteristik numerik. Hasil ini
tidak secara langsung menunjukkan tingkat bahaya
atau memprediksi kejadian gempa bumi.
""")

st.success("Analisis clustering selesai!")

st.caption(
    "Praktikum Analisis dan Visualisasi Clustering "
    "Menggunakan Algoritma K-Means."
)
