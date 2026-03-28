# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==================== CẤU HÌNH ====================
st.set_page_config(
    page_title="Hệ thống Đề xuất & Phân cụm Bất động sản", 
    layout="wide"
)

# ==================== ĐƯỜNG DẪN FILE ====================
# Lấy đường dẫn thư mục hiện tại (nơi chứa file app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tạo đường dẫn đến các thư mục con
PATH_BT1 = os.path.join(BASE_DIR, "file_pkl_bt1")
PATH_BT2 = os.path.join(BASE_DIR, "file_pkl_bt2")

# ==================== KIỂM TRA FILE ====================
# Kiểm tra sự tồn tại của thư mục và file
def check_paths():
    """Kiểm tra đường dẫn và hiển thị cảnh báo nếu thiếu file"""
    issues = []
    
    if not os.path.exists(PATH_BT1):
        issues.append(f"Khong tim thay thu muc: {PATH_BT1}")
    else:
        # Kiểm tra các file cần thiết trong BT1
        required_bt1 = ['df_recommend.pkl', 'hybrid_sim.pkl', 'cosine_sim.pkl']
        for file in required_bt1:
            if not os.path.exists(os.path.join(PATH_BT1, file)):
                issues.append(f"Thieu file: {file} trong thu muc file_pkl_bt1")
    
    if not os.path.exists(PATH_BT2):
        issues.append(f"Khong tim thay thu muc: {PATH_BT2}")
    else:
        # Kiểm tra các file cần thiết trong BT2
        required_bt2 = ['scaler.pkl', 'kmeans.pkl', 'gmm.pkl', 'agg.pkl', 
                        'pca.pkl', 'df_clustered.pkl', 'cluster_info.pkl']
        for file in required_bt2:
            if not os.path.exists(os.path.join(PATH_BT2, file)):
                issues.append(f"Thieu file: {file} trong thu muc file_pkl_bt2")
    
    return issues

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
"""Load tất cả các model với xử lý lỗi"""
models = {}

try:
    # Load BT1 models
    with st.spinner("Dang tai models BT1..."):
        models['df_recommend'] = joblib.load(os.path.join(PATH_BT1, "df_recommend.pkl"))
        models['hybrid_sim'] = joblib.load(os.path.join(PATH_BT1, "hybrid_sim.pkl"))
        models['cosine_sim'] = joblib.load(os.path.join(PATH_BT1, "cosine_sim.pkl"))
    
    # Load BT2 models
    with st.spinner("Dang tai models BT2..."):
        models['scaler'] = joblib.load(os.path.join(PATH_BT2, "scaler.pkl"))
        models['kmeans'] = joblib.load(os.path.join(PATH_BT2, "kmeans.pkl"))
        models['gmm'] = joblib.load(os.path.join(PATH_BT2, "gmm.pkl"))
        models['agg'] = joblib.load(os.path.join(PATH_BT2, "agg.pkl"))
        models['pca'] = joblib.load(os.path.join(PATH_BT2, "pca.pkl"))
        models['df_clustered'] = joblib.load(os.path.join(PATH_BT2, "df_clustered.pkl"))
        models['cluster_info'] = joblib.load(os.path.join(PATH_BT2, "cluster_info.pkl"))
    
    return models
    
except Exception as e:
    st.error(f"Loi khi tai model: {str(e)}")
    st.stop()

# Load models
with st.spinner("Dang tai mo hinh..."):
models = load_models()

st.sidebar.success("Tai mo hinh thanh cong!")

# Hiển thị thông tin debug (có thể bỏ sau khi chạy ổn định)
with st.sidebar.expander("Thong tin he thong"):
st.write(f"Thu muc goc: `{BASE_DIR}`")
st.write(f"file_pkl_bt1: `{PATH_BT1}`")
st.write(f"file_pkl_bt2: `{PATH_BT2}`")
st.write(f"Python: `{sys.version}`")

# ==================== MENU ====================
menu = st.sidebar.radio(
"MENU",
["Bai toan kinh doanh", "Danh gia Mo hinh", "Du doan phan cum", "De xuat bat dong san", "Info Team"]
)

# ==================== BUSINESS PROBLEM ====================
if menu == "Bai toan kinh doanh":
st.title("Bai toan Kinh doanh")

st.markdown("""
### Van de
- Khach hang co nhu cau mua nha tai cac quan **Binh Thanh, Go Vap, Phu Nhuan**
- Can he thong de xuat nha phu hop
- Can phan cum de hieu ro phan khuc thi truong

### Muc tieu
1. De xuat nha dua tren noi dung (TF-IDF + Cosine)
2. Phan cum bang KMeans, GMM, Agglomerative
3. Hybrid Recommender (noi dung + gia + vi tri)

### Du lieu
- **7,881** bat dong san tai 3 quan
- Thuoc tinh: gia, dien tich, so phong, quan
- Mo ta chi tiet tu tin dang
""")

col1, col2, col3 = st.columns(3)
df = models['df_recommend']
with col1:
    st.metric("Tong so BDS", f"{len(df):,}")
with col2:
    st.metric("Gia trung binh", f"{df['gia_ban_num'].mean()/1e9:.1f} ty")
with col3:
    st.metric("Dien tich TB", f"{df['dien_tich_num'].mean():.0f} m²")

# ==================== EVALUATION ====================
elif menu == "Danh gia Mo hinh":
st.title("Danh gia Mo hinh")

tab1, tab2 = st.tabs(["Clustering", "Recommendation"])

with tab1:
    st.subheader("Danh gia phan cum")
    
    info = models['cluster_info']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("KMeans", f"{info['kmeans_score']:.4f}")
        st.write(f"Cum 0: {info['cluster_counts'][0]:,} BDS")
        st.write(f"Cum 1: {info['cluster_counts'][1]:,} BDS")
    
    with col2:
        st.metric("GMM", f"{info['gmm_score']:.4f}")
        st.write("Diem thap nhat")
    
    with col3:
        st.metric("Agglomerative", f"{info['agg_score']:.4f}")
        st.write("Tot nhat")

with tab2:
    st.subheader("Danh gia Recommender")
    st.write("""
    **TF-IDF + Cosine:** Du tren noi dung mo ta
    
    **Hybrid:** Ket hop 3 yeu to
    - Noi dung (50%)
    - Gia (25%)
    - Vi tri (25%)
    
    **Khuyen nghi: Hybrid**
    """)

# ==================== PREDICTION ====================
elif menu == "Du doan phan cum":
st.title("Du doan phan cum")

# Thêm thông tin hướng dẫn
with st.expander("Huong dan", expanded=False):
    st.write("""
    **Cach su dung:**
    1. Nhap gia ban va dien tich
    2. Chon quan
    3. Nhan "Du doan" de xem ket qua phan cum
    
    **Giai thich ket qua:**
    - **Cum 0:** Nha pho thong (gia ~6.5 ty, dien tich ~48m²)
    - **Cum 1:** Nha cao cap (gia ~20 ty, dien tich ~114m²)
    """)

# Form nhập liệu
col1, col2 = st.columns(2)

with col1:
    gia = st.number_input("Gia (ty)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
    dien_tich = st.number_input("Dien tich (m²)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)

with col2:
    quan = st.selectbox("Quan", ["Binh Thanh", "Go Vap", "Phu Nhuan"])
    gia_tham_khao = {
        "Binh Thanh": "6.5 - 20 ty",
        "Go Vap": "5.5 - 18 ty",
        "Phu Nhuan": "6 - 22 ty"
    }
    st.info(f"Gia tham khao quan {quan}: {gia_tham_khao[quan]}")

if st.button("Du doan", type="primary"):
    try:
        # Tính toán
        gia_num = gia * 1e9
        price_per_m2 = gia_num / dien_tich
        quan_map = {"Binh Thanh": 0, "Go Vap": 1, "Phu Nhuan": 2}
        quan_encoded = quan_map[quan]
        
        new_data = np.array([[gia_num, dien_tich, price_per_m2, quan_encoded]])
        new_scaled = models['scaler'].transform(new_data)
        
        # Dự đoán
        kmeans_pred = models['kmeans'].predict(new_scaled)[0]
        gmm_pred = models['gmm'].predict(new_scaled)[0]
        
        st.divider()
        st.subheader("Ket qua du doan")
        
        # Hiển thị kết quả chi tiết
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### KMeans")
            st.metric("Phan cum", f"Cum {kmeans_pred}")
            
            if kmeans_pred == 0:
                st.success("Phan khuc pho thong")
                st.write("""
                **Dac diem:**
                - Gia: ~6.5 ty
                - Dien tich: ~48 m²
                - Phu hop: Gia dinh tre, dau tu
                """)
            else:
                st.success("Phan khuc cao cap")
                st.write("""
                **Dac diem:**
                - Gia: ~20 ty
                - Dien tich: ~114 m²
                - Phu hop: Gia dinh lon, cao cap
                """)
        
        with col_b:
            st.markdown("### GMM")
            st.metric("Phan cum", f"Cum {gmm_pred}")
            st.warning("Do tin cay thap hon KMeans")
        
        # Thêm thông tin so sánh
        st.divider()
        st.subheader("Phan tich")
        
        # So sánh với giá trung bình
        avg_price = models['df_recommend']['gia_ban_num'].mean() / 1e9
        if gia > avg_price:
            st.info(f"Gia nhap ({gia} ty) cao hon gia trung binh thi truong ({avg_price:.1f} ty)")
        else:
            st.info(f"Gia nhap ({gia} ty) thap hon gia trung binh thi truong ({avg_price:.1f} ty)")
        
        # Gợi ý dựa trên kết quả
        if kmeans_pred == 0:
            st.success("Goi y: Day la phan khuc nha pho thong, phu hop voi nhu cau o hoac dau tu cho thue.")
        else:
            st.success("Goi y: Day la phan khuc nha cao cap, phu hop voi khach hang co tai chinh manh, tim kiem khong gian song rong rai.")
    
    except Exception as e:
        st.error(f"Loi khi du doan: {str(e)}")

# ==================== RECOMMENDATION ====================
elif menu == "De xuat bat dong san":
st.title("De xuat bat dong san")

# Chọn bất động sản
df = models['df_recommend']
df_display = df.head(100).copy()
df_display['display'] = df_display.apply(
    lambda x: f"{str(x['tieu_de'])[:45]}... - {x['gia_ban']}", axis=1
)

selected_idx = st.selectbox(
    "Chon bat dong san:",
    range(len(df_display)),
    format_func=lambda x: df_display.iloc[x]['display']
)

# Hiển thị chi tiết
with st.expander("Xem chi tiet", expanded=True):
    prop = df_display.iloc[selected_idx]
    st.write(f"**Tieu de:** {prop['tieu_de']}")
    st.write(f"**Gia:** {prop['gia_ban']} | **Dien tich:** {prop['dien_tich']} | **Quan:** {prop['quan']}")

n_recommend = st.slider("So luong de xuat:", 3, 10, 5)
rec_type = st.radio("Loai de xuat:", ["Hybrid", "Content-based"])

if st.button("De xuat", type="primary"):
    try:
        sim_matrix = models['hybrid_sim'] if rec_type == "Hybrid" else models['cosine_sim']
        
        sim_scores = list(enumerate(sim_matrix[selected_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommend+1]
        
        st.divider()
        st.subheader("Ket qua de xuat:")
        
        for i, (idx, score) in enumerate(sim_scores, 1):
            prop = df.iloc[idx]
            st.write(f"**{i}. {str(prop['tieu_de'])[:80]}...**")
            st.write(f"Gia: {prop['gia_ban']} | Dien tich: {prop['dien_tich']} | Quan: {prop['quan']}")
            st.write(f"Do tuong dong: {score:.3f}")
            st.divider()
    
    except Exception as e:
        st.error(f"Loi khi de xuat: {str(e)}")

# ==================== INFO TEAM ====================
elif menu == "Info Team":
st.title("Thong tin nhom")

st.markdown("""
**De tai:** He thong De xuat & Phan cum Bat dong san

**Thanh vien:**
| STT | Ho va ten | Cong viec |
|-----|-----------|-----------|
| 1 | Dang Duc Duy | Xu ly du lieu |
| 2 | Huynh Le Xuan Anh | Xay dung models He thong de xuat |
| 3 | Nguyen Thi Tuyet Van | Xay dung models he thong phan cum BDS |
            
**Cong nghe su dung:** 
- **Scikit-learn**: 
    - KMeans, GMM, Agglomerative Clustering
    - StandardScaler, PCA
- **Recommender System**:
    - TF-IDF Vectorizer
    - Cosine Similarity
    - Hybrid Recommender

**Ket qua dat duoc:**
- Phan cum BDS thanh 2 nhom chinh
- He thong de xuat voi 2 phuong phap
- Giao dien Streamlit truc quan
""")
