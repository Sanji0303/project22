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
        issues.append(f"❌ Không tìm thấy thư mục: {PATH_BT1}")
    else:
        # Kiểm tra các file cần thiết trong BT1
        required_bt1 = ['df_recommend.pkl', 'hybrid_sim.pkl', 'cosine_sim.pkl']
        for file in required_bt1:
            if not os.path.exists(os.path.join(PATH_BT1, file)):
                issues.append(f"❌ Thiếu file: {file} trong thư mục file_pkl_bt1")
    
    if not os.path.exists(PATH_BT2):
        issues.append(f"❌ Không tìm thấy thư mục: {PATH_BT2}")
    else:
        # Kiểm tra các file cần thiết trong BT2
        required_bt2 = ['scaler.pkl', 'kmeans.pkl', 'gmm.pkl', 'agg.pkl', 
                        'pca.pkl', 'df_clustered.pkl', 'cluster_info.pkl']
        for file in required_bt2:
            if not os.path.exists(os.path.join(PATH_BT2, file)):
                issues.append(f"❌ Thiếu file: {file} trong thư mục file_pkl_bt2")
    
    return issues

""")
st.stop()

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
"""Load tất cả các model với xử lý lỗi"""
models = {}

try:
    # Load BT1 models
    with st.spinner("Đang tải models BT1..."):
        models['df_recommend'] = joblib.load(os.path.join(PATH_BT1, "df_recommend.pkl"))
        models['hybrid_sim'] = joblib.load(os.path.join(PATH_BT1, "hybrid_sim.pkl"))
        models['cosine_sim'] = joblib.load(os.path.join(PATH_BT1, "cosine_sim.pkl"))
    
    # Load BT2 models
    with st.spinner("Đang tải models BT2..."):
        models['scaler'] = joblib.load(os.path.join(PATH_BT2, "scaler.pkl"))
        models['kmeans'] = joblib.load(os.path.join(PATH_BT2, "kmeans.pkl"))
        models['gmm'] = joblib.load(os.path.join(PATH_BT2, "gmm.pkl"))
        models['agg'] = joblib.load(os.path.join(PATH_BT2, "agg.pkl"))
        models['pca'] = joblib.load(os.path.join(PATH_BT2, "pca.pkl"))
        models['df_clustered'] = joblib.load(os.path.join(PATH_BT2, "df_clustered.pkl"))
        models['cluster_info'] = joblib.load(os.path.join(PATH_BT2, "cluster_info.pkl"))
    
    return models
    
except Exception as e:
    st.error(f"❌ Lỗi khi tải model: {str(e)}")
    st.stop()

# Load models
with st.spinner("🔄 Đang tải mô hình..."):
models = load_models()

st.sidebar.success("✅ Tải mô hình thành công!")

# Hiển thị thông tin debug (có thể bỏ sau khi chạy ổn định)
with st.sidebar.expander("📁 Thông tin hệ thống"):
st.write(f"Thư mục gốc: `{BASE_DIR}`")
st.write(f"file_pkl_bt1: `{PATH_BT1}`")
st.write(f"file_pkl_bt2: `{PATH_BT2}`")
st.write(f"Python: `{sys.version}`")

# ==================== MENU ====================
menu = st.sidebar.radio(
"📋 MENU",
["🏢 Bài toán kinh doanh", "📊 Đánh giá Mô hình", "🎯 Dự đoán phân cụm", "🔍 Đề xuất bất động sản", "👥 Info Team"]
)

# ==================== BUSINESS PROBLEM ====================
if menu == "🏢 Bài toán kinh doanh":
st.title("🏢 Bài toán Kinh doanh")

st.markdown("""
### 📌 Vấn đề
- Khách hàng có nhu cầu mua nhà tại các quận **Bình Thạnh, Gò Vấp, Phú Nhuận**
- Cần hệ thống đề xuất nhà phù hợp
- Cần phân cụm để hiểu rõ phân khúc thị trường

### 🎯 Mục tiêu
1. Đề xuất nhà dựa trên nội dung (TF-IDF + Cosine)
2. Phân cụm bằng KMeans, GMM, Agglomerative
3. Hybrid Recommender (nội dung + giá + vị trí)

### 📊 Dữ liệu
- **7,881** bất động sản tại 3 quận
- Thuộc tính: giá, diện tích, số phòng, quận
- Mô tả chi tiết từ tin đăng
""")

col1, col2, col3 = st.columns(3)
df = models['df_recommend']
with col1:
    st.metric("Tổng số BĐS", f"{len(df):,}")
with col2:
    st.metric("Giá trung bình", f"{df['gia_ban_num'].mean()/1e9:.1f} tỷ")
with col3:
    st.metric("Diện tích TB", f"{df['dien_tich_num'].mean():.0f} m²")

# ==================== EVALUATION ====================
elif menu == "📊 Đánh giá Mô hình":
st.title("📊 Đánh giá Mô hình")

tab1, tab2 = st.tabs(["📈 Clustering", "🎯 Recommendation"])

with tab1:
    st.subheader("Đánh giá phân cụm")
    
    info = models['cluster_info']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("KMeans", f"{info['kmeans_score']:.4f}")
        st.write(f"Cụm 0: {info['cluster_counts'][0]:,} BĐS")
        st.write(f"Cụm 1: {info['cluster_counts'][1]:,} BĐS")
    
    with col2:
        st.metric("GMM", f"{info['gmm_score']:.4f}")
        st.write("Điểm thấp nhất")
    
    with col3:
        st.metric("Agglomerative", f"{info['agg_score']:.4f}")
        st.write("✅ **Tốt nhất**")

with tab2:
    st.subheader("Đánh giá Recommender")
    st.write("""
    **TF-IDF + Cosine:** Dựa trên nội dung mô tả
    
    **Hybrid:** Kết hợp 3 yếu tố
    - Nội dung (50%)
    - Giá (25%)
    - Vị trí (25%)
    
    **✅ Khuyến nghị: Hybrid**
    """)

# ==================== PREDICTION ====================
elif menu == "🎯 Dự đoán phân cụm":
st.title("🎯 Dự đoán phân cụm")

# Thêm thông tin hướng dẫn
with st.expander("📌 Hướng dẫn", expanded=False):
    st.write("""
    **Cách sử dụng:**
    1. Nhập giá bán và diện tích
    2. Chọn quận
    3. Nhấn "Dự đoán" để xem kết quả phân cụm
    
    **Giải thích kết quả:**
    - **Cụm 0:** Nhà phổ thông (giá ~6.5 tỷ, diện tích ~48m²)
    - **Cụm 1:** Nhà cao cấp (giá ~20 tỷ, diện tích ~114m²)
    """)

# Form nhập liệu
col1, col2 = st.columns(2)

with col1:
    gia = st.number_input("💰 Giá (tỷ)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
    dien_tich = st.number_input("📐 Diện tích (m²)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)

with col2:
    quan = st.selectbox("📍 Quận", ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"])
    gia_tham_khao = {
        "Bình Thạnh": "6.5 - 20 tỷ",
        "Gò Vấp": "5.5 - 18 tỷ",
        "Phú Nhuận": "6 - 22 tỷ"
    }
    st.info(f"💡 Giá tham khảo quận {quan}: {gia_tham_khao[quan]}")

if st.button("🔮 Dự đoán", type="primary"):
    try:
        # Tính toán
        gia_num = gia * 1e9
        price_per_m2 = gia_num / dien_tich
        quan_map = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
        quan_encoded = quan_map[quan]
        
        new_data = np.array([[gia_num, dien_tich, price_per_m2, quan_encoded]])
        new_scaled = models['scaler'].transform(new_data)
        
        # Dự đoán
        kmeans_pred = models['kmeans'].predict(new_scaled)[0]
        gmm_pred = models['gmm'].predict(new_scaled)[0]
        
        st.divider()
        st.subheader("📊 Kết quả dự đoán")
        
        # Hiển thị kết quả chi tiết
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 🎯 KMeans")
            st.metric("Phân cụm", f"Cụm {kmeans_pred}")
            
            if kmeans_pred == 0:
                st.success("🏠 **Phân khúc phổ thông**")
                st.write("""
                **Đặc điểm:**
                - Giá: ~6.5 tỷ
                - Diện tích: ~48 m²
                - Phù hợp: Gia đình trẻ, đầu tư
                """)
            else:
                st.success("🏰 **Phân khúc cao cấp**")
                st.write("""
                **Đặc điểm:**
                - Giá: ~20 tỷ
                - Diện tích: ~114 m²
                - Phù hợp: Gia đình lớn, cao cấp
                """)
        
        with col_b:
            st.markdown("### 📊 GMM")
            st.metric("Phân cụm", f"Cụm {gmm_pred}")
            st.warning("⚠️ Độ tin cậy thấp hơn KMeans")
        
        # Thêm thông tin so sánh
        st.divider()
        st.subheader("📈 Phân tích")
        
        # So sánh với giá trung bình
        avg_price = models['df_recommend']['gia_ban_num'].mean() / 1e9
        if gia > avg_price:
            st.info(f"💰 Giá nhập ({gia} tỷ) cao hơn giá trung bình thị trường ({avg_price:.1f} tỷ)")
        else:
            st.info(f"💰 Giá nhập ({gia} tỷ) thấp hơn giá trung bình thị trường ({avg_price:.1f} tỷ)")
        
        # Gợi ý dựa trên kết quả
        if kmeans_pred == 0:
            st.success("💡 **Gợi ý:** Đây là phân khúc nhà phổ thông, phù hợp với nhu cầu ở hoặc đầu tư cho thuê.")
        else:
            st.success("💡 **Gợi ý:** Đây là phân khúc nhà cao cấp, phù hợp với khách hàng có tài chính mạnh, tìm kiếm không gian sống rộng rãi.")
    
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {str(e)}")

# ==================== RECOMMENDATION ====================
elif menu == "🔍 Đề xuất bất động sản":
st.title("🔍 Đề xuất bất động sản")

# Chọn bất động sản
df = models['df_recommend']
df_display = df.head(100).copy()
df_display['display'] = df_display.apply(
    lambda x: f"{str(x['tieu_de'])[:45]}... - {x['gia_ban']}", axis=1
)

selected_idx = st.selectbox(
    "Chọn bất động sản:",
    range(len(df_display)),
    format_func=lambda x: df_display.iloc[x]['display']
)

# Hiển thị chi tiết
with st.expander("📋 Xem chi tiết", expanded=True):
    prop = df_display.iloc[selected_idx]
    st.write(f"**Tiêu đề:** {prop['tieu_de']}")
    st.write(f"**Giá:** {prop['gia_ban']} | **Diện tích:** {prop['dien_tich']} | **Quận:** {prop['quan']}")

n_recommend = st.slider("Số lượng đề xuất:", 3, 10, 5)
rec_type = st.radio("Loại đề xuất:", ["Hybrid", "Content-based"])

if st.button("🔍 Đề xuất", type="primary"):
    try:
        sim_matrix = models['hybrid_sim'] if rec_type == "Hybrid" else models['cosine_sim']
        
        sim_scores = list(enumerate(sim_matrix[selected_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommend+1]
        
        st.divider()
        st.subheader("🏠 Kết quả đề xuất:")
        
        for i, (idx, score) in enumerate(sim_scores, 1):
            prop = df.iloc[idx]
            st.write(f"**{i}. {str(prop['tieu_de'])[:80]}...**")
            st.write(f"💰 {prop['gia_ban']} | 📐 {prop['dien_tich']} | 📍 {prop['quan']}")
            st.write(f"🎯 Độ tương đồng: {score:.3f}")
            st.divider()
    
    except Exception as e:
        st.error(f"❌ Lỗi khi đề xuất: {str(e)}")

# ==================== INFO TEAM ====================
elif menu == "👥 Info Team":
st.title("👥 Thông tin nhóm")

st.markdown("""
**Đề tài:** Hệ thống Đề xuất & Phân cụm Bất động sản

**Thành viên:**
| STT | Họ và tên | Công việc |
|-----|-----------|-----------|
| 1 | Đặng Đức Duy | Xử lý dữ liệu |
| 2 | Huỳnh Lê Xuân Ánh | Xây dựng models Hệ thống đề xuất |
| 3 | Nguyễn Thị Tuyết Vân | Xây dựng models hệ thống phân cụm BĐS |
            
**Công nghệ sử dụng:** 
- **Scikit-learn**: 
    - KMeans, GMM, Agglomerative Clustering
    - StandardScaler, PCA
- **Recommender System**:
    - TF-IDF Vectorizer
    - Cosine Similarity
    - Hybrid Recommender

**Kết quả đạt được:**
- ✅ Phân cụm BĐS thành 2 nhóm chính
- ✅ Hệ thống đề xuất với 2 phương pháp
- ✅ Giao diện Streamlit trực quan
""")
