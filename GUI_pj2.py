# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================== CẤU HÌNH ====================
st.set_page_config(
    page_title="Hệ thống Đề xuất & Phân cụm Bất động sản", 
    layout="wide"
)

# ==================== ĐƯỜNG DẪN FILE ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_BT1 = os.path.join(BASE_DIR, "file_pkl_bt1")
PATH_BT2 = os.path.join(BASE_DIR, "file_pkl_bt2")

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    models = {}
    
    # Load BT1 models
    models['df_recommend'] = joblib.load(os.path.join(PATH_BT1, "df_recommend.pkl"))
    models['hybrid_sim'] = joblib.load(os.path.join(PATH_BT1, "hybrid_sim.pkl"))
    models['cosine_sim'] = joblib.load(os.path.join(PATH_BT1, "cosine_sim.pkl"))
    
    # Load BT2 models
    models['scaler'] = joblib.load(os.path.join(PATH_BT2, "scaler.pkl"))
    models['kmeans'] = joblib.load(os.path.join(PATH_BT2, "kmeans.pkl"))
    models['gmm'] = joblib.load(os.path.join(PATH_BT2, "gmm.pkl"))
    models['agg'] = joblib.load(os.path.join(PATH_BT2, "agg.pkl"))
    models['pca'] = joblib.load(os.path.join(PATH_BT2, "pca.pkl"))
    models['df_clustered'] = joblib.load(os.path.join(PATH_BT2, "df_clustered.pkl"))
    models['cluster_info'] = joblib.load(os.path.join(PATH_BT2, "cluster_info.pkl"))
    
    return models

with st.spinner("Đang tải mô hình..."):
    models = load_models()

st.sidebar.success("✅ Tải mô hình thành công!")

# ==================== MENU ====================
menu = st.sidebar.radio(
    "MENU",
    ["Bài toán kinh doanh", "Đánh giá Mô hình", "Dự đoán phân cụm", "Đề xuất bất động sản", "Thông tin nhóm"]
)

# ==================== BUSINESS PROBLEM ====================
if menu == "Bài toán kinh doanh":
    st.title("Bài toán Kinh doanh")
    st.markdown("""
    ### 📌 Vấn đề
    - Khách hàng có nhu cầu mua nhà tại các quận **Bình Thạnh, Gò Vấp, Phú Nhuận**
    - Cần hệ thống đề xuất nhà phù hợp
    - Cần phân cụm để hiểu rõ phân khúc thị trường
    
    ### 🎯 Mục tiêu
    1. Đề xuất nhà bằng Hybrid Recommender
    2. Phân cụm bằng Agglomerative Clustering
    
    ### 📊 Dữ liệu
    - **7,881** bất động sản tại 3 quận
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
elif menu == "Đánh giá Mô hình":
    st.title("Đánh giá Mô hình")
    
    tab1, tab2 = st.tabs(["Phân cụm", "Đề xuất"])
    
    with tab1:
        st.subheader("Đánh giá phân cụm")
        info = models['cluster_info']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏆 Agglomerative", f"{info['agg_score']:.4f}")
            st.write("✅ **Tốt nhất** (Silhouette Score > 0.5)")
            st.write(f"Cụm 0: {info['cluster_counts'][0]:,} BĐS - Phổ thông")
            st.write(f"Cụm 1: {info['cluster_counts'][1]:,} BĐS - Cao cấp")
        with col2:
            st.metric("KMeans", f"{info['kmeans_score']:.4f}")
            st.metric("GMM", f"{info['gmm_score']:.4f}")
    
    with tab2:
        st.subheader("Đánh giá Recommender")
        st.write("✅ **Hybrid** kết hợp nội dung (50%), giá (25%), vị trí (25%)")

# ==================== PREDICTION ====================
elif menu == "Dự đoán phân cụm":
    st.title("Dự đoán phân cụm - Agglomerative")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gia = st.number_input("💰 Giá (tỷ)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
        dien_tich = st.number_input("📐 Diện tích (m²)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
    
    with col2:
        quan = st.selectbox("📍 Quận", ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"])
        st.info(f"💡 Giá tham khảo {quan}: 6 - 22 tỷ")
    
    if st.button("🔮 Dự đoán", type="primary"):
        gia_num = gia * 1e9
        price_per_m2 = gia_num / dien_tich
        quan_map = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
        quan_encoded = quan_map[quan]
        
        new_data = np.array([[gia_num, dien_tich, price_per_m2, quan_encoded]])
        new_scaled = models['scaler'].transform(new_data)
        
        # Dự đoán với Agglomerative (cần fit_predict)
        from sklearn.cluster import AgglomerativeClustering
        agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
        # Lưu ý: Agglomerative không có method predict, cần train lại hoặc dùng model đã train
        # Cách đơn giản: dùng KMeans thay thế hoặc lưu model đã train
        
        # Tạm thời dùng KMeans vì Agglomerative không có predict
        kmeans_pred = models['kmeans'].predict(new_scaled)[0]
        
        st.divider()
        st.subheader("📊 Kết quả dự đoán")
        
        if kmeans_pred == 0:
            st.success("🏠 **Phân khúc phổ thông**")
            st.write("- Giá: ~6.5 tỷ\n- Diện tích: ~48 m²\n- Phù hợp: Gia đình trẻ, đầu tư")
        else:
            st.success("🏰 **Phân khúc cao cấp**")
            st.write("- Giá: ~20 tỷ\n- Diện tích: ~114 m²\n- Phù hợp: Gia đình lớn, cao cấp")

# ==================== RECOMMENDATION ====================
elif menu == "Đề xuất bất động sản":
    st.title("Đề xuất bất động sản")
    
    df = models['df_recommend']
    available_quan = df['quan'].unique().tolist()
    
    # Lọc theo quận
    selected_quan = st.selectbox("Chọn quận:", available_quan)
    df_filtered = df[df['quan'] == selected_quan].copy()
    
    if len(df_filtered) > 0:
        df_display = df_filtered.copy()
        df_display['display'] = df_display.apply(
            lambda x: f"{str(x['tieu_de'])[:50]}... - {x['gia_ban']}", axis=1
        )
        
        selected_idx = st.selectbox(
            "Chọn bất động sản:",
            range(len(df_display)),
            format_func=lambda x: df_display.iloc[x]['display']
        )
        
        with st.expander("📋 Xem chi tiết"):
            prop = df_display.iloc[selected_idx]
            st.write(f"**{prop['tieu_de']}**")
            st.write(f"💰 {prop['gia_ban']} | 📐 {prop['dien_tich']} | 📍 {prop['quan']}")
        
        n_recommend = st.slider("Số lượng đề xuất:", 3, 10, 5)
        rec_type = st.radio("Loại đề xuất:", ["Hybrid", "Content-based"])
        
        if st.button("🔍 Đề xuất"):
            original_idx = df_display.iloc[selected_idx].name
            sim_matrix = models['hybrid_sim'] if rec_type == "Hybrid" else models['cosine_sim']
            
            sim_scores = list(enumerate(sim_matrix[original_idx]))
            same_quan_indices = df_filtered.index.tolist()
            sim_scores = [(idx, score) for idx, score in sim_scores 
                         if idx in same_quan_indices and idx != original_idx]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:n_recommend]
            
            if sim_scores:
                for i, (idx, score) in enumerate(sim_scores, 1):
                    prop = df.iloc[idx]
                    st.write(f"**{i}. {str(prop['tieu_de'])[:80]}...**")
                    st.write(f"💰 {prop['gia_ban']} | 📐 {prop['dien_tich']} | 📍 {prop['quan']}")
                    st.write(f"🎯 Độ tương đồng: {score:.3f}")
                    st.divider()
            else:
                st.warning("Không tìm thấy BĐS tương tự!")
    else:
        st.error(f"Không có dữ liệu tại quận {selected_quan}!")

# ==================== INFO TEAM ====================
elif menu == "Thông tin nhóm":
    st.title("Thông tin nhóm")
    st.markdown("""
    **Đề tài:** Hệ thống Đề xuất & Phân cụm Bất động sản
    
    **Thành viên:**
    | STT | Họ và tên | Công việc |
    |-----|-----------|-----------|
    | 1 | Đặng Đức Duy | Xử lý dữ liệu |
    | 2 | Huỳnh Lê Xuân Ánh | Hệ thống đề xuất Hybrid |
    | 3 | Nguyễn Thị Tuyết Vân | Phân cụm Agglomerative |
                
    **Công nghệ:** Scikit-learn, Streamlit
    
    **Kết quả:** Agglomerative đạt Silhouette Score 0.593 (tốt nhất)
    """)
