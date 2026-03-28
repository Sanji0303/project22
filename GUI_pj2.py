# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

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
    """Load tất cả các model"""
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

# Load models
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
elif menu == "Đánh giá Mô hình":
    st.title("Đánh giá Mô hình")
    
    tab1, tab2 = st.tabs(["Clustering", "Recommendation"])
    
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
            st.write("✅ Tốt nhất")
    
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
elif menu == "Dự đoán phân cụm":
    st.title("Dự đoán phân cụm")
    
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
        gia_num = gia * 1e9
        price_per_m2 = gia_num / dien_tich
        quan_map = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
        quan_encoded = quan_map[quan]
        
        new_data = np.array([[gia_num, dien_tich, price_per_m2, quan_encoded]])
        new_scaled = models['scaler'].transform(new_data)
        
        kmeans_pred = models['kmeans'].predict(new_scaled)[0]
        gmm_pred = models['gmm'].predict(new_scaled)[0]
        
        st.divider()
        st.subheader("📊 Kết quả dự đoán")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 🎯 KMeans")
            st.metric("Phân cụm", f"Cụm {kmeans_pred}")
            if kmeans_pred == 0:
                st.success("🏠 Phân khúc phổ thông")
                st.write("""
                **Đặc điểm:**
                - Giá: ~6.5 tỷ
                - Diện tích: ~48 m²
                - Phù hợp: Gia đình trẻ, đầu tư
                """)
            else:
                st.success("🏰 Phân khúc cao cấp")
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
        
        st.divider()
        st.subheader("📈 Phân tích")
        
        avg_price = models['df_recommend']['gia_ban_num'].mean() / 1e9
        if gia > avg_price:
            st.info(f"💰 Giá nhập ({gia} tỷ) cao hơn giá trung bình ({avg_price:.1f} tỷ)")
        else:
            st.info(f"💰 Giá nhập ({gia} tỷ) thấp hơn giá trung bình ({avg_price:.1f} tỷ)")

# ==================== RECOMMENDATION ====================
elif menu == "Đề xuất bất động sản":
    st.title("Đề xuất bất động sản")
    
    df = models['df_recommend']
    
    # ==================== THỐNG KÊ DỮ LIỆU ====================
    st.subheader("📊 Thống kê dữ liệu")
    quan_stats = df['quan'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    for i, (quan, count) in enumerate(quan_stats.items()):
        if i == 0:
            with col1:
                st.metric(f"📍 {quan.upper()}", f"{count:,} BĐS")
        elif i == 1:
            with col2:
                st.metric(f"📍 {quan.upper()}", f"{count:,} BĐS")
        else:
            with col3:
                st.metric(f"📍 {quan.upper()}", f"{count:,} BĐS")
    
    st.divider()
    
    # ==================== PHẦN 1: TÌM KIẾM THEO NHU CẦU ====================
    st.subheader("🔍 Tìm kiếm bất động sản theo nhu cầu")
    
    with st.expander("📝 Nhập thông tin nhu cầu", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_quan = st.selectbox(
                "📍 Quận mong muốn:",
                options=["Tất cả"] + df['quan'].unique().tolist(),
                index=0
            )
        
        with col2:
            search_price_min = st.number_input(
                "💰 Giá tối thiểu (tỷ):",
                min_value=0.5,
                max_value=100.0,
                value=0.5,
                step=0.5
            )
            search_price_max = st.number_input(
                "💰 Giá tối đa (tỷ):",
                min_value=0.5,
                max_value=100.0,
                value=20.0,
                step=0.5
            )
        
        with col3:
            search_area_min = st.number_input(
                "📐 Diện tích tối thiểu (m²):",
                min_value=10.0,
                max_value=500.0,
                value=30.0,
                step=5.0
            )
            search_area_max = st.number_input(
                "📐 Diện tích tối đa (m²):",
                min_value=10.0,
                max_value=500.0,
                value=100.0,
                step=5.0
            )
        
        search_btn = st.button("🔍 Tìm kiếm", type="primary")
        
        if search_btn:
            # Lọc theo quận
            if search_quan != "Tất cả":
                search_df = df[df['quan'] == search_quan].copy()
            else:
                search_df = df.copy()
            
            # Lọc theo giá
            search_df = search_df[
                (search_df['gia_ban_num'] >= search_price_min * 1e9) &
                (search_df['gia_ban_num'] <= search_price_max * 1e9)
            ]
            
            # Lọc theo diện tích
            search_df = search_df[
                (search_df['dien_tich_num'] >= search_area_min) &
                (search_df['dien_tich_num'] <= search_area_max)
            ]
            
            if len(search_df) > 0:
                st.success(f"✅ Tìm thấy {len(search_df)} bất động sản phù hợp!")
                
                # Hiển thị kết quả tìm kiếm
                with st.expander("📋 Kết quả tìm kiếm", expanded=True):
                    search_display = search_df.head(10).copy()
                    search_display['display'] = search_display.apply(
                        lambda x: f"[{x['quan'].upper()}] {str(x['tieu_de'])[:60]}... - {x['gia_ban']} | {x['dien_tich']}",
                        axis=1
                    )
                    
                    for idx, row in search_display.iterrows():
                        st.write(f"**{idx+1}. {row['display']}**")
                    
                    if len(search_df) > 10:
                        st.info(f"Hiển thị 10/ {len(search_df)} kết quả. Vui lòng chọn BĐS trong danh sách bên dưới để xem chi tiết và đề xuất.")
            else:
                st.warning("⚠️ Không tìm thấy bất động sản phù hợp với nhu cầu của bạn. Vui lòng điều chỉnh lại thông tin!")
    
    st.divider()
    
    # ==================== PHẦN 2: CHỌN BĐS ĐỂ ĐỀ XUẤT ====================
    st.subheader("🏠 Chọn bất động sản để xem đề xuất")
    
    # Lấy danh sách quận thực tế từ dữ liệu
    available_quan = df['quan'].unique().tolist()
    
    # Nếu chỉ có 1 quận, hiển thị cảnh báo
    if len(available_quan) == 1:
        st.warning(f"⚠️ Dữ liệu hiện tại chỉ có quận {available_quan[0]}. Vui lòng kiểm tra lại file df_recommend.pkl!")
    
    selected_quan = st.selectbox(
        "Chọn quận:",
        options=available_quan,
        index=0
    )
    
    # Lọc theo quận đã chọn
    df_filtered = df[df['quan'] == selected_quan].copy()
    
    st.info(f"📊 Hiển thị {len(df_filtered)} bất động sản tại quận {selected_quan.upper()}")
    
    # Hiển thị danh sách BĐS
    if len(df_filtered) > 0:
        # Hiển thị tất cả BĐS
        df_display = df_filtered.copy()
        df_display['display'] = df_display.apply(
            lambda x: f"{str(x['tieu_de'])[:50]}... - {x['gia_ban']} | {x['dien_tich']}", 
            axis=1
        )
        
        selected_idx = st.selectbox(
            "Chọn bất động sản:",
            range(len(df_display)),
            format_func=lambda x: df_display.iloc[x]['display']
        )
        
        with st.expander("📋 Xem chi tiết", expanded=True):
            prop = df_display.iloc[selected_idx]
            st.write(f"**Tiêu đề:** {prop['tieu_de']}")
            st.write(f"**Giá:** {prop['gia_ban']} | **Diện tích:** {prop['dien_tich']} | **Quận:** {prop['quan']}")
        
        n_recommend = st.slider("Số lượng đề xuất:", 3, 10, 5)
        rec_type = st.radio("Loại đề xuất:", ["Hybrid", "Content-based"])
        
        if st.button("🔍 Đề xuất", type="primary"):
            # Lấy index gốc trong dataframe đầy đủ
            original_idx = df_display.iloc[selected_idx].name
            
            sim_matrix = models['hybrid_sim'] if rec_type == "Hybrid" else models['cosine_sim']
            
            # Lấy độ tương đồng
            sim_scores = list(enumerate(sim_matrix[original_idx]))
            
            # Chỉ lấy các BĐS trong cùng quận đã chọn
            same_quan_indices = df_filtered.index.tolist()
            sim_scores = [(idx, score) for idx, score in sim_scores 
                         if idx in same_quan_indices and idx != original_idx]
            
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:n_recommend]
            
            if len(sim_scores) == 0:
                st.warning("Không tìm thấy bất động sản tương tự trong cùng quận!")
            else:
                st.divider()
                st.subheader(f"🏠 Kết quả đề xuất tại quận {selected_quan.upper()}:")
                
                for i, (idx, score) in enumerate(sim_scores, 1):
                    prop = df.iloc[idx]
                    st.write(f"**{i}. {str(prop['tieu_de'])[:80]}...**")
                    st.write(f"💰 {prop['gia_ban']} | 📐 {prop['dien_tich']} | 📍 {prop['quan']}")
                    st.write(f"🎯 Độ tương đồng: {score:.3f}")
                    st.divider()
    else:
        st.error(f"❌ Không có dữ liệu bất động sản tại quận {selected_quan}!")

# ==================== INFO TEAM ====================
elif menu == "Thông tin nhóm":
    st.title("Thông tin nhóm")
    
    st.markdown("""
    **Đề tài:** Hệ thống Đề xuất & Phân cụm Bất động sản
    
    **Thành viên:**
    | STT | Họ và tên | Công việc |
    |-----|-----------|-----------|
    | 1 | Đặng Đức Duy | Xử lý dữ liệu |
    | 2 | Huỳnh Lê Xuân Ánh | Xây dựng models Hệ thống đề xuất |
    | 3 | Nguyễn Thị Tuyết Vân | Xây dựng models hệ thống phân cụm BĐS |
                
    **Công nghệ sử dụng:** 
    - **Scikit-learn**: KMeans, GMM, Agglomerative Clustering, StandardScaler, PCA
    - **Recommender System**: TF-IDF Vectorizer, Cosine Similarity, Hybrid Recommender
    
    **Kết quả đạt được:**
    - ✅ Phân cụm BĐS thành 2 nhóm chính
    - ✅ Hệ thống đề xuất với 2 phương pháp
    - ✅ Giao diện Streamlit trực quan
    """)
