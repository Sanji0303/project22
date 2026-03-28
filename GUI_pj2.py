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

# Kiểm tra và hiển thị lỗi nếu có
path_issues = check_paths()
if path_issues:
    for issue in path_issues:
        st.error(issue)
    st.error("""
    Khong the tai ung dung do thieu file!
    
    Vui long dam bao cau truc thu muc:
