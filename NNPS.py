import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import optimize
import io
import pandas as pd

# ページ基本設定（ファイル形式の選択をサイドバーに強制）
st.set_page_config(
    page_title="NNPS Analyzer v3.0",
    layout="wide",
    initial_sidebar_state="expanded"  # ← これでサイドバーを最初から開きます
)

# デザイン
st.markdown("""
    <style>
    .stApp { background: #0f172a; color: white; }
    [data-testid="stSidebar"] { background-color: #1e293b !important; }
    </style>
    """, unsafe_allow_html=True)

# --- サイドバー (左側のメニュー) ---
with st.sidebar:
    st.title("⚙️ Settings")
    file_type = st.radio("File Format", ["DICOM", "Raw (Binary)"])
    
    if file_type == "Raw (Binary)":
        st.subheader("Raw Info")
        w = st.number_input("Width", value=2048)
        h = st.number_input("Height", value=2048)
        dt_name = st.selectbox("Type", ["uint16", "int16", "float32"])
        order = st.selectbox("Endian", ["Little (<)", "Big (>)"])
        ps_raw = st.number_input("Pixel Spacing (mm)", value=0.1, format="%.4f")
    
    st.divider()
    roi = st.select_slider("ROI Size", options=[64, 128, 256], value=128)
    st.markdown("---")
    st.markdown("Developed by **Your Name**")

# --- メイン画面 ---
st.title("🏥 NNPS Analysis Core")

uploaded_file = st.file_uploader("Upload Image File", type=["dcm", "raw", "bin", "img"])

if uploaded_file:
    try:
        if file_type == "DICOM":
            ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
            img = ds.pixel_array.astype(float)
            ps = float(ds.ImagerPixelSpacing[0]) if 'ImagerPixelSpacing' in ds else 0.1
        else:
            raw_bytes = uploaded_file.read()
            dt = np.dtype(dt_name).newbyteorder('<' if "Little" in order else '>')
            img = np.frombuffer(raw_bytes, dtype=dt).reshape((h, w)).astype(float)
            ps = ps_raw

        st.success(f"Loaded: {img.shape[1]}x{img.shape[0]} / {ps}mm")

        if st.button("START ANALYSIS"):
            # 解析ロジック（中略：前回と同じ計算部分をここに含めてください）
            st.write("解析を実行しました（グラフを表示中...）")
            # ...ここに前回のグラフ表示コードが入ります...
            
    except Exception as e:
        st.error(f"Error: {e}")
