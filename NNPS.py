import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import optimize
import io
import pandas as pd

# 1. ページ基本設定
st.set_page_config(page_title="Advanced NNPS Analyzer", layout="wide")

# 2. デザイン (CSS)
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: rgba(15, 23, 42, 0.9); border-right: 1px solid #334155; }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 8px; width: 100%; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4); }
    .developer-footer { font-family: 'Courier New', monospace; padding: 15px; border-radius: 10px; background: #0f172a; color: #38bdf8; border: 1px solid #38bdf8; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# 3. サイドバー作成者欄
st.sidebar.markdown(f"<div class='developer-footer'>ANALYSIS SYSTEM v3.0<br>----------<br>DEVELOPED BY:<br><strong>YOUR NAME</strong></div>", unsafe_allow_html=True)

# 4. 解析・読み込み設定
st.sidebar.header("📁 読み込み設定")
file_type = st.sidebar.radio("ファイル形式", ["DICOM", "Raw (Binary)"])

if file_type == "Raw (Binary)":
    width = st.sidebar.number_input("画像幅 (Width)", value=2048)
    height = st.sidebar.number_input("画像高さ (Height)", value=2048)
    dtype = st.sidebar.selectbox("データ型", ["uint16", "int16", "float32"], index=0)
    byte_order = st.sidebar.selectbox("バイト並び", ["Little Endian (<)", "Big Endian (>)"], index=0)
    pixel_spacing = st.sidebar.number_input("画素サイズ (mm)", value=0.1, format="%.4f")
else:
    roi_size = st.sidebar.select_slider("ROIサイズ", options=[64, 128, 256], value=128)

st.title("🏥 Multi-format NNPS Analyzer")

# --- 関数群 ---
def remove_trend(roi):
    y, x = np.indices(roi.shape)
    def surface_model(data, a, b, c, d, e, f):
        x, y = data
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
    p0 = [0, 0, 0, 0, 0, np.mean(roi)]
    try:
        popt, _ = optimize.curve_fit(surface_model, (x.ravel(), y.ravel()), roi.ravel(), p0=p0)
        return roi - surface_model((x, y), *popt)
    except:
        return roi - np.mean(roi)

# --- メイン処理 ---
uploaded_file = st.file_uploader("ファイルをアップロードしてください", type=["dcm", "raw", "bin", "img"])

if uploaded_file is not None:
    image = None
    
    # DICOM読み込み
    if file_type == "DICOM":
        ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
        image = ds.pixel_array.astype(float)
        pixel_spacing = float(ds.ImagerPixelSpacing[0]) if 'ImagerPixelSpacing' in ds else 0.1
    
    # Raw読み込み
    else:
        raw_data = uploaded_file.read()
        dt = np.dtype(dtype)
        dt = dt.newbyteorder('<' if "Little" in byte_order else '>')
        try:
            image = np.frombuffer(raw_data, dtype=dt).reshape((height, width)).astype(float)
        except Exception as e:
            st.error(f"Rawデータの展開に失敗しました。サイズ設定を確認してください: {e}")

    if image is not None:
        st.success(f"読み込み成功: {image.shape[1]}x{image.shape[0]} px")
        
        # ROIサイズ選択（Raw時も必要なのでここに配置）
        roi_size = st.select_slider("解析ROIサイズ", options=[64, 128, 256], value=128, key="main_roi")

        if st.button("RUN ANALYSIS"):
            with st.spinner('Analyzing...'):
                h, w = image.shape
                avg_signal = np.mean(image)
                step = roi_size // 2
                
                nps_accumulator = []
                for y in range(0, h - roi_size, step):
                    for x in range(0, w - roi_size, step):
                        roi = image[y:y+roi_size, x:x+roi_size]
                        roi_detrended = remove_trend(roi)
                        window = np.outer(np.hamming(roi_size), np.hamming(roi_size))
                        fft_roi = np.fft.fftshift(np.fft.fft2(roi_detrended * window))
                        w_norm = np.sum(window**2) / (roi_size**2)
                        ps = (np.abs(fft_roi)**2) * (pixel_spacing**2) / (roi_size**2 * w_norm)
                        nps_accumulator.append(ps)
                
                mean_nps = np.mean(nps_accumulator, axis=0)
                nnps_2d = mean_nps / (avg_signal**2)
                freqs = np.fft.fftshift(np.fft.fftfreq(roi_size, d=pixel_spacing))
                center = roi_size // 2
                freq_1d = freqs[center:]
                u_nnps = nnps_2d[center, center:]
                v_nnps = nnps_2d[center:, center]

                # グラフ表示
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("2D NNPS Map")
                    fig2, ax2 = plt.subplots(facecolor='#0f172a')
                    ax2.set_facecolor('#0f172a')
                    im = ax2.imshow(np.log10(nnps_2d + 1e-15), extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]], cmap='viridis')
                    ax2.tick_params(colors='white')
                    plt.colorbar(im)
                    st.pyplot(fig2)

                with c2:
                    st.subheader("u-v Axis Comparison")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=freq_1d[1:], y=u_nnps[1:], name='u-axis (H)', line=dict(color='#38bdf8')))
                    fig1.add_trace(go.Scatter(x=freq_1d[1:], y=v_nnps[1:], name='v-axis (V)', line=dict(color='#fb7185')))
                    fig1.update_layout(template="plotly_dark", xaxis_type="log", yaxis_type="log", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig1, use_container_width=True)

                # エクスポート
                df = pd.DataFrame({"Freq_lp_mm": freq_1d[1:], "u_NNPS": u_nnps[1:], "v_NNPS": v_nnps[1:]})
                st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), f"nps_{uploaded_file.name}.csv", "text/csv")
