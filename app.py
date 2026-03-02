import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import optimize
import io
import pandas as pd

# ページ基本設定
st.set_page_config(page_title="NNPS Analyzer v3.1", layout="wide", initial_sidebar_state="expanded")

# 背景デザイン (CSS)
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: rgba(15, 23, 42, 0.9) !important; border-right: 1px solid #334155; }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 8px; width: 100%; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4); }
    .developer-footer { font-family: 'Courier New', monospace; padding: 15px; border-radius: 10px; background: #0f172a; color: #38bdf8; border: 1px solid #38bdf8; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("<div class='developer-footer'>ANALYSIS SYSTEM v3.1<br>DEVELOPED BY: MASATO IMAHANA</div>", unsafe_allow_html=True)
    st.divider()
    st.header("⚙️ Settings")
    file_type = st.radio("File Format", ["DICOM", "Raw (Binary)"])
    
    # --- DICOM設定時の追加項目 ---
    gamma = 1.0
    if file_type == "DICOM":
        st.subheader("Characteristic Curve")
        gamma = st.number_input("特性曲線の傾き (Gradient: γ)", value=1.0, min_value=0.01, step=0.01, help="有効露光量変換に使用します。変換しない場合は1.0を入力してください。")
    
    if file_type == "Raw (Binary)":
        st.subheader("Raw Parameters")
        w = st.number_input("Width", value=2048)
        h = st.number_input("Height", value=2048)
        dt_choice = st.selectbox("Data Type", ["uint16", "int16", "float32"])
        order = st.selectbox("Byte Order", ["Little Endian (<)", "Big Endian (>)"])
        ps_raw = st.number_input("Pixel Spacing (mm)", value=0.1, format="%.4f")
    
    st.divider()
    roi_size = st.select_slider("ROI Size", options=[64, 128, 256], value=128)

st.title("NNPS Analyzer (Exposure Corrected)")

def remove_trend(roi):
    y, x = np.indices(roi.shape)
    def surface_model(data, a, b, c, d, e, f):
        x, y = data
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
    p0 = [0, 0, 0, 0, 0, np.mean(roi)]
    try:
        popt, _ = optimize.curve_fit(surface_model, (x.ravel(), y.ravel()), roi.ravel(), p0=p0)
        return roi - surface_model((x, y), *popt)
    except: return roi - np.mean(roi)

uploaded_file = st.file_uploader("Upload Image File", type=["dcm", "raw", "bin", "img"])

if uploaded_file:
    try:
        if file_type == "DICOM":
            ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
            img = ds.pixel_array.astype(float)
            ps = float(ds.ImagerPixelSpacing[0]) if 'ImagerPixelSpacing' in ds else 0.1
            
            # --- 有効露光量変換 (Relative Exposure Conversion) ---
            # NNPS解析の規約：デジタル値をγで除して「相対的な露光量変動」として扱う
            img = img / gamma
            st.info(f"有効露光量変換を適用しました (γ={gamma})")
            
        else:
            raw_data = uploaded_file.read()
            dt = np.dtype(dt_choice).newbyteorder('<' if "Little" in order else '>')
            img = np.frombuffer(raw_data, dtype=dt).reshape((h, w)).astype(float)
            ps = ps_raw

        st.success(f"Image Loaded: {img.shape[1]}x{img.shape[0]} / {ps}mm")
        
        with st.expander("📷 Preview Image"):
            fig_p, ax_p = plt.subplots(facecolor='#0f172a')
            ax_p.imshow(img, cmap='gray')
            ax_p.axis('off')
            st.pyplot(fig_p)

        if st.button("RUN ANALYSIS"):
            with st.spinner('Calculating NNPS...'):
                img_h, img_w = img.shape
                avg_signal = np.mean(img)
                step = roi_size // 2
                nps_accumulator = []
                
                # ROI extraction and NPS calculation
                for y in range(0, img_h - roi_size, step):
                    for x in range(0, img_w - roi_size, step):
                        roi_data = img[y:y+roi_size, x:x+roi_size]
                        roi_detrended = remove_trend(roi_data)
                        window = np.outer(np.hamming(roi_size), np.hamming(roi_size))
                        fft_roi = np.fft.fftshift(np.fft.fft2(roi_detrended * window))
                        w_norm = np.sum(window**2) / (roi_size**2)
                        ps_val = (np.abs(fft_roi)**2) * (ps**2) / (roi_size**2 * w_norm)
                        nps_accumulator.append(ps_val)
                
                mean_nps = np.mean(nps_accumulator, axis=0)
                # NNPSの定義：NPS / (平均信号値^2)
                nnps_2d = mean_nps / (avg_signal**2)
                
                freqs = np.fft.fftshift(np.fft.fftfreq(roi_size, d=ps))
                center = roi_size // 2
                freq_1d = freqs[center:]
                u_nnps = nnps_2d[center, center:]
                v_nnps = nnps_2d[center:, center]

                # 結果表示
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("2D NNPS Map")
                    fig2, ax2 = plt.subplots(facecolor='#0f172a')
                    im = ax2.imshow(np.log10(nnps_2d + 1e-15), extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]], cmap='viridis')
                    plt.colorbar(im)
                    st.pyplot(fig2)
                with c2:
                    st.subheader("u-v Axis Comparison")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=freq_1d[1:], y=u_nnps[1:], name='u-axis', line=dict(color='#38bdf8')))
                    fig1.add_trace(go.Scatter(x=freq_1d[1:], y=v_nnps[1:], name='v-axis', line=dict(color='#fb7185')))
                    fig1.update_layout(template="plotly_dark", xaxis_type="log", yaxis_type="log", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig1, use_container_width=True)

                df = pd.DataFrame({
                    "Frequency(lp/mm)": freq_1d[1:], 
                    "u_NNPS": u_nnps[1:], 
                    "v_NNPS": v_nnps[1:],
                    "Gamma_Used": gamma
                })
                st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "nnps_corrected_result.csv", "text/csv")
                
    except Exception as e:
        st.error(f"Analysis Error: {e}")
