import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import optimize
import io
import pandas as pd

# 1. ページ基本設定（一番最初に書く必要があります）
st.set_page_config(page_title="Multi-format NNPS Analyzer", layout="wide")

# --- 背景・デザインのカスタマイズ (CSS) ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: rgba(15, 23, 42, 0.9) !important; border-right: 1px solid #334155; }
    .developer-footer { font-family: 'Courier New', monospace; padding: 10px; border-radius: 10px; background: #0f172a; color: #38bdf8; border: 1px solid #38bdf8; text-align: center; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

# 2. サイドバーの設定（ここからサイドバーの記述）
with st.sidebar:
    st.markdown("<div class='developer-footer'>SYSTEM v3.0<br>DEVELOPED BY: YOUR NAME</div>", unsafe_allow_html=True)
    st.divider()
    
    st.header("⚙️ 読み込み設定")
    file_type = st.radio("ファイル形式を選択", ["DICOM", "Raw (Binary)"])
    
    if file_type == "Raw (Binary)":
        st.subheader("Raw Parameter")
        width = st.number_input("画像幅 (Width)", value=2048)
        height = st.number_input("画像高さ (Height)", value=2048)
        dtype_choice = st.selectbox("データ型", ["uint16", "int16", "float32"])
        byte_order = st.selectbox("バイト並び", ["Little Endian (<)", "Big Endian (>)"])
        raw_pixel_spacing = st.number_input("画素サイズ (mm)", value=0.1, format="%.4f")
    
    st.divider()
    roi_size = st.select_slider("解析ROIサイズ", options=[64, 128, 256], value=128)
    st.info("サイドバーで設定を行い、中央画面でファイルをアップロードしてください。")

# 3. メイン画面の表示
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
    except: return roi - np.mean(roi)

# --- ファイルアップロード処理 ---
uploaded_file = st.file_uploader("ファイルをアップロードしてください", type=["dcm", "raw", "bin", "img"])

if uploaded_file is not None:
    image = None
    pixel_spacing = 0.1 # デフォルト値
    
    # 読み込み処理
    try:
        if file_type == "DICOM":
            ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
            image = ds.pixel_array.astype(float)
            pixel_spacing = float(ds.ImagerPixelSpacing[0]) if 'ImagerPixelSpacing' in ds else 0.1
        else:
            raw_data = uploaded_file.read()
            dt = np.dtype(dtype_choice).newbyteorder('<' if "Little" in byte_order else '>')
            image = np.frombuffer(raw_data, dtype=dt).reshape((height, width)).astype(float)
            pixel_spacing = raw_pixel_spacing
            
        st.success(f"読み込み成功: {image.shape[1]}x{image.shape[0]} px / 画素サイズ: {pixel_spacing}mm")
        
        # プレビュー
        with st.expander("📷 画像プレビュー確認"):
            fig_p, ax_p = plt.subplots()
            ax_p.imshow(image, cmap='gray')
            ax_p.axis('off')
            st.pyplot(fig_p)

        # 4. 解析実行ボタン
        if st.button("RUN NNPS ANALYSIS"):
            with st.spinner('解析中...'):
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

                # グラフとデータ表示
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("2D NNPS Map")
                    fig2, ax2 = plt.subplots(facecolor='#0f172a')
                    ax2.set_facecolor('#0f172a')
                    im = ax2.imshow(np.log10(nnps_2d + 1e-15), extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]], cmap='viridis')
                    ax2.tick_params(colors='white')
                    st.pyplot(fig2)

                with c2:
                    st.subheader("u-v Axis Comparison")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=freq_1d[1:], y=u_nnps[1:], name='u-axis', line=dict(color='#38bdf8')))
                    fig1.add_trace(go.Scatter(x=freq_1d[1:], y=v_nnps[1:], name='v-axis', line=dict(color='#fb7185')))
                    fig1.update_layout(template="plotly_dark", xaxis_type="log", yaxis_type="log", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig1, use_container_width=True)

                # CSV保存
                df = pd.DataFrame({"Freq": freq_1d[1:], "u_NNPS": u_nnps[1:], "v_NNPS": v_nnps[1:]})
                st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "nnps_result.csv", "text/csv")

    except Exception as e:
        st.error(f"エラーが発生しました。設定（サイズやデータ型）を見直してください: {e}")
