import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import optimize
import io
import pandas as pd

# ページ設定
st.set_page_config(page_title="Advanced NNPS Analyzer", layout="wide")

# --- カスタムCSSで作成者欄を装飾 ---
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    .developer-footer {
        font-family: 'Courier New', Courier, monospace;
        padding: 10px;
        border-radius: 5px;
        background-color: #1e1e1e;
        color: #00ff00;
        text-align: center;
        border: 1px solid #333;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("NNPS解析ツール　関東DR研究会")
st.write("© 2026 NNPS解析ツール |  Copyright ©　関東DR研究会　　All Rights Reserved ")

# サイドバー：作成者情報
st.sidebar.markdown("""
    <div class='developer-footer'>
        SYSTEM VERSION 1.0<br>
        DEVELOPED BY:<br>
        [ Masato Imahana ]<br>
        RT / Image Engineering
    </div>
    """, unsafe_allow_html=True)

st.sidebar.divider()
roi_size = st.sidebar.select_slider("ROIサイズ", options=[64, 128, 256], value=128)
st.sidebar.info("2025.12.24　NNPS解析ツール　v1.0　リリース")
st.sidebar.info("2025.12.25　u軸とv軸の比較機能を搭載しました。")

# トレンド除去関数
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

uploaded_file = st.file_uploader("DICOMファイルを選択してください", type=["dcm"])

if uploaded_file is not None:
    ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
    image = ds.pixel_array.astype(float)
    
    # 画素サイズの取得
    pixel_spacing = float(ds.ImagerPixelSpacing[0]) if 'ImagerPixelSpacing' in ds else 0.1
    
    if st.button("解析開始"):
        with st.spinner('u軸/v軸の同時解析を実行中...'):
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
            
            # 周波数軸の設定
            freqs = np.fft.fftshift(np.fft.fftfreq(roi_size, d=pixel_spacing))
            center = roi_size // 2
            freq_1d = freqs[center:]
            
            # u軸(水平)とv軸(垂直)を抽出
            u_axis_nnps = nnps_2d[center, center:]
            v_axis_nnps = nnps_2d[center:, center]

            # --- 表示 ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🖼️ 2D NNPS Map (Log Scale)")
                fig_2d, ax_2d = plt.subplots()
                im = ax_2d.imshow(np.log10(nnps_2d + 1e-15), 
                                 extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
                                 cmap='viridis')
                ax_2d.set_xlabel("u (cycles/mm)")
                ax_2d.set_ylabel("v (cycles/mm)")
                plt.colorbar(im, ax=ax_2d)
                st.pyplot(fig_2d)

            with col2:
                st.subheader("📈 u-v Axis Comparison (Interactive)")
                fig_1d = go.Figure()
                # u軸（水平方向）
                fig_1d.add_trace(go.Scatter(x=freq_1d[1:], y=u_axis_nnps[1:], mode='lines+markers', name='u-axis (Horizontal)'))
                # v軸（垂直方向）
                fig_1d.add_trace(go.Scatter(x=freq_1d[1:], y=v_axis_nnps[1:], mode='lines+markers', name='v-axis (Vertical)'))
                
                fig_1d.update_xaxes(type="log", title="Spatial Frequency (cycles/mm)")
                fig_1d.update_yaxes(type="log", title="NNPS (mm^2)")
                fig_1d.update_layout(height=500, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
                st.plotly_chart(fig_1d, use_container_width=True)

            # --- データ出力 ---
            st.divider()
            df_result = pd.DataFrame({
                "Frequency(lp/mm)": freq_1d[1:],
                "u-axis_NNPS": u_axis_nnps[1:],
                "v-axis_NNPS": v_axis_nnps[1:]
            })
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(label="解析結果(CSV)を保存", data=csv, file_name="nnps_uv_result.csv", mime='text/csv')
            st.dataframe(df_result, height=200)

# フッター
st.caption("© 2026 Wiener Spectrum Analyzer Project | Created by Masato Imahana @Nihon Institute of Medical Science")
