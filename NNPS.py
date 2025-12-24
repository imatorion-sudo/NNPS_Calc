import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import optimize
import io
import pandas as pd

st.set_page_config(page_title="Advanced NNPS Analyzer", layout="wide")

st.title("NNPS解析ツール　関東DR研究会")
st.write("Ver1.0")

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

# ファイルアップローダー
uploaded_file = st.file_uploader("DICOMファイルを選択してください", type=["dcm"])

if uploaded_file is not None:
    ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
    image = ds.pixel_array.astype(float)
    
    # 画素サイズの取得
    if 'ImagerPixelSpacing' in ds:
        pixel_spacing = float(ds.ImagerPixelSpacing[0])
    elif 'PixelSpacing' in ds:
        pixel_spacing = float(ds.PixelSpacing[0])
    else:
        pixel_spacing = 0.1 # デフォルト
    
    st.sidebar.success(f"画像サイズ: {image.shape[1]}x{image.shape[0]}")
    st.sidebar.success(f"画素サイズ: {pixel_spacing} mm")
    
    roi_size = st.sidebar.select_slider("ROIサイズ", options=[64, 128, 256], value=128)

    if st.button("解析実行"):
        with st.spinner('高度な解析を実行中...'):
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
            nnps_1d = nnps_2d[center, center:] # 水平方向プロファイル

            # --- 表示セクション ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🖼️ 2D NNPS Map (Log Scale)")
                fig_2d, ax_2d = plt.subplots()
                # 0によるエラーを防ぐため微小値を加算してlog10
                im = ax_2d.imshow(np.log10(nnps_2d + 1e-15), 
                                 extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
                                 cmap='viridis')
                ax_2d.set_xlabel("u (cycles/mm)")
                ax_2d.set_ylabel("v (cycles/mm)")
                plt.colorbar(im, ax=ax_2d, label="log10(NNPS)")
                st.pyplot(fig_2d)

            with col2:
                st.subheader("📈 Interactive 1D Profile")
                # Plotlyによる対話型グラフ
                fig_1d = go.Figure()
                fig_1d.add_trace(go.Scatter(
                    x=freq_1d[1:], 
                    y=nnps_1d[1:],
                    mode='lines+markers',
                    name='Horizontal NNPS',
                    hovertemplate='周波数: %{x:.3f} lp/mm<br>NNPS: %{y:.2e}'
                ))
                fig_1d.update_xaxes(type="log", title="Spatial Frequency (cycles/mm)")
                fig_1d.update_yaxes(type="log", title="NNPS (mm^2)")
                fig_1d.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_1d, use_container_width=True)

            # --- データ出力セクション ---
            st.divider()
            st.subheader("💾 データエクスポート")
            
            # Pandasデータフレーム作成
            df_result = pd.DataFrame({
                "Frequency(cycles/mm)": freq_1d[1:],
                "NNPS(mm^2)": nnps_1d[1:]
            })
            
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="解析結果をCSVとしてダウンロード",
                data=csv,
                file_name=f"nnps_result_{uploaded_file.name}.csv",
                mime='text/csv',
            )
            st.dataframe(df_result, height=200) # 簡易テーブル表示

with st.sidebar.expander("About This Tool"):
    st.write("""
        本ツールは、デジタルX線画像における粒状性評価（NNPS）を
        客観的に行うために開発されました。
        - **Author:** Your Name
        - **Contact:** your-email@example.com
    """)
