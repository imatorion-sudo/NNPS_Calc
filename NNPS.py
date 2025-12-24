import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy import optimize
import io

st.set_page_config(page_title="NNPS Analyzer", layout="wide")

st.title("🏥 診療放射線技師向け：NNPS解析ツール")

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
    if 'ImagerPixelSpacing' in ds:
        pixel_spacing = float(ds.ImagerPixelSpacing[0])
    elif 'PixelSpacing' in ds:
        pixel_spacing = float(ds.PixelSpacing[0])
    else:
        pixel_spacing = 0.1
    
    st.info(f"画像サイズ: {image.shape[1]}x{image.shape[0]} / 画素サイズ: {pixel_spacing}mm")

    if st.button("解析開始"):
        with st.spinner('計算中...'):
            roi_size = 128
            h, w = image.shape
            avg_signal = np.mean(image)
            
            nps_accumulator = []
            for y in range(0, h - roi_size, roi_size // 2):
                for x in range(0, w - roi_size, roi_size // 2):
                    roi = image[y:y+roi_size, x:x+roi_size]
                    roi_detrended = remove_trend(roi)
                    window = np.outer(np.hamming(roi_size), np.hamming(roi_size))
                    fft_roi = np.fft.fftshift(np.fft.fft2(roi_detrended * window))
                    w_norm = np.sum(window**2) / (roi_size**2)
                    ps = (np.abs(fft_roi)**2) * (pixel_spacing**2) / (roi_size**2 * w_norm)
                    nps_accumulator.append(ps)
            
            mean_nps = np.mean(nps_accumulator, axis=0)
            nnps = mean_nps / (avg_signal**2)
            freqs = np.fft.fftshift(np.fft.fftfreq(roi_size, d=pixel_spacing))
            
            fig, ax = plt.subplots()
            ax.loglog(freqs[roi_size//2+1:], nnps[roi_size//2, roi_size//2+1:])
            ax.set_xlabel("Frequency (cycles/mm)")
            ax.set_ylabel("NNPS ($mm^2$)")
            ax.grid(True, which="both", alpha=0.3)
            st.pyplot(fig)
