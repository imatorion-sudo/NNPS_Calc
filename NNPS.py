import streamlit as st
import numpy as np\
import pydicom\
import matplotlib.pyplot as plt\
from scipy import optimize\
import io\
\
st.set_page_config(page_title="Wiener Spectrum Analyzer", layout="wide")\
\
st.title("NNPS\uc0\u35299 \u26512 \u12454 \u12455 \u12502 \u12450 \u12503 \u12522 \u12288 \u38306 \u26481 DR\u30740 \u31350 \u20250 \u12288 \u20170 \u33457 \'94)\
st.write("DICOM\uc0\u12501 \u12449 \u12452 \u12523 \u12434 \u12450 \u12483 \u12503 \u12525 \u12540 \u12489 \u12375 \u12390 \u12289 \u27491 \u35215 \u21270 \u12454 \u12451 \u12490 \u12540 \u12473 \u12506 \u12463 \u12488 \u12523 \u12434 \u31639 \u20986 \u12375 \u12414 \u12377 \u12290 ")\
\
# \uc0\u12469 \u12452 \u12489 \u12496 \u12540 \u12398 \u35373 \u23450 \
st.sidebar.header("\uc0\u35299 \u26512 \u12497 \u12521 \u12513 \u12540 \u12479 ")\
roi_size = st.sidebar.select_slider("ROI\uc0\u12469 \u12452 \u12474 ", options=[64, 128, 256], value=128)\
overlap = st.sidebar.checkbox("50%\uc0\u12458 \u12540 \u12496 \u12540 \u12521 \u12483 \u12503 \u12434 \u36969 \u29992 ", value=True)\
\
uploaded_file = st.file_uploader("DICOM\uc0\u12501 \u12449 \u12452 \u12523 \u12434 \u36984 \u25246 \u12375 \u12390 \u12367 \u12384 \u12373 \u12356 ", type=["dcm"])\
\
def remove_trend(roi):\
    y, x = np.indices(roi.shape)\
    def surface_model(data, a, b, c, d, e, f):\
        x, y = data\
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f\
    p0 = [0, 0, 0, 0, 0, np.mean(roi)]\
    try:\
        popt, _ = optimize.curve_fit(surface_model, (x.ravel(), y.ravel()), roi.ravel(), p0=p0)\
        return roi - surface_model((x, y), *popt)\
    except:\
        return roi - np.mean(roi)\
\
if uploaded_file is not None:\
    ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))\
    image = ds.pixel_array.astype(float)\
    pixel_spacing = float(ds.ImagerPixelSpacing[0]) if 'ImagerPixelSpacing' in ds else 0.1\
    \
    st.info(f"\uc0\u30011 \u20687 \u12469 \u12452 \u12474 : \{image.shape[1]\}x\{image.shape[0]\} / \u30011 \u32032 \u12469 \u12452 \u12474 : \{pixel_spacing\}mm")\
\
    if st.button("\uc0\u35299 \u26512 \u38283 \u22987 "):\
        with st.spinner('\uc0\u35336 \u31639 \u20013 ...'):\
            h, w = image.shape\
            avg_signal = np.mean(image)\
            step = roi_size // 2 if overlap else roi_size\
            \
            nps_accumulator = []\
            for y in range(0, h - roi_size, step):\
                for x in range(0, w - roi_size, step):\
                    roi = image[y:y+roi_size, x:x+roi_size]\
                    roi_detrended = remove_trend(roi)\
                    window = np.outer(np.hamming(roi_size), np.hamming(roi_size))\
                    fft_roi = np.fft.fftshift(np.fft.fft2(roi_detrended * window))\
                    w_norm = np.sum(window**2) / (roi_size**2)\
                    ps = (np.abs(fft_roi)**2) * (pixel_spacing**2) / (roi_size**2 * w_norm)\
                    nps_accumulator.append(ps)\
            \
            mean_nps = np.mean(nps_accumulator, axis=0)\
            nnps = mean_nps / (avg_signal**2)\
            freqs = np.fft.fftshift(np.fft.fftfreq(roi_size, d=pixel_spacing))\
            \
            # \uc0\u32080 \u26524 \u34920 \u31034 \
            col1, col2 = st.columns(2)\
            \
            with col1:\
                st.subheader("1\uc0\u27425 \u20803 \u12503 \u12525 \u12501 \u12449 \u12452 \u12523 ")\
                fig, ax = plt.subplots()\
                ax.loglog(freqs[roi_size//2+1:], nnps[roi_size//2, roi_size//2+1:], marker='o', markersize=3)\
                ax.set_xlabel("Frequency (cycles/mm)")\
                ax.set_ylabel("NNPS ($mm^2$)")\
                ax.grid(True, which="both", alpha=0.3)\
                st.pyplot(fig)\
            \
            with col2:\
                st.subheader("\uc0\u25968 \u20516 \u12487 \u12540 \u12479 ")\
                # CSV\uc0\u12480 \u12454 \u12531 \u12525 \u12540 \u12489 \u27231 \u33021 \
                csv_data = np.column_stack((freqs[roi_size//2:], nnps[roi_size//2, roi_size//2:]))\
                st.download_button(\
                    label="\uc0\u35299 \u26512 \u32080 \u26524 (CSV)\u12434 \u20445 \u23384 ",\
                    data=io.StringIO("Frequency,NNPS\\n" + "\\n".join([f"\{f\},\{n\}" for f, n in csv_data])).getvalue(),\
                    file_name="nnps_result.csv",\
                    mime="text/csv"\
                )}
