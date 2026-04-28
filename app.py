import streamlit as st
import numpy as np
import tempfile
import cv2

from model import predict

st.set_page_config(page_title="Tumor Detection", layout="centered")

st.title(" Pituitary Tumor Detection System")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # ==============================
    # PREDICT
    # ==============================
    img, pred_mask = predict(temp_path)

    # ==============================
    # SAME AS COLAB
    # ==============================
    img_for_plot = np.transpose(img, (1, 2, 0))

    # Normalize for display
    img_for_plot = (img_for_plot - img_for_plot.min()) / (img_for_plot.max() - img_for_plot.min() + 1e-8)

    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # ==============================
    # REMOVE SMALL REGIONS
    # ==============================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    min_area_pixels = 200
    filtered_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_pixels:
            filtered_mask[labels == i] = 1

    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ==============================
    # AREA CALCULATION
    # ==============================
    pixel_area = np.sum(filtered_mask)
    mm_per_pixel = 0.5
    area_mm2 = pixel_area * (mm_per_pixel ** 2)

    # ==============================
    # OVERLAY (COLAB STYLE)
    # ==============================
    red_mask = np.zeros_like(img_for_plot)
    red_mask[:, :, 0] = filtered_mask

    overlay = (0.7 * img_for_plot + 0.3 * red_mask)

    # ==============================
    # CONTOUR IMAGE
    # ==============================
    contour_img = overlay.copy()

    for cnt in contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) == 2:
            for point in cnt:
                y, x = point[1], point[0]
                contour_img[y, x] = [0, 1, 0]

    # ==============================
    # DISPLAY (ROW-WISE)
    # ==============================
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original**")
        st.image(img_for_plot, width=250)

    with col2:
        st.markdown("**Mask**")
        st.image(filtered_mask * 255, width=250)

    with col3:
        st.markdown("**Overlay**")
        st.image(overlay, width=250)

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Tumor Boundary**")
        st.image(contour_img, width=300)

    with col5:
        st.markdown("**Tumor Area**")
        st.metric(label="Area (mm²)", value=f"{area_mm2:.2f}")