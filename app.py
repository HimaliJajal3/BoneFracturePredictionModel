import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import math
from streamlit_extras.metric_cards import style_metric_cards



# st.set_page_config(layout="wide")
style_metric_cards(border_left_color = "#365370", box_shadow=False)

st.markdown(
    """
    <style>
        .stApp {
            max-width: 1200px;
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Bone Fracture Detection & Analysis")
# st.write("---")
st.write("Enter Patient Details & Upload X-ray")

# Initialize session state for storing submissions
if "submissions" not in st.session_state:
    st.session_state.submissions = []  # Stores last 10 submissions

# Expander for input
with st.popover("Enter Patient Details", use_container_width=True):
    name = st.text_input("Patient name")
    uploaded = st.file_uploader("Upload Patient X-ray", type=["jpg", "jpeg", "png"])

    # Submit Button to store data in session state
    if name and uploaded and st.button("Submit"):
        # Store submission (Limit to last 10)
        st.session_state.submissions.insert(0, (name, uploaded))
        st.session_state.submissions = st.session_state.submissions[:5]  # Keep only last 10
        st.rerun()  # Refresh UI to update state


# st.write("---")
col1, col2, col3 = st.columns([2,2,1])
# col4, col5, col6 = st.columns([2,2,1])

# Load YOLO model
model = YOLO('best.pt')
FIXED_WIDTH = 500

if st.session_state.submissions:

    latest_name, latest_image = st.session_state.submissions[0]

    # Convert image
    image = Image.open(latest_image).convert("RGB")
    image_np = np.array(image).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # YOLO Prediction
    results = model.predict(image_bgr, imgsz=(640,640), save_conf=False)

    annotated_image = results[0].plot()
    for r in results:
        conf = torch.mean(r.boxes.conf).item()
        x = round(conf, 3)
        conf = '---' if math.isnan(conf) else f"{ x * 100}%"

    # st.write(results)



    # if results[0].boxes is not None and len(results[0].boxes) != 0:
    #     for box in results[0].boxes.xyxy:
    #         x_min, y_min, x_max, y_max = map(int, box.tolist())

    #         # Create a blank red mask
    #         mask = np.zeros_like(image_bgr, dtype=np.uint8)
    #         cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)  # Red rectangle

    #         # Apply Gaussian Blur for spreading effect
    #         mask = cv2.GaussianBlur(mask, (91, 91), 50)

    #         # Overlay mask on original image with transparency
    #         alpha = 0.5  # Transparency level
    #         image_bgr = cv2.addWeighted(image_bgr, 1, mask, alpha, 0)

    # Convert back to RGB for Streamlit
    masked_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Column 1: Patient Details & Images
    with col1:
        # st.button(f"Patient\n# **{name}**", use_container_width=True)
        st.metric(label="Patient Name:", value=f'{name}', label_visibility="visible", border=True )
        # st.write("---")
        st.write("**Uploaded X-ray**")
        st.image(latest_image, width=FIXED_WIDTH)
        # st.write("---")
        # st.write("Masked Image")
        # st.image(masked_image)


    # Column 2: Detection & Processed Image
    with col2:
        if_detected = results[0].boxes is not None and len(results[0].boxes) != 0
        # st.button(f"Fracture Detected\n# **{'Yes' if if_detected else 'No'}**", use_container_width=True)
        st.metric(label="Fracture Identified:", value=f"{'Yes' if if_detected else 'No'}", border=True)
        # st.write("---")
        st.write("**Detection Output**")
        st.image(annotated_image, width=FIXED_WIDTH)



    # Column 3: Show Last 10 Submissions
    with col3:
        if conf:
            # st.button(f"Avg Prediction Accuracy\n# **{conf}**", use_container_width=True)
            st.metric(label="Prediction Confidence:", value=f"{conf}",  border=True)

        st.write("---")
        st.write("**Download Report**")
        st.button("Download PDF")
        st.write("**Previous Diagnosis**")
        if len(st.session_state.submissions) > 0:
            for i, (submitted_name, _) in enumerate(st.session_state.submissions):
                st.button(f"{i+1}. {submitted_name}")
        else:
            st.info("No submissions yet.")
