import sys

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update the top layout CSS
css_inject = """
st.set_page_config(
    page_title="AI Image Authenticity Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS to eliminate scrolling
st.markdown(\"""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
\""", unsafe_allow_html=True)
"""

# Replace the existing set_page_config
if "st.set_page_config" in content:
    idx_start = content.find("st.set_page_config")
    idx_end = content.find(")", idx_start) + 1
    content = content[:idx_start] + css_inject + content[idx_end:]


# 2. Move uploader to sidebar
uploader_code_old = """
uploaded = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)
"""
uploader_code_new = """
with st.sidebar:
    st.title("🛡️ AI Authenticity")
    st.markdown("Upload an image to detect if it's AI-generated or real.")
    uploaded = st.file_uploader(
        "Select Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )
    st.divider()
    with st.expander("Model Details"):
        st.markdown("**Architecture:** FusionDetector")
        st.markdown("**Backbone:** CLIP ViT-B/16 + 2D-FFT")
        st.markdown("**Threshold:** 0.5")
"""

content = content.replace(uploader_code_old.strip(), uploader_code_new.strip())

# Remove the old titles and expanders from main area to keep it clean
content = content.replace('st.title("AI Image Authenticity Detector")', "")
content = content.replace('st.markdown("**Determine whether an image is AI-generated or real.**")', "")
content = content.replace('st.divider()\n    \n    col_img, col_res = st.columns', 'col_img, col_res = st.columns')

# Adjust height of image container to 75vh to use space better
content = content.replace("height: 60vh;", "height: 80vh;")

# Remove the old technical expander at the bottom to eliminate scrolling
start_expander = content.find('with st.expander("')
if start_expander != -1:
    end_expander = content.find('else:', start_expander)
    if end_expander != -1:
        # Also remove the <br> before it
        br_idx = content.rfind('st.markdown("<br>", unsafe_allow_html=True)', 0, start_expander)
        if br_idx != -1:
            content = content[:br_idx] + content[end_expander:]

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Applied professional zero-scroll layout.")
