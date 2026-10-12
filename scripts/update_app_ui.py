import sys

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = "st.divider()"
start_idx = content.find(start_marker, content.find("col_img"))
# Wait, it's safer to find col_img, col_res = st.columns
if "col_img, col_res = st.columns" not in content:
    print("Could not find columns")
    sys.exit(1)

start_idx = content.find("    st.divider()", content.find("UI Layout: Side-by-side"))
if start_idx == -1:
    print("Could not find start idx")
    sys.exit(1)

end_idx = content.find("    # ", start_idx + 10)
if end_idx == -1:
    end_idx = content.find("else:", start_idx)

new_ui = """    st.divider()
    
    col_img, col_res = st.columns([1.5, 1], gap="large")

    with col_img:
        st.subheader("Analyzed Image")
        img_b64 = get_image_base64(raw_img)
        stamp_class = "stamp-fake" if is_fake else "stamp-real"
        stamp_text = "FAKE" if is_fake else "REAL"
        
        html_code = f\"\"\"
        <style>
        @keyframes stampAnimation {{
            0% {{ opacity: 0; transform: translate(-50%, -50%) scale(3) rotate(-15deg); }}
            50% {{ opacity: 1; transform: translate(-50%, -50%) scale(0.9) rotate(-15deg); }}
            100% {{ opacity: 0.85; transform: translate(-50%, -50%) scale(1) rotate(-15deg); }}
        }}
        @keyframes stampAnimationReal {{
            0% {{ opacity: 0; transform: translate(-50%, -50%) scale(3) rotate(10deg); }}
            50% {{ opacity: 1; transform: translate(-50%, -50%) scale(0.9) rotate(10deg); }}
            100% {{ opacity: 0.85; transform: translate(-50%, -50%) scale(1) rotate(10deg); }}
        }}
        .image-container {{
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: rgba(0,0,0,0.02);
            border-radius: 8px;
            padding: 10px;
            height: 60vh;
            overflow: hidden;
            border: 1px solid rgba(0,0,0,0.1);
        }}
        @media (prefers-color-scheme: dark) {{
            .image-container {{
                background-color: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.1);
            }}
        }}
        .uploaded-image {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 4px;
        }}
        .stamp {{
            position: absolute;
            top: 50%;
            left: 50%;
            font-size: clamp(3rem, 6vw, 6rem);
            font-weight: 900;
            text-transform: uppercase;
            border: clamp(5px, 1vw, 10px) solid;
            border-radius: 15px;
            padding: 10px 40px;
            pointer-events: none;
            box-shadow: 0 0 20px rgba(0,0,0,0.1), inset 0 0 20px rgba(0,0,0,0.1);
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            z-index: 10;
        }}
        .stamp-fake {{
            color: #ff3b30;
            border-color: #ff3b30;
            background-color: rgba(255, 59, 48, 0.1);
            animation: stampAnimation 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) 0.3s forwards;
            opacity: 0;
        }}
        .stamp-real {{
            color: #34c759;
            border-color: #34c759;
            background-color: rgba(52, 199, 89, 0.1);
            animation: stampAnimationReal 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) 0.3s forwards;
            opacity: 0;
        }}
        </style>
        
        <div class="image-container">
            <img src="data:image/jpeg;base64,{img_b64}" class="uploaded-image">
            <div class="stamp {stamp_class}">{stamp_text}</div>
        </div>
        \"\"\"
        st.markdown(html_code, unsafe_allow_html=True)

    with col_res:
        st.subheader("Analysis Result")
        
        if is_fake:
            st.error(f"### {{verdict}}\\n**Confidence: {{confidence:.1%}}**", icon="🤖")
        else:
            st.success(f"### {{verdict}}\\n**Confidence: {{confidence:.1%}}**", icon="📸")
            
        st.progress(prob_fake, text=f"P(AI-Generated) = {{prob_fake:.1%}}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**Model Attention / Grad-CAM**")
        img_cam_b64 = get_image_base64(overlay_img)
        st.markdown(
            f'<div style="height: 25vh; display: flex; justify-content: left;">'
            f'<img src="data:image/jpeg;base64,{img_cam_b64}" style="max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 8px;">'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption("ℹ️ *Grad-CAM highlights regions influencing the model's prediction.*")
        
"""

new_content = content[:start_idx] + new_ui + content[end_idx:]

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
