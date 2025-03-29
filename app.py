import os
import base64
import datetime
import streamlit as st
from google import genai
from google.genai import types
from pathlib import Path
import tempfile
import time
from PIL import Image

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ.get('GEMINI_API_KEY', '')

# Custom CSS for better styling
def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #3B82F6;
        font-weight: bold;
    }
    div.stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #2563EB;
    }
    .image-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .sidebar .block-container {
        padding-top: 2rem;
    }
    .success-message {
        padding: 0.75rem;
        background-color: #ECFDF5;
        color: #047857;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
    }
    .bot-message {
        background-color: #F3F4F6;
        border-left: 5px solid #6B7280;
    }
    </style>
    """, unsafe_allow_html=True)

def ensure_output_dir():
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_unique_filename(prompt):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a safe filename from the prompt (first 30 chars)
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
    return f"{timestamp}_{safe_prompt}.jpg"

def save_binary_file(file_path, data):
    with open(file_path, "wb") as f:
        f.write(data)
    return file_path

def is_valid_image(file_path):
    """Check if the file exists and is a valid image."""
    try:
        if not Path(file_path).exists():
            return False
        # Try opening the image with PIL to verify it's valid
        Image.open(file_path).verify()
        return True
    except:
        return False

def get_client():
    """Get Gemini client using the API key from session state"""
    if not st.session_state.api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        return None
    
    try:
        return genai.Client(api_key=st.session_state.api_key)
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

def generate(prompt):
    client = get_client()
    if not client:
        return None

    model = "gemini-2.0-flash-exp-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        response_modalities=["image", "text"],
        safety_settings=[
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
,
        response_mime_type="text/plain",
    )
    
    output_dir = ensure_output_dir()
    file_path = None
    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            file_name = generate_unique_filename(prompt)
            file_path = output_dir / file_name
            save_binary_file(
                file_path, chunk.candidates[0].content.parts[0].inline_data.data
            )
            # Wait briefly to ensure file is fully written
            time.sleep(0.5)
            if is_valid_image(file_path):
                st.success(f"Image saved to: {file_path}")
            else:
                st.error(f"Failed to save valid image at: {file_path}")
                file_path = None
        else:
            st.write(chunk.text)
    
    return file_path

def generate_from_image(image_file, prompt):
    client = get_client()
    if not client:
        return None
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(image_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Upload the image file to Gemini
        uploaded_file = client.files.upload(file=tmp_path)
        
        model = "gemini-2.0-flash-exp-image-generation"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_modalities=["image", "text"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_CIVIC_INTEGRITY",
                    threshold="OFF",
                ),
            ],
            response_mime_type="text/plain",
        )
        
        output_dir = ensure_output_dir()
        file_path = None
        
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            if chunk.candidates[0].content.parts[0].inline_data:
                file_name = generate_unique_filename(f"image_edit_{prompt}")
                file_path = output_dir / file_name
                save_binary_file(
                    file_path, chunk.candidates[0].content.parts[0].inline_data.data
                )
                st.success(f"Edited image saved to: {file_path}")
            else:
                st.write(chunk.text)
        
        return file_path
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def chat_with_ai(prompt):
    client = get_client()
    if not client:
        return "Error: Please provide a valid API key in the sidebar."
    
    model = "gemini-2.0-flash-exp-image-generation"
    
    # Include chat history for context
    contents = []
    for msg in st.session_state.chat_history:
        role = "user" if msg["is_user"] else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["text"])]
        ))
    
    # Add current prompt
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)]
    ))
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
    )
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    if hasattr(response, 'text'):
        return response.text
    elif hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'content'):
        if hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
    return "No response generated."

def display_image_safely(image_path, caption=None):
    """Display an image with proper error handling."""
    try:
        if not isinstance(image_path, (str, Path)):
            st.error("Invalid image path type")
            return False
        
        # Ensure path is a string and file exists
        path_str = str(image_path)
        if not Path(path_str).exists():
            st.error(f"Image file not found: {path_str}")
            return False
            
        # Try to open with PIL first to verify it's valid
        img = Image.open(path_str)
        # Convert to something streamlit can display directly
        st.image(img, caption=caption)
        return True
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        return False

def display_image_card(image_path, caption=None, prompt=None):
    """Display image in an attractive card format"""
    if not isinstance(image_path, (str, Path)):
        return False
    
    path_str = str(image_path)
    if not Path(path_str).exists():
        return False
    
    try:
        img = Image.open(path_str)
        
        st.markdown(f"""
        <div class="image-card">
            <h4>{caption or "Generated Image"}</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        if prompt:
            st.markdown(f"<p><strong>Prompt:</strong> {prompt}</p>", unsafe_allow_html=True)
        
        # Add download button for the image
        with open(path_str, "rb") as file:
            btn = st.download_button(
                label="Download Image",
                data=file,
                file_name=Path(path_str).name,
                mime="image/jpeg"
            )
        return True
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        return False

def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Sidebar for API Key and app info
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API key input
        api_key_input = st.text_input(
            "Enter Gemini API Key:",
            value='',
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
        
        st.markdown("---")
        st.info("💡 This app lets you generate and edit images using Gemini AI, as well as chat with the model.")
        
        # Add a clear history button in sidebar
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        st.markdown("---")
        
        # Creator information and social media links
        st.markdown("### 👨‍💻 Created by Pawan Kumar")
        
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 15px;">
            <a href="https://www.youtube.com/channel/UClgbj0iYh5mqY_81CMCw25Q/" target="_blank" style="display: flex; align-items: center; gap: 8px; text-decoration: none;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" width="20" style="margin-right: 5px;">
                YouTube Channel
            </a>
            <a href="https://www.instagram.com/p_awan__kumar/" target="_blank" style="display: flex; align-items: center; gap: 8px; text-decoration: none;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Instagram_icon.png/600px-Instagram_icon.png" width="20" style="margin-right: 5px;">
                Instagram
            </a>
            <a href="https://github.com/pawan941394/" target="_blank" style="display: flex; align-items: center; gap: 8px; text-decoration: none;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="margin-right: 5px;">
                GitHub
            </a>
            <a href="https://www.linkedin.com/in/pawan941394/" target="_blank" style="display: flex; align-items: center; gap: 8px; text-decoration: none;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/640px-LinkedIn_logo_initials.png" width="20" style="margin-right: 5px;">
                LinkedIn
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit and Gemini AI")
    
    # Main content
    st.title("🎨 AI Image Generator & Assistant")
    
    if not st.session_state.api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to use this app.")
        st.info("Don't have a key? Get one from [Google AI Studio](https://makersuite.google.com/)")
        return
    
    # Create tabs with better styling
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🖼️ Generate Image", "✏️ Edit Image"])

    # Tab 1: Chat Interface
    with tab1:
        st.header("Chat with AI")
        
        # Display chat history with improved styling
        for idx, message in enumerate(st.session_state.chat_history):
            message_type = "user" if message["is_user"] else "assistant"
            with st.chat_message(message_type):
                st.write(message["text"])
        
        # Chat input
        user_input = st.chat_input("Type a message...")
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"text": user_input, "is_user": True})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_ai(user_input)
                    st.write(response)
                    
                    # Add AI response to history
                    st.session_state.chat_history.append({"text": response, "is_user": False})

    # Tab 2: Generate Image with improved layout
    with tab2:
        st.header("Generate Images from Text")
        
        # Create a nice input area
        st.markdown("#### Enter your prompt below:")
        prompt = st.text_area(
            "",  # No label here as we're using the markdown header above
            placeholder="Describe the image you want to create...",
            value="create an image where tiger is dancing",
            height=100
        )
        
        # Use columns for button and settings
        col1, col2 = st.columns([1, 2])
        with col1:
            generate_button = st.button("🚀 Generate Image", use_container_width=True)
        
        # Generate image when button is clicked
        if generate_button:
            with st.spinner("🎨 Creating your masterpiece..."):
                image_path = generate(prompt)
                if image_path:
                    st.session_state.generated_image = image_path
                    st.session_state.current_prompt = prompt
        
        # Display generated image in a nice card
        if 'generated_image' in st.session_state and st.session_state.generated_image:
            st.markdown("### Your Generated Image:")
            display_image_card(
                st.session_state.generated_image, 
                caption="Generated Image", 
                prompt=st.session_state.get('current_prompt', prompt)
            )
        
        # Display gallery of previous images
        with st.expander("🖼️ Image Gallery", expanded=False):
            output_dir = ensure_output_dir()
            image_files = list(output_dir.glob("*.jpg"))
            if image_files:
                st.markdown(f"#### Found {len(image_files)} previously generated images")
                
                # Sort by creation time (newest first)
                image_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
                
                # Display in a grid
                cols = st.columns(3)
                for i, img_file in enumerate(image_files[:9]):  # Show only latest 9 images
                    with cols[i % 3]:
                        st.image(str(img_file), use_container_width=True)
                        timestamp = datetime.datetime.fromtimestamp(os.path.getctime(img_file))
                        st.caption(f"Created: {timestamp.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.info("No previously generated images found")

    # Tab 3: Edit Image with improved layout
    with tab3:
        st.header("Upload and Edit Images")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Upload your image")
            uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
            
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("#### Enter edit instructions")
            edit_prompt = st.text_area(
                "",
                placeholder="Describe how you want to edit the image...",
                value="Please color this beautifully using red, green, blue",
                height=100
            )
            
            # Only enable the button if an image is uploaded
            if uploaded_image is not None:
                if st.button("✨ Edit Image", use_container_width=True):
                    with st.spinner("🖌️ Editing your image..."):
                        edited_image_path = generate_from_image(uploaded_image, edit_prompt)
                        if edited_image_path:
                            st.session_state.edited_image = edited_image_path
        
        # Display edited image result
        if 'edited_image' in st.session_state and st.session_state.edited_image:
            st.markdown("### Your Edited Image:")
            display_image_card(st.session_state.edited_image, caption="Edited Image", prompt=edit_prompt)

if __name__ == "__main__":
    main()
