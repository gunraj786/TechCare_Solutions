import streamlit as st
import os
import tempfile
import traceback
from typing import List

# Speech recognition imports
try:
    import speech_recognition as sr
    from gtts import gTTS
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    st.warning("Speech recognition libraries not available. Install SpeechRecognition and gTTS for voice features.")

from backend import get_response

# Page configuration
st.set_page_config(
    page_title="Gemini-like Chatbot", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp > header {
        background-color: transparent;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 0.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🤖 Interactive Gemini-like Chatbot</h1>', unsafe_allow_html=True)

# Initialize session state FIRST
if "messages" not in st.session_state:
    st.session_state.messages = []

if "listening" not in st.session_state:
    st.session_state.listening = False

# Initialize uploaded_files as None to avoid NameError
uploaded_files = None
enable_voice = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Voice settings
    if SPEECH_AVAILABLE:
        enable_voice = st.checkbox("🎤 Enable Voice Assistance", value=False)
        if enable_voice:
            st.info("Click 'Listen' button to speak, and responses will be read aloud.")
    else:
        enable_voice = False
        st.error("Voice features disabled - missing dependencies")
    
    # File upload for RAG
    st.subheader("📁 Upload Files for RAG")
    uploaded_files = st.file_uploader(
        "Choose files", 
        accept_multiple_files=True, 
        type=['txt', 'pdf', 'csv'],
        help="Upload documents to enhance responses with your content"
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} file(s) uploaded successfully!")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages with improved styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Voice input handling (improved with error handling)
def handle_voice_input():
    """Handle voice input with proper error handling and user feedback"""
    if not SPEECH_AVAILABLE:
        st.error("Speech recognition not available")
        return None
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            # Show listening indicator
            with st.spinner("🎤 Listening... Please speak now"):
                # Adjust for ambient noise
                r.adjust_for_ambient_noise(source, duration=0.5)
                # Set timeout and phrase time limit for better UX
                audio = r.listen(source, timeout=10, phrase_time_limit=10)
            
            # Show processing indicator
            with st.spinner("🔄 Processing your speech..."):
                prompt = r.recognize_google(audio)
                st.success(f"✅ You said: '{prompt}'")
                return prompt
                
    except sr.WaitTimeoutError:
        st.error("⏰ No speech detected within 10 seconds")
        return None
    except sr.UnknownValueError:
        st.error("🤔 Could not understand the audio clearly. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"🌐 Speech service error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"⚠ Unexpected error: {str(e)}")
        return None

# Voice output handling (improved)
def handle_voice_output(text: str):
    """Convert text to speech with error handling"""
    if not SPEECH_AVAILABLE or not text.strip():
        return
    
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            # Display audio player
            with open(tmp_file.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
    except Exception as e:
        st.warning(f"Could not generate speech: {str(e)}")

# Input section with voice option
col1, col2 = st.columns([4, 1])

with col1:
    prompt = st.chat_input("💬 Ask me anything...")

with col2:
    if SPEECH_AVAILABLE and enable_voice:
        if st.button("🎤 Listen", key="voice_button"):
            voice_prompt = handle_voice_input()
            if voice_prompt:
                prompt = voice_prompt

# Process user input
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process uploaded files for RAG - FIX: Check if uploaded_files exists and is not None
    file_paths = []
    if uploaded_files is not None and uploaded_files:
        try:
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                    tmp_file.write(file.getvalue())
                    file_paths.append(tmp_file.name)
        except Exception as e:
            st.error(f"Error processing uploaded files: {str(e)}")
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = get_response(prompt, file_paths)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Voice output if enabled
                if enable_voice and SPEECH_AVAILABLE:
                    handle_voice_output(response)
                    
            except Exception as e:
                error_msg = f"⚠ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                # Show detailed error in expander for debugging
                with st.expander("🔍 Debug Information"):
                    st.code(traceback.format_exc())
    
    # Clean up temporary files
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            st.warning(f"Could not clean up temp file {path}: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, LangChain, and Google Gemini | "
    "💡 Upload documents for enhanced RAG responses"
)