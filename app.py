import streamlit as st
import torch
import sentencepiece as spm
from model import initialize_transformer, translate, decode_sentence
import time
import random

# Set page configuration
st.set_page_config(
    page_title="Arabic to English Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"  # Change to collapsed to minimize sidebar
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .translation-box {
        border: 1px solid #1D4ED8;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #1E3A8A;
        color: white;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-weight: 500;
    }
    .example-card {
        padding: 1rem;
        background-color: #2563EB;  /* Consistent blue color for all examples */
        color: white;
        border-radius: 8px;
        margin-bottom: 1rem;  /* Increased margin between examples */
        cursor: pointer;
        transition: transform 0.2s;
        border: 1px solid #1D4ED8;  /* Added border */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Added shadow */
    }
    .example-card:hover {
        transform: scale(1.02);
        background-color: #1D4ED8;
    }
    .language-label {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .app-footer {
        text-align: center;
        color: #6B7280;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 2rem;  /* Increased vertical padding */
        font-weight: 600;
        border: none;
        width: 100%;
        margin: 1rem 0;  /* Added margin for better spacing */
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    /* Hide the sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'ar_en_transformer.pt'
MAX_SEQ_LEN = 200
VOCAB_SIZE = 8000
PAD, UNK, BOS, EOS = 0, 1, 2, 3

# Example phrases
EXAMPLES = [
    {"ar": "مرحبا بالعالم", "en": "Hello world"},
    {"ar": "كيف حالك اليوم؟", "en": "How are you today?"},
    {"ar": "أنا أحب اللغة العربية", "en": "I love the Arabic language"},
    {"ar": "هذا نموذج ترجمة آلية", "en": "This is a machine translation model"},
    {"ar": "ذهبت إلى السوق صباح اليوم", "en": "I went to the market this morning"},
    {"ar": "أريد أن أتعلم اللغة الإنجليزية بسرعة", "en": "I want to learn English quickly"},
    {"ar": "الصبر مفتاح الفرج", "en": "Patience is the key to relief"}
]

# Initialize tokenizers
def load_tokenizers():
    ar_sp = spm.SentencePieceProcessor()
    en_sp = spm.SentencePieceProcessor()
    ar_sp.load('ar_tokenizer.model')
    en_sp.load('en_tokenizer.model')
    return {
        "ar": ar_sp.encode_as_ids,
        "en": en_sp.encode_as_ids
    }, {
        "ar": ar_sp.decode_ids,
        "en": en_sp.decode_ids
    }

@st.cache_resource
def load_model():
    model = initialize_transformer(
        d_embed=256,
        feedforward_dim=512,
        num_attention_heads=8,
        n_encoder=4,
        n_decoder=4,
        dropout=0.1,
        max_seq_len=MAX_SEQ_LEN,
        encoder_vocab_size=VOCAB_SIZE,
        decoder_vocab_size=VOCAB_SIZE
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def translate_text(model, tokenizers, detokenizers, arabic_text):
    with torch.no_grad():
        # Tokenize input text
        src_tokens = torch.tensor([[BOS] + tokenizers["ar"](arabic_text) + [EOS]]).to(DEVICE)
        # Create source padding mask
        src_mask = (src_tokens == PAD).unsqueeze(1).unsqueeze(2).to(DEVICE)
        # Get translation
        translation = translate(model, src_tokens)
        # Decode translation
        english_text = decode_sentence(detokenizers["en"], translation[0])
    return english_text

# Animated loading function
def animated_loading():
    progress_text = "Translating... Please wait"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

# Initialize session state for storing the selected example
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""

# Streamlit UI
st.markdown('<h1 class="main-header">🌐 Arabic to English Translation</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Translate Arabic text to English using a Neural Machine Translation model</p>', unsafe_allow_html=True)

# Add model info in the main section instead of sidebar
with st.expander("ℹ️ About this translator"):
    st.markdown("""
    This application uses a **Transformer** model trained to translate Arabic text to English.
    
    **Features**:
    - Neural machine translation
    - Quick and efficient translation
    - Example phrases to try
    
    **Model Details**:
    - Architecture: Transformer
    - Encoder layers: 4
    - Decoder layers: 4
    - Attention heads: 8
    - Running on: """ + str(DEVICE))

# Main app content
try:
    model = load_model()
    tokenizers, detokenizers = load_tokenizers()
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="language-label">📝 Enter Arabic text:</p>', unsafe_allow_html=True)
        
        # Get the selected example first if there is one
        initial_text = st.session_state.selected_example if st.session_state.selected_example else ""
        
        # Create the text area with the initial value
        arabic_text = st.text_area("", 
                                  value=initial_text, 
                                  height=150, 
                                  placeholder="اكتب النص العربي هنا...", 
                                  key="arabic_input")
        
        st.markdown("### 📚 Examples")
        st.markdown("Click on any example to use it:")
        
        # Create example buttons that set the session state
        for i, example in enumerate(EXAMPLES):
            if st.button(f"{example['ar']} - {example['en']}", key=f"example_{i}"):
                st.session_state.selected_example = example['ar']
                st.rerun()
    
    with col2:
        st.markdown('<p class="language-label">🔄 Translation:</p>', unsafe_allow_html=True)
        
        # Create a placeholder for the translation
        translation_placeholder = st.empty()
        translation_placeholder.markdown('<div class="translation-box">Translation will appear here...</div>', unsafe_allow_html=True)
        
        # Create centered column for the translate button
        _, button_col, _ = st.columns([1, 2, 1])
        
        with button_col:
            translate_button = st.button("Translate ✨", use_container_width=True)
        
        if translate_button:
            if arabic_text.strip():
                # Show animated progress
                animated_loading()
                
                # Perform translation
                translation = translate_text(model, tokenizers, detokenizers, arabic_text)
                
                # Display the result with some animation effect
                translation_placeholder.markdown(f'<div class="translation-box">{translation}</div>', unsafe_allow_html=True)
                
                # Add a success message
                st.success("✅ Translation complete!")
                
                # Add a "Copy" button
                if st.button("📋 Copy Translation"):
                    st.code(translation)
                    st.info("Text copied to clipboard!")
            else:
                st.warning("⚠️ Please enter some Arabic text to translate.")
    
    # Footer
    st.markdown('<div class="app-footer"></div>', unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")