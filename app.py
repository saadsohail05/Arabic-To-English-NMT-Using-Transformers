import streamlit as st
import torch
import sentencepiece as spm
from model import initialize_transformer, translate, decode_sentence

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'ar_en_transformer.pt'
MAX_SEQ_LEN = 200
VOCAB_SIZE = 8000
PAD, UNK, BOS, EOS = 0, 1, 2, 3

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

# Streamlit UI
st.title("Arabic to English Translation")
st.write("This app translates Arabic text to English using a Transformer model.")

# Load model and tokenizers
try:
    model = load_model()
    tokenizers, detokenizers = load_tokenizers()
    
    # Text input
    arabic_text = st.text_area("Enter Arabic text:", height=150)
    
    if st.button("Translate"):
        if arabic_text.strip():
            with st.spinner("Translating..."):
                translation = translate_text(model, tokenizers, detokenizers, arabic_text)
                st.success("Translation complete!")
                st.write("English translation:")
                st.write(translation)
        else:
            st.warning("Please enter some Arabic text to translate.")
            
except Exception as e:
    st.error(f"Error loading model or tokenizers: {str(e)}")
    st.write("Please make sure all model files (ar_en_transformer.pt, ar_tokenizer.model, en_tokenizer.model) are present in the correct location.")