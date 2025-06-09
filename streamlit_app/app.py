import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.translate import load_model, translate_text

st.title("🌍 Translator App")

src_lang = st.selectbox("From Language", ["en", "fr", "de", "es"])
tgt_lang = st.selectbox("To Language", ["en", "fr", "de", "es"])

text = st.text_area("Enter text to translate")

if st.button("Translate"):
    st.info("Loading model and translating...")
    try:
        tokenizer, model = load_model(src_lang, tgt_lang)
        result = translate_text(text, tokenizer, model)
        st.success("Translation complete!")
        st.markdown(f"**Translated Text:**\n\n{result}")
    except Exception as e:
        st.error("❌ An error occurred during translation.")
        st.text(str(e))
