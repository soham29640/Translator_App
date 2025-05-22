import streamlit as st
from src.translate import load_model, translate_text

st.title("🌍 AI Translator App")
st.write("Translate text between languages using Hugging Face models.")

text = st.text_area("Enter text to translate:", "Hello, how are you?")
src_lang = st.selectbox("From Language", ["en", "de", "es", "fr", "it"])
tgt_lang = st.selectbox("To Language", ["fr", "en", "de", "es", "it"])

if st.button("Translate"):
    if src_lang == tgt_lang:
        st.warning("Please choose different source and target languages.")
    else:
        try:
            with st.spinner("Translating..."):
                model, tokenizer = load_model(src_lang, tgt_lang)
                output = translate_text(text, model, tokenizer)
                st.success("Translation complete!")
                st.text_area("Translated Text", output, height=150)
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Unexpected error: {e}")
