from transformers import MarianMTModel, MarianTokenizer

def load_model(src_lang="en", tgt_lang="fr"):
    if src_lang == tgt_lang:
        raise ValueError("Source and target languages must be different.")
    
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f"Model '{model_name}' not found on HuggingFace. Details: {e}")
    
    return model, tokenizer

def translate_text(text, model, tokenizer):
    if not text.strip():
        return "Input is empty."
    
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_ids = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    
    return translated_text
