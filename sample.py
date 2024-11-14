import streamlit as st
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import re

# Load model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

@st.cache_data
def predict_next_word(text, top_k=5, temperature=1.0):
    # Trim trailing spaces to handle space bar prediction issues
    text = text.strip()
    
    # Use the last complete sentence for context, if possible
    last_sentence = re.split(r'[.!?]', text)[-1].strip() if text else text
    input_ids = tokenizer.encode(last_sentence, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model(input_ids)[0][:, -1, :] / temperature
    predictions = torch.topk(output, top_k, dim=-1)
    predicted_tokens = predictions.indices.tolist()[0]
    predicted_words = [tokenizer.decode(token).strip() for token in predicted_tokens]
    prediction_scores = predictions.values.tolist()[0]
    
    return dict(zip(predicted_words, prediction_scores))

def main():
    st.title(" Next Word Prediction")
    st.markdown("This app suggests the next word based on the input text. Adjust settings to customize your predictions.")

    text = st.text_input("Start typing:", value="Hello how")
    top_k = st.slider("Number of Predictions:", min_value=1, max_value=10, value=5)
    temperature = st.slider("Prediction Creativity (Temperature):", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    
    if text:
        next_word_predictions = predict_next_word(text, top_k, temperature)
        st.subheader("Predicted Next Words:")
        for word, score in next_word_predictions.items():
            st.write(f"**{word}** - Confidence: {score:.3f}")

if __name__ == "__main__":
    main()
