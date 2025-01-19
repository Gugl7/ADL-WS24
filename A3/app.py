import streamlit as st
from PIL import Image
import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from paddleocr import PaddleOCR
import numpy as np

def extract_text_from_image(image:Image) -> list:
    """Extracts text and coordinates from the image. Satisfies the necessary format for later usage in LayoutLM.

    Args:
        image (PIL.Image): The image to extract text from.

    Returns:
        list: A list of tuples containing coordinates and text.
    """
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    image = np.array(image)
    result = ocr.ocr(image, cls=True)
    coords = []
    txt = []

    for row in result[0]:
        coordinates = row[0][:2]
        text = row[1][0]
        # extracted_data.append((coordinates, text))
        coordinates = [item for sublist in coordinates for item in sublist]
        coords.append(coordinates)
        txt.append(text)
    
    return coords, txt

@st.cache_resource
def load_model() -> list:
    """Loads the model from the already predetermined path. The model is cached to avoid loading it multiple times.

    Returns:
        model: Model loaded from the path.
        tokenizer: Tokenizer loaded from the path.
    """
    model_path = "./layoutlm_model/"  # Update with your local path
    model = LayoutLMForTokenClassification.from_pretrained(model_path)
    tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def concatenate_words_by_label(words:list, labels: list) -> dict:
    """
    Concatenates words for each unique label.

    Args:
    - words (list of str): List of words.
    - labels (list of str): Corresponding list of labels.

    Returns:
    - dict: A dictionary where keys are labels, and values are concatenated strings of words for each label.
    """
    if len(labels) > len(words):
        labels = labels[:len(words)]
    if len(words) != len(labels):
        raise ValueError("The number of words and labels must be the same.")

    result = {}
    for word, label in zip(words, labels):
        if label not in result:
            result[label] = word
        else:
            result[label] += " " + word
    
    return result

def infer_single_image(image:Image, model_path:str, tokenizer_path:str, labels:list, device:str='cuda' if torch.cuda.is_available() else 'cpu') -> dict:
    """
    Perform inference on a single PIL.Image file with a pre-trained LayoutLM model.
    
    Args:
        image (PIL.Image): The input image.
        model_path (str): Path to the saved model checkpoint directory.
        tokenizer_path (str): Path to the tokenizer.
        labels (list): List of label strings.
        device (str): Device to run inference on ('cpu' or 'cuda').
        
    Returns:
        dict: Predicted entities with label and position information.
    """
    tokenizer = LayoutLMTokenizer.from_pretrained(tokenizer_path)
    model = LayoutLMForTokenClassification.from_pretrained(model_path)
    model.to(device).eval()
    
    _, text = extract_text_from_image(image)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    input_ids = input_ids[:, :len(text)]
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    label_map = {i: label for i, label in enumerate(labels)}
    pred_labels = [label_map[pred] for pred in predictions]
    
    return {'input_text': text, 'predicted_labels': pred_labels}

st.title('Receipt OCR Scanner')
st.write('Upload a receipt image to extract text using a LayoutLM model')

uploaded_file = st.file_uploader("Choose a receipt image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.divider()
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Receipt")
        st.text("Image:")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Extracted Text")
        with st.spinner('Loading model and performing OCR...'):
            model, tokenizer = load_model()
            results = infer_single_image(image, "./layoutlm_model/", "./layoutlm_model/", ["S-COMPANY", "S-DATE", "S-ADDRESS", "S-TOTAL", "O"], device='cuda')
            st.write("Predicted Labels:")
            st.json(concatenate_words_by_label(results['input_text'], results['predicted_labels']))
