import torch
import numpy as np
import transformers
from flask import Flask, render_template, request

from model import SentimentClassifier, class_names, tokenizer, MAX_LEN

app = Flask(__name__)

# load the model
FILE = "best_model_state.bin"
device = torch.device("cpu")
model = SentimentClassifier(len(class_names))
loaded_model = model.load_state_dict(torch.load(FILE, map_location=device))


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        review_text = inp
        encoded_review = tokenizer.encode_plus(
            review_text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_review['input_ids']
        attention_mask = encoded_review['attention_mask']

        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

        if class_names[prediction] == "positive":
            return render_template('home.html', message="Positive Review 😌😌")
        if class_names[prediction] == "negative":
            return render_template('home.html', message="Negative Review ☹😔")
        if class_names[prediction] == "neutral":
            return render_template('home.html', message="Neutral Review 👏👏👏")

    return render_template('home.html')
