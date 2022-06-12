# ML-project--Sentiment-Classifier

This project is a basic sentiment classifier. It is a web-based application. It takes raw english sentence as input and generates one of the output as "Positive", "Negative" and "Neutral" sentiment. This sentiment classifier was created using pretrained BART-Base-cased model from huggingface library.

# Dependancy
1. We use Python 3.7.13 , pandas 1.3.5 and google_play_scraper: 1.0.3 versions.
2. Install google-play-scrapper from https://github.com/JoMingyu/google-play-scraper
3. "!pip install transformers" to install transformers.

## File Structure

<b>Sentiment_classification.ipynb</b> : This is jupiter notebook contains code for the model and the custom dataset. <br> 
<b> main.py </b>: This file contains python code for deploying website, Flask python framework was used. <br>
<b> model.py </b>: This file contains code for defining and loading the model. <br>

