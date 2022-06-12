# ML-project--Sentiment-Classifier

This project is a basic sentiment classifier. It is a web-based application. It takes raw english sentence as input and generates one of the output as "Positive", "Negative" and "Neutral" sentiment. This sentiment classifier was created using pretrained BART-Base-cased model from huggingface library. This model was trained using a custom dataset. 

## Dependency
1. We use Python 3.7.13 , pandas 1.3.5 and google_play_scraper: 1.0.3 versions.
2. Install google-play-scrapper from https://github.com/JoMingyu/google-play-scraper
3. "!pip install transformers" to install transformers.

## File Structure

<b>Sentiment_classification.ipynb</b> : This is jupiter notebook contains code for the model and the custom dataset. <br> 
<b> main.py </b>: This file contains python code for deploying website, Flask python framework was used. <br>
<b> model.py </b>: This file contains code for defining and loading the model. <br>


## Training and evaluation 
 Download the jupyter notebook https://drive.google.com/file/d/13lJXnmBHHSl3KUZew9aocXK-CQXtgAWz/view?usp=sharing <br>
 Download the dataset https://drive.google.com/file/d/1rfDGHKfxHX7Ww2rXfqUwCWQ0_fgIXNFK/view?usp=sharing <br>
 Download the model https://drive.google.com/file/d/1b8UEmZuHtnHoqZdomxSHFz2Z48l3AnZy/view?usp=sharing <br>
 
 To train the model, load the model using "model.load_state_dict(torch.load('best_model_state.bin'))" and save the model using torch.save(model.state_dict(),path).
 Download the model from google drive and then execute the following codes: <br>
 
  
  
    def get_predictions(model, data_loader): \
    model = model.eval() \
    review_texts = []\
    predictions = []\
    prediction_probs = [] \
    real_values = [] \
    with torch.no_grad(): \
        for d in data_loader: \
            texts = d["review_text"] \
            input_ids = d["input_ids"].to(device) \
            attention_mask = d["attention_mask"].to(device) \
            targets = d["targets"].to(device) \
            outputs = model( \
                input_ids=input_ids, \
                attention_mask=attention_mask \
            )\
            _, preds = torch.max(outputs, dim=1) \
            review_texts.extend(texts) \
            predictions.extend(preds) \
            prediction_probs.extend(outputs) \
            real_values.extend(targets) \
    predictions = torch.stack(predictions).cpu() # converts list of tensors into single tensor \
    prediction_probs = torch.stack(prediction_probs).cpu() \
    real_values = torch.stack(real_values).cpu() \

    return review_texts, predictions, prediction_probs, real_values \
    
Run this code to do evaluation on test set only.


## Conclusion
The accuracy achieved on test set is 77.97%. Maybe this is because of the custom dataset that was created using google play store reviews. 
 
 
 
