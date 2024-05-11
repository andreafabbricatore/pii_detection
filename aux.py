import json
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch

def json_to_Dataset(filepath:str) -> Dataset:
    """
    Pass a .json filepath generated during the pipeline phase and get a Dataset file format for training and evaluation.
    """

    data = []
    with open(filepath) as f:
        data = json.load(f)

    ids = []
    tokens = []
    token_ids = []
    tokenized_bios = []
    source_texts = []
    for i in data:
        ids.append(i['id'])
        tokens.append(i['tokens'])
        token_ids.append(i['token_ids'])
        tokenized_bios.append(i['bio_labels'])
        source_texts.append(i['source_text'])

    dataset = Dataset.from_dict({'id': ids, 'tokens': tokens, 'token_ids': token_ids, 'tokenized_bios': tokenized_bios, 'source_texts':source_texts})

    return dataset

def inference(model:AutoModelForTokenClassification, tokenizer:AutoTokenizer, text:str):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

    return logits, predictions, predicted_token_class
