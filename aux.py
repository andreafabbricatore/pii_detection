import json
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch
import wandb
import spacy
import evaluate
nlp = spacy.load("en_core_web_sm")

all_labels = ['B-STREET',
 'B-CITY',
 'I-DATE',
 'B-PASS',
 'I-CITY',
 'B-TIME',
 'B-EMAIL',
 'I-DRIVERLICENSE',
 'I-POSTCODE',
 'I-BOD',
 'B-USERNAME',
 'B-BOD',
 'B-COUNTRY',
 'B-SECADDRESS',
 'B-IDCARD',
 'I-SOCIALNUMBER',
 'I-PASSPORT',
 'B-IP',
 'O',
 'B-TEL',
 'B-SOCIALNUMBER',
 'I-TIME',
 'B-BUILDING',
 'B-PASSPORT',
 'I-TITLE',
 'I-SEX',
 'I-STREET',
 'B-STATE',
 'I-STATE',
 'B-TITLE',
 'B-DATE',
 'B-GEOCOORD',
 'I-IDCARD',
 'I-TEL',
 'B-POSTCODE',
 'B-DRIVERLICENSE',
 'I-GEOCOORD',
 'I-COUNTRY',
 'I-EMAIL',
 'I-PASS',
 'B-SEX',
 'I-USERNAME',
 'I-BUILDING',
 'I-IP',
 'I-SECADDRESS',
 'B-CARDISSUER',
 'I-CARDISSUER']
seqeval = evaluate.load("seqeval")


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
    attention_masks = []
    for i in data:
        #ids.append(int(i['id']))
        tokens.append(i['tokens'])
        token_ids.append(i['token_ids'])
        tokenized_bios.append(i['bio_labels'])
        source_texts.append(i['source_text'])
        # attention_masks.append([1 for i in range(len(i['token_ids']))])

    dataset = Dataset.from_dict({'input_ids': token_ids, 'labels': tokenized_bios, 'source_text': source_texts, 'tokens': tokens})

    return dataset

def inference(model:AutoModelForTokenClassification, tokenizer:AutoTokenizer, text:str, is_albert:bool):
    if is_albert:
        docs = nlp(text)
        spacy_tokens = [token.text for token in docs]
        inputs = tokenizer(spacy_tokens, is_split_into_words=True, truncation=True, return_tensors="pt")
    else:
        inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

    return logits, predictions, predicted_token_class, inputs

def compute_metrics(model:AutoModelForTokenClassification, tokenizer:AutoTokenizer, data:dict, is_albert:bool):
    logits, predictions, predicted_token_class, inputs = inference(model, tokenizer, data['source_text'], is_albert)
    labels = data['labels']
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    #confusion = confusion_matrix.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        #"confusion_matrix": confusion["confusion_matrix"]
    }

def compute_all_metrics(model:AutoModelForTokenClassification, tokenizer:AutoTokenizer, data:Dataset):
    pass


def download_distilbert():
    run = wandb.init()
    print("API KEY: 501ec742d9174ae2b5538dbba9d348d69f98ec93")
    artifact = run.use_artifact('splenderai/<pii_detection>/model-1rbdb33p:v2', type='model')
    artifact_dir = artifact.download()

def download_distilbert2():
    run = wandb.init()
    print("API KEY: 501ec742d9174ae2b5538dbba9d348d69f98ec93")
    artifact = run.use_artifact('splenderai/<pii_detection>/model-vv5zcoip:v1', type='model')
    artifact_dir = artifact.download()

def download_albert():
    run = wandb.init()
    print("API KEY: 501ec742d9174ae2b5538dbba9d348d69f98ec93")
    artifact = run.use_artifact('splenderai/<pii_detection>/model-ugfz5xla:v2', type='model')
    artifact_dir = artifact.download()

def download_albert2():
    run = wandb.init()
    print("API KEY: 501ec742d9174ae2b5538dbba9d348d69f98ec93")
    artifact = run.use_artifact('splenderai/<pii_detection>/model-2eaz8bhc:v1', type='model')
    artifact_dir = artifact.download()