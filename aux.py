import json
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch
import wandb
import spacy
import evaluate
from tqdm import tqdm
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
id2label = {i: l for i, l in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}
target = [l for l in all_labels if l != "O"]
seqeval = evaluate.load("seqeval")
confusion_matrix = evaluate.load("confusion_matrix")


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
        attention_masks.append([1 for i in range(len(i['token_ids']))])

    dataset = Dataset.from_dict({'input_ids': token_ids, 'labels': tokenized_bios, 'source_text': source_texts, 'tokens': tokens, 'attention_mask': attention_masks})

    return dataset

def inference(model:AutoModelForTokenClassification, input_ids:torch.tensor, attention_mask:torch.tensor):
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

    return logits, predictions, predicted_token_class, inputs

def compute_metrics(predictions:list, labels:list):
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    conf_preds = []
    conf_labels = []
    for i, preds in enumerate(true_predictions):
        conf_preds += [label2id[i] for i in preds]
        conf_labels += [label2id[i] for i in true_labels[i]]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    confusion = confusion_matrix.compute(predictions=conf_preds, references=conf_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "confusion_matrix": confusion["confusion_matrix"]
    }

def compute_all_metrics(model:AutoModelForTokenClassification, data:Dataset):
    predictions = []
    labels = []
    for datum in tqdm(data, desc="Inference Progress"):
        logits, prediction, predicted_token_class, inputs = inference(model, torch.tensor([datum['input_ids']]), torch.tensor([datum['attention_mask']]))
        predictions.append(prediction.tolist()[0])
        labels.append(datum['labels'])
    
    return compute_metrics(predictions, labels)

def ensembler(output1, output2, words_ids1, word_ids2, labels):
    new_output1 = 

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