from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from aux import json_to_Dataset
import evaluate
import numpy as np

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

n_labels = len(all_labels)

confusion_matrix = evaluate.load("confusion_matrix")
seqeval = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    confusion = confusion_matrix.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "confusion_matrix": confusion["confusion_matrix"]
    }


def get_trainer_from_model_name(our_name, model_name):

    model = AutoModelForTokenClassification.from_pretrained(
    model_name , num_labels=n_labels, id2label=id2label, label2id=label2id
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer = tokenizer)

    training_args = TrainingArguments(
    output_dir=our_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    remove_unused_columns= False,
    report_to= "tensorboard"
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= json_to_Dataset("/train.json"),
    eval_dataset=json_to_Dataset("/test.json"),
    tokenizer= tokenizer,
    data_collator= data_collator,
    compute_metrics= compute_metrics,
    )

    return trainer


if __name__ == "__main__":
    our_names = ["deberta_pii_finetuned", "distillbert_pii_finetuned"]
    models_names = ["microsoft/deberta-v2-xlarge", "distilbert-base-uncased"]

    for our_name, model_name in zip(our_names, models_names):
        trainer = get_trainer_from_model_name(our_name=our_name, model_name = model_name)
        trainer.train()