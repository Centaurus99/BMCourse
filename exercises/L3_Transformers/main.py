from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np

dataset = load_dataset("glue", 'qnli')
metric = load_metric('glue', 'qnli')
tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')


def preprocess_function(examples):
    return tokenizer(examples['question'], examples['sentence'], max_length=512, truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    'google/bert_uncased_L-2_H-128_A-2', num_labels=2)

batch_size = 16

args = TrainingArguments(
    "tiny-bert-finetuned-qnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
