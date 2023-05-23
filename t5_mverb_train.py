import os

import nltk
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForSeq2Seq, MT5ForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer, MT5Tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


dataset_name = f'cckres-mverb-small'
data_files = {"train": f"data/{dataset_name}/train.jsonl",
              "test": f"data/{dataset_name}/test.jsonl",
              "val": f"data/{dataset_name}/val.jsonl"}
newsela = load_dataset('json', data_files=data_files)
# newsela['val'] = newsela['val'].select(list(range(25)))

# load metric
metric = load_metric("rouge")

model_name = 'cjvt/t5-sl-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# preprocess data
prefix = "correct: "  # no need for that in MT5


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["src_sent"]]
    model_inputs = tokenizer(inputs, max_length=24, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tgt_sent"], max_length=24, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_newsela = newsela.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    for s, p, l in zip(newsela['val']['src_sent_sl'][:25], decoded_preds[:25], decoded_labels[:25]):
        # print('Source:\t', s)
        print('Target:\t', l)
        print('Prediction:\t', p)
        print('\n' * 3)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


exp = f'{model_name.replace("/", "-")}-mverb-{dataset_name}'
output_dir = os.path.join("./results", exp)
save_path = os.path.join('./model/mverb', exp)
log_dir = os.path.join('./logs', exp)
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=False,
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    fp16=False,
    save_steps=10000,
    eval_steps=10000,
    logging_steps=100,
    logging_dir=log_dir,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    predict_with_generate=True,  # important for generation tasks
    generation_max_length=24,  # import if using predict instead of generate
    # generation_num_beams=4,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_newsela["train"],
    eval_dataset=tokenized_newsela["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(save_path)

print(f"Evaluating model {model_name}")
val_results = trainer.evaluate()
test_results = trainer.predict(test_dataset=tokenized_newsela['test'])

print('Val results: ', val_results)
print('Test results:', test_results.metrics)