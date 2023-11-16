import os

import nltk
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

if __name__ == "__main__":
    data_files = {"train": f"data/target-grade-V4-split/train.jsonl",
                  "test": f"data/target-grade-V4-split/test.jsonl",
                  "val": f"data/target-grade-V4-split/dev.jsonl"}
    newsela = load_dataset('json', data_files=data_files)
    metric = load_metric("rouge")

    model_name = 'cjvt/t5-sl-large'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

    prefix = "simplify: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["src_sent_sl"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["tgt_sent_sl"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    tokenized_newsela = newsela.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}


    exp = f't5-sl-large-v4-maxlen128'
    output_dir = os.path.join("./results", exp)
    save_path = os.path.join('./model/baseline', exp)
    log_dir = os.path.join('./logs', exp)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        metric_for_best_model="rougeL", greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=3,
        num_train_epochs=3,
        fp16=False,
        save_steps=10000//32,
        eval_steps=10000//32,
        logging_steps=10000//32,
        logging_dir=log_dir,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        predict_with_generate=True,
        generation_max_length=128
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

    print(f"Evaluating model {model_name}")
    val_results = trainer.evaluate()
    test_results = trainer.predict(test_dataset=tokenized_newsela['test'])

    print('Val results: ', val_results)
    print('Test results:', test_results.metrics)
