from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import os

import nltk
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForLanguageModeling
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


target_grades = ['V4']
for g in target_grades:
    dataset_name = f'target-grade-{g}-dedup'
    data_files = {"train": f"data/newsela_data/newsela-translated/{dataset_name}/train.jsonl",
                  "test": f"data/newsela_data/newsela-translated/{dataset_name}/test.jsonl",
                  "val": f"data/newsela_data/newsela-translated/{dataset_name}/val.jsonl"}
    newsela = load_dataset('json', data_files=data_files)
    newsela['val'] = newsela['val'].select(list(range(25)))

    # load metric
    metric = load_metric("rouge")

    model_name = 'cjvt/gpt-sl-base'

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def preprocess_function(examples):
        return tokenizer([f"{x} simplify: {y} {tokenizer.eos_token}" for x, y in zip(examples["src_sent_sl"], examples["tgt_sent_sl"])], max_length=256, truncation=True)


    tokenized_newsela = newsela.map(preprocess_function, batched=True)

    # tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


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


    exp = f'{model_name.replace("/", "-")}-baseline-target-grade-{g}-eos-dedup'
    output_dir = os.path.join("./results", exp)
    save_path = os.path.join('./model/baseline', exp)
    log_dir = os.path.join('./logs', exp)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        evaluation_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        # fp16=False,
        save_steps=2500,
        eval_steps=1000,
        logging_steps=200,
        logging_dir=log_dir,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        # predict_with_generate=True,  # important for generation tasks
        # generation_max_length=128,  # import if using predict instead of generate
        # # generation_num_beams=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_newsela["train"],
        eval_dataset=tokenized_newsela["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(save_path)

    print(f"Evaluating model {model_name}")
    val_results = trainer.evaluate()
    test_results = trainer.predict(test_dataset=tokenized_newsela['test'])

    print('Val results: ', val_results)
    print('Test results:', test_results.metrics)