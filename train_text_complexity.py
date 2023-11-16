import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Dict

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction


if __name__ == "__main__":
	PRETRAINED_NAME_OR_PATH = "EMBEDDIA/sloberta"
	BATCH_SIZE = 128
	DEV_BATCH_SIZE = int(1.5 * BATCH_SIZE)
	EVAL_EVERY_N_EXAMPLES = 10_000
	EVAL_EVERY_N_BATCHES = (EVAL_EVERY_N_EXAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
	MAX_LENGTH = 65
	LEARNING_RATE = 2e-5
	NUM_EPOCHS = 10
	EXPERIMENT_DIR = f"sloberta_slokit_maxlen{MAX_LENGTH}_{NUM_EPOCHS}e_lr{LEARNING_RATE}"
	RANDOM_SEED = 17

	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)

	os.makedirs(EXPERIMENT_DIR, exist_ok=True)
	ts = time.time()
	# Set up logging to file and stdout
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	for curr_handler in [logging.StreamHandler(sys.stdout),
						 logging.FileHandler(os.path.join(EXPERIMENT_DIR, f"train{ts}.log"))]:
		curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
		logger.addHandler(curr_handler)

	uniq_labels = ["not_V4", "V4"]
	id2label = {_i: _lbl for _i, _lbl in enumerate(uniq_labels)}
	label2id = {_lbl: _i for _i, _lbl in id2label.items()}
	num_classes = len(id2label)

	train_data = pd.read_json("data/target-grade-V4-split/train.jsonl", orient="records", lines=True)
	train_data = pd.concat((
		train_data[["doc_id", "pair_id", "src_sent_sl", "src_grade"]].rename(columns={"src_sent_sl": "sent", "src_grade": "labels"}),
		train_data[["doc_id", "pair_id", "tgt_sent_sl", "tgt_grade"]].rename(columns={"tgt_sent_sl": "sent", "tgt_grade": "labels"})
	), axis=0).reset_index(drop=True)
	train_data["labels"] = train_data["labels"].apply(lambda _lbl: label2id["V4" if _lbl == "V4" else "not_V4"])
	train_data = Dataset.from_pandas(train_data)

	dev_data = pd.read_json("data/target-grade-V4-split/dev.jsonl", orient="records", lines=True)
	dev_data = pd.concat((
		dev_data[["doc_id", "pair_id", "src_sent_sl", "src_grade"]].rename(columns={"src_sent_sl": "sent", "src_grade": "labels"}),
		dev_data[["doc_id", "pair_id", "tgt_sent_sl", "tgt_grade"]].rename(columns={"tgt_sent_sl": "sent", "tgt_grade": "labels"})
	), axis=0).reset_index(drop=True)
	dev_data["labels"] = dev_data["labels"].apply(lambda _lbl: label2id["V4" if _lbl == "V4" else "not_V4"])
	dev_data = Dataset.from_pandas(dev_data)

	test_data = pd.read_json("data/target-grade-V4-split/test.jsonl", orient="records", lines=True)
	test_data = pd.concat((
		test_data[["doc_id", "pair_id", "src_sent_sl", "src_grade"]].rename(columns={"src_sent_sl": "sent", "src_grade": "labels"}),
		test_data[["doc_id", "pair_id", "tgt_sent_sl", "tgt_grade"]].rename(columns={"tgt_sent_sl": "sent", "tgt_grade": "labels"})
	), axis=0).reset_index(drop=True)
	test_data["labels"] = test_data["labels"].apply(lambda _lbl: label2id["V4" if _lbl == "V4" else "not_V4"])
	test_data = Dataset.from_pandas(test_data)

	logging.info(f"Loaded {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")

	tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME_OR_PATH)

	def tokenize_function(examples):
		return tokenizer(examples["sent"], padding="max_length", max_length=MAX_LENGTH, truncation=True)

	COLUMNS_TO_REMOVE = train_data.column_names
	COLUMNS_TO_REMOVE.remove("labels")

	enc_train_data = train_data.map(tokenize_function, batched=True, remove_columns=COLUMNS_TO_REMOVE)
	enc_dev_data = dev_data.map(tokenize_function, batched=True, remove_columns=COLUMNS_TO_REMOVE)
	enc_test_data = test_data.map(tokenize_function, batched=True, remove_columns=COLUMNS_TO_REMOVE)

	train_distribution = {id2label[_lbl_int]: _count / len(train_data)
						  for _lbl_int, _count in Counter(train_data['labels']).most_common()}
	logging.info(f"Training distribution: {train_distribution}")

	model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_NAME_OR_PATH,
															   num_labels=num_classes,
															   id2label=id2label, label2id=label2id)

	training_args = TrainingArguments(
		output_dir=EXPERIMENT_DIR,
		do_train=True, do_eval=True, do_predict=True,
		per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=DEV_BATCH_SIZE,
		learning_rate=LEARNING_RATE,
		num_train_epochs=NUM_EPOCHS,
		logging_strategy="steps", logging_steps=EVAL_EVERY_N_BATCHES,
		save_strategy="steps", save_steps=EVAL_EVERY_N_BATCHES, save_total_limit=1,
		seed=RANDOM_SEED, data_seed=RANDOM_SEED,
		evaluation_strategy="steps", eval_steps=EVAL_EVERY_N_BATCHES,
		load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
		optim="adamw_torch",
		report_to="none",
	)
	accuracy_func = evaluate.load("accuracy")
	precision_func = evaluate.load("precision")
	recall_func = evaluate.load("recall")
	f1_func = evaluate.load("f1")

	def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, int]:
		pred_logits, ground_truth = eval_pred
		predictions = np.argmax(pred_logits, axis=-1)

		metrics = {}
		metrics.update(accuracy_func.compute(predictions=predictions, references=ground_truth))
		macro_f1 = 0.0
		for _lbl_int, _lbl_str in id2label.items():
			bin_preds = (predictions == _lbl_int).astype(np.int32)
			bin_ground_truth = (ground_truth == _lbl_int).astype(np.int32)

			curr = precision_func.compute(predictions=bin_preds, references=bin_ground_truth)
			curr[f"precision_{_lbl_str}"] = curr.pop("precision")
			curr.update(recall_func.compute(predictions=bin_preds, references=bin_ground_truth))
			curr[f"recall_{_lbl_str}"] = curr.pop("recall")
			curr.update(f1_func.compute(predictions=bin_preds, references=bin_ground_truth))
			curr[f"f1_{_lbl_str}"] = curr.pop("f1")
			macro_f1 += curr[f"f1_{_lbl_str}"]

			metrics.update(curr)

		metrics["f1_macro"] = macro_f1 / max(1, len(id2label))

		return metrics

	trainer = Trainer(
		model=model, args=training_args, tokenizer=tokenizer,
		train_dataset=enc_train_data, eval_dataset=enc_dev_data,
		compute_metrics=compute_metrics
	)

	train_metrics = trainer.train()

	test_metrics = trainer.predict(test_dataset=enc_test_data)
	logging.info(test_metrics.metrics)
	pred_probas = torch.softmax(torch.from_numpy(test_metrics.predictions), dim=-1)
	pred_class = torch.argmax(pred_probas, dim=-1)
	test_res = pd.DataFrame({
		"text": test_data["sent"],
		"pred_probas": pred_probas.tolist(),
		"pred_class": list(map(lambda _lbl_int: id2label[_lbl_int], pred_class.tolist()))
	})
	test_res.to_json(os.path.join(EXPERIMENT_DIR, "test_preds.json"), orient="records", lines=True)














