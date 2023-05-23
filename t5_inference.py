import os

import nltk
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForSeq2Seq, MT5ForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer, MT5Tokenizer
from random import choices
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

target_grades = ['V4']
for g in target_grades:
    dataset_name = f'target-grade-{g}-dedup'
    data_files = {"train": f"data/newsela_data/newsela-translated/{dataset_name}/train.jsonl",
                  "test": f"data/newsela_data/newsela-translated/{dataset_name}/test.jsonl",
                  "val": f"data/newsela_data/newsela-translated/{dataset_name}/val.jsonl"}
    newsela = load_dataset('json', data_files=data_files)

    # load metric
    metric = load_metric("rouge")

    model_name = f'model/baseline/cjvt-t5-sl-small-baseline-target-grade-{g}'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # preprocess data
    prefix = "simplify: "

    indices = choices(list(range(len(newsela['test']))), k=25)
    print(model_name)
    for n in indices:
        sent = newsela["test"][n]['src_sent_sl']
        print('SOURCE:\t', sent)

        decoded_labels = newsela["test"][n]['tgt_sent_sl'].replace('\n\n', '')
        print("TARGET:\t", decoded_labels)

        input_ids = tokenizer(f"{prefix} {sent}", return_tensors="pt", max_length=128, truncation=True).input_ids
        input_ids = input_ids.to("cpu")
        outputs = model.generate(input_ids,
                                 max_length=128,
                                 num_beams=5,
                                 do_sample=True,
                                 top_k=5,
                                 temperature=0.7
                                 # num_return_sequences=5
                                 )

        decoded_preds = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("PREDICTION:\t", decoded_preds, '\n')

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in [decoded_preds]]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in [decoded_labels]]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Extract a few results
        rouge_results = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        print('ROUGE:', rouge_results)


        # for i in list(range(len(outputs))):
        #     decoded_preds = tokenizer.decode(outputs[i], skip_special_tokens=True)
        #     print(f"PREDICTION {i}:\t", decoded_preds, '\n')
        #
        #     # Rouge expects a newline after each sentence
        #     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in [decoded_preds]]
        #     decoded_labels_list = ["\n".join(nltk.sent_tokenize(label.strip())) for label in [decoded_labels]]
        #
        #     result = metric.compute(predictions=decoded_preds, references=decoded_labels_list, use_stemmer=True)
        #
        #     # Extract a few results
        #     rouge_results = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        #     print('ROUGE:', rouge_results)
        #
        #     # print('\nCANDIDATES:\n')
        #     # for can in tokenizer.batch_decode(outputs, skip_special_tokens=True):
        #     #     print(can)

        print('\n'*3)
