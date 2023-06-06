import os
from random import choices

import nltk
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load data
target_grades = ['V4']
inferences = defaultdict(list)
for g in target_grades:
    dataset_name = f'target-grade-{g}-dedup'
    data_files = {"train": f"data/newsela_data/newsela-translated/{dataset_name}/train.jsonl",
                  "test": f"data/newsela_data/newsela-translated/{dataset_name}/test.jsonl",
                  "val": f"data/newsela_data/newsela-translated/{dataset_name}/val.jsonl"}
    newsela = load_dataset('json', data_files=data_files)

    # load metric
    metric = load_metric("rouge")

    # import tokenizer and model
    headline_model_path = '/home/azagar/myfiles/t5/models/SloT5-sta_headline'
    headline_tokenizer = T5Tokenizer.from_pretrained(headline_model_path)
    headline_model = T5ForConditionalGeneration.from_pretrained(headline_model_path)

    # mverb
    mverb_model_name = f'/home/azagar/myfiles/slo-kit/model/mverb/cjvt-t5-sl-small-mverb-cckres-mverb'
    mverb_tokenizer = T5Tokenizer.from_pretrained(mverb_model_name)
    mverb_model = T5ForConditionalGeneration.from_pretrained(mverb_model_name)

    indices = choices(list(range(len(newsela['test']))), k=25)
    for n in indices:
        sent = newsela["test"][n]['src_sent_sl']
        print('SOURCE:\t', sent)

        decoded_labels = newsela["test"][n]['tgt_sent_sl'].replace('\n\n', '')
        print("TARGET:\t", decoded_labels)

        input_ids = headline_tokenizer(f"summarize: {sent}", return_tensors="pt", max_length=512, truncation=True).input_ids
        input_ids = input_ids.to("cpu")
        outputs = headline_model.generate(input_ids, max_length=128)

        decoded_preds_text = headline_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds_calc = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in [decoded_preds_text]]
        decoded_labels_calc = ["\n".join(nltk.sent_tokenize(label.strip())) for label in [decoded_labels]]

        result = metric.compute(predictions=decoded_preds_calc, references=decoded_labels_calc, use_stemmer=True)

        # Extract a few results
        rouge_results = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        print("HEADLINE-PREDICTION:\t", decoded_preds_text, '\tROUGE:', rouge_results)

        prefix = "correct: "
        input_ids = mverb_tokenizer(f"{prefix} {decoded_preds_text}", return_tensors="pt", max_length=32, truncation=True).input_ids
        input_ids = input_ids.to("cpu")
        outputs = mverb_model.generate(input_ids,
                                 max_length=32,
                                 num_beams=5,
                                 do_sample=True,
                                 top_k=5,
                                 temperature=0.7
                                 # num_return_sequences=5
                                 )

        decoded_preds_text = mverb_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds_calc = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in [decoded_preds_text]]
        decoded_labels_calc = ["\n".join(nltk.sent_tokenize(label.strip())) for label in [decoded_labels]]

        result = metric.compute(predictions=decoded_preds_calc, references=decoded_labels_calc, use_stemmer=True)

        # Extract a few results
        rouge_results = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        print("MVERB-PREDICTION:\t", decoded_preds_text, '\tROUGE:', rouge_results)

        print('\n'*3)
