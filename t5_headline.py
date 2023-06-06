import os
from random import choices

import nltk
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load data
target_grades = ['V4']
for g in target_grades:
    dataset_name = f'target-grade-{g}-dedup'
    data_files = {"train": f"data/newsela_data/newsela-translated/{dataset_name}/train.jsonl",
                  "test": f"data/newsela_data/newsela-translated/{dataset_name}/test.jsonl",
                  "val": f"data/newsela_data/newsela-translated/{dataset_name}/val.jsonl"}
    newsela = load_dataset('json', data_files=data_files)

    # load metric
    metric = load_metric("rouge")

    # import tokenizer and model
    model_path = '/home/azagar/myfiles/t5/models/SloT5-sta_headline'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    indices = choices(list(range(len(data_files['test']))), k=25)
    for n in indices:
        sent = newsela["test"][n]['src_sent_sl']
        print('SOURCE:\t', sent)

        decoded_labels = newsela["test"][n]['tgt_sent_sl'].replace('\n\n', '')
        print("TARGET:\t", decoded_labels)

        input_ids = tokenizer(f"summarize: {sent}", return_tensors="pt", max_length=512, truncation=True).input_ids
        input_ids = input_ids.to("cpu")
        outputs = model.generate(input_ids, max_length=128)
        # outputs = model.generate(
        #     input_ids,
        #     do_sample=True,
        #     max_length=256,
        #     top_k=50
        # )
        decoded_preds = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("PREDICTION:\t", decoded_preds, '\n')

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in [decoded_preds]]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in [decoded_labels]]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Extract a few results
        rouge_results = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        print('ROUGE:', rouge_results)

        print('\n' * 3)
