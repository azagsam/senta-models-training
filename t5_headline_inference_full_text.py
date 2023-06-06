import os
from random import choices

import nltk
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load data

# import tokenizer and model
headline_model_path = '/home/azagar/myfiles/t5/models/SloT5-sta_headline'
headline_tokenizer = T5Tokenizer.from_pretrained(headline_model_path)
headline_model = T5ForConditionalGeneration.from_pretrained(headline_model_path)

# mverb
mverb_model_name = f'/home/azagar/myfiles/slo-kit/model/mverb/cjvt-t5-sl-small-mverb-cckres-mverb'
mverb_tokenizer = T5Tokenizer.from_pretrained(mverb_model_name)
mverb_model = T5ForConditionalGeneration.from_pretrained(mverb_model_name)

with open('/home/azagar/myfiles/slo-kit/data/eval/literatura/detektiv-dante.txt') as f:
    text = f.read()

print(text)
text_sentences = nltk.sent_tokenize(text)

for sent in text_sentences:
    input_ids = headline_tokenizer(f"summarize: {sent}", return_tensors="pt", max_length=512,
                                   truncation=True).input_ids
    input_ids = input_ids.to("cpu")
    outputs = headline_model.generate(input_ids, max_length=128)

    headline_text = headline_tokenizer.decode(outputs[0], skip_special_tokens=True)

    prefix = "correct: "
    input_ids = mverb_tokenizer(f"{prefix} {headline_text}", return_tensors="pt", max_length=32,
                                truncation=True).input_ids
    input_ids = input_ids.to("cpu")
    outputs = mverb_model.generate(input_ids,
                                   max_length=32,
                                   num_beams=5,
                                   do_sample=True,
                                   top_k=5,
                                   temperature=0.7
                                   # num_return_sequences=5
                                   )

    simplified_text = mverb_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(sent, '\t', simplified_text)
    print('\n')
