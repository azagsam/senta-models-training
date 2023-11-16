import nltk
import pandas as pd
import torch
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

if __name__ == "__main__":
    cls_handle = "sloberta_slokit_maxlen65_10e_lr2e-05/checkpoint-474/"
    gen_name = f"slokit-mt5-xl/checkpoint-2502"
    test_path = "data/eval_datasets.jsonl"
    cls_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_data = pd.read_json(test_path, orient="records", lines=True)

    inference_results = {"source": [], "id": [], "idx_sent_in_doc": [], "path": [],
                         "complex_sent": [], "is_simple_detected": [], "simplified_sent": []}
    for idx_ex in trange(test_data.shape[0]):
        ex = test_data.iloc[idx_ex]
        doc = ex["text"]

        sents = nltk.sent_tokenize(doc, language="slovene")
        idx_s = 0
        for s in sents:
            if "\n" in s:
                sent_parts = s.split("\n")
                for _part in sent_parts:
                    simplified_part, is_already_simple = "N/A", False

                    inference_results["source"].append(ex["source"])
                    inference_results["id"].append(ex["id"])
                    inference_results["idx_sent_in_doc"].append(idx_s)
                    inference_results["path"].append(ex["path"])
                    inference_results["complex_sent"].append(_part)
                    inference_results["is_simple_detected"].append(is_already_simple)
                    inference_results["simplified_sent"].append(simplified_part)
                    idx_s += 1

                continue

            simplified_part, is_already_simple = "N/A", False
            inference_results["source"].append(ex["source"])
            inference_results["id"].append(ex["id"])
            inference_results["idx_sent_in_doc"].append(idx_s)
            inference_results["path"].append(ex["path"])
            inference_results["complex_sent"].append(s)
            inference_results["is_simple_detected"].append(is_already_simple)
            inference_results["simplified_sent"].append(simplified_part)
            idx_s += 1

    inference_results = pd.DataFrame(inference_results)

    cls_tokenizer = AutoTokenizer.from_pretrained(cls_handle)
    cls_model = AutoModelForSequenceClassification.from_pretrained(cls_handle).to(cls_device)
    cls_model.eval()

    encoded_complex = cls_tokenizer(inference_results["complex_sent"].tolist(),
                                    max_length=65, truncation=True, padding="max_length", return_tensors="pt")
    cls_batch_size = 1024
    num_batches = (inference_results.shape[0] + cls_batch_size - 1) // cls_batch_size

    print(f"Step 1: detecting if sentences are already simplified using '{cls_handle}'...")
    are_sents_simple = []
    with torch.inference_mode():
        for idx_batch in trange(num_batches):
            s_b, e_b = idx_batch * cls_batch_size, (idx_batch + 1) * cls_batch_size

            input_data = {_k: _v[s_b: e_b].to(cls_device) for _k, _v in encoded_complex.items()}
            logits = cls_model(**input_data).logits
            probas = torch.softmax(logits, dim=-1).cpu()
            preds = torch.argmax(probas, dim=-1)

            are_sents_simple.extend(preds.bool().tolist())

    inference_results["is_simple_detected"] = are_sents_simple
    del cls_tokenizer
    del cls_model

    gen_tokenizer = AutoTokenizer.from_pretrained(gen_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_name, device_map="balanced", max_memory={0: "10GiB", 1: "10GiB"})


    # preprocess data
    prefix = "simplify: "


    def handle_sentence(input_sentence, is_simple_detected=False):
        simplified = str(input_sentence)

        if not is_simple_detected:
            input_ids = gen_tokenizer(f"{prefix} {input_sentence}", return_tensors="pt", max_length=128, truncation=True).input_ids.to(gen_model.device)
            outputs = gen_model.generate(input_ids,
                                         max_length=128,
                                         num_beams=5,
                                         do_sample=True,
                                         top_k=5,
                                         temperature=0.7)

            simplified = gen_tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)

        return simplified


    print(f"Step 2: generating simplified sentences using '{gen_name}'...")
    simplified_sents = []
    with torch.inference_mode():
        for idx_ex in trange(inference_results.shape[0]):
            curr_input_str = inference_results.iloc[idx_ex]["complex_sent"]
            mask = inference_results.iloc[idx_ex]["is_simple_detected"]

            simplified_sents.append(handle_sentence(curr_input_str, is_simple_detected=mask))

    inference_results = pd.DataFrame(inference_results)
    inference_results["simplified_sent"] = simplified_sents
    inference_results.to_json("inference_results_mt5_xl.jsonl", lines=True, orient="records")
