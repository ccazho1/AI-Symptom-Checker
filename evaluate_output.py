import json
from utils import normalize_scores
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json

model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

with open("./test/bm25_true.json", "r") as f:
    bm25 = json.load(f)
with open("./test/bert.json", "r") as f:
    bert = json.load(f)
with open("./test/key.json", "r") as f:
    key = json.load(f)

with open("mapping.json", "r") as f:
    mapping = json.load(f)

best_coeff = [0.5, 0.4, 0.1]

bm25 = normalize_scores(bm25)
bert = normalize_scores(bert)
key = normalize_scores(key)

all_outputs = []

for query_idx in range(len(bm25)):
    query_text = bm25[query_idx]["query_text"]
    correct_label = bm25[query_idx]["correct_label"]

    labels = set(bm25[query_idx]["top_labels"]) | set(bert[query_idx]["top_labels"]) | set(key[query_idx]["top_labels"])
    label_weighted_scores = {}

    for label in labels:
        score1 = score2 = score3 = 0
        if label in bm25[query_idx]["top_labels"]:
            pos = bm25[query_idx]["top_labels"].index(label)
            score1 = bm25[query_idx]["norm_scores"][pos]
        if label in bert[query_idx]["top_labels"]:
            pos = bert[query_idx]["top_labels"].index(label)
            score2 = bert[query_idx]["norm_scores"][pos]
        if label in key[query_idx]["top_labels"]:
            pos = key[query_idx]["top_labels"].index(label)
            score3 = key[query_idx]["norm_scores"][pos]

        weighted_score = best_coeff[0] * score1 + best_coeff[1] * score2 + best_coeff[2] * score3
        label_weighted_scores[label] = weighted_score

    sorted_items = sorted(label_weighted_scores.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_items[:3]
    top_pred_label = top3[0][0]

    if top_pred_label != correct_label:
        continue

    lines = ["Following are potential diseases and the patient may have (Ranked by decreasing probability):\n"]
    for i, (symptom, score) in enumerate(top3, start=1):
        str_symptom = str(symptom)
        name = mapping.get(str_symptom, {}).get("disease", f"[Unknown ID: {str_symptom}]")
        output = mapping.get(str_symptom, {}).get("output", "N/A")
        lines.append(f"{i}. {name}")
        lines.append(f"\tPotential casuses and treatments: {output}")

    output_text = "\n".join(lines)

    all_outputs.append({
        "query": query_text,
        "output_text": output_text
    })

with open("./output/output_evaluate.json", "w", encoding="utf-8") as f:
    json.dump(all_outputs, f, indent=2, ensure_ascii=False)

def predict_nli(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    labels = ["entailment","contradiction"]
    pred = labels[torch.argmax(probs)]
    return pred, probs[0].tolist()

with open("./output/output_evaluate.json", "r") as f:
    all_outputs = json.load(f)

results = []

for item in all_outputs:
    symptom = item["query"]
    hypothesis = item["output_text"]
    label, prob = predict_nli(symptom, hypothesis)

    results.append({
        "query": symptom,
        "hypothesis": hypothesis,
        "nli_label": label,
        "nli_probabilities": {
            "entailment": prob[0],
            "contradiction": prob[1]
        }
    })

with open("./output/nli_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to ./output/nli_results.json")
