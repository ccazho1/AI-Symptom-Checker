import json
from tqdm import tqdm
from utils import obtain_data
from retreival import key_retreival

source_set, train_set, query_set = obtain_data()
query_set = train_set

top_k = 5

for threshold in [round(0.6 + 0.05 * i, 2) for i in range(9)]:
    correct = 0
    all_results = []

    for i, query_entry in enumerate(tqdm(query_set)):
        query = query_entry["text"]
        label_gt = query_entry["label"]

        print(f"\n[{i+1}/{len(query_set)}] Query: {query}")
        top_indices, scores = key_retreival(query, source_set, top_k, threshold)

        pred_label = source_set[top_indices[0]]["label"] if top_indices else "N/A"
        print(f"Predicted label: {pred_label} | Ground truth: {label_gt}")
        if pred_label == label_gt:
            correct += 1

        result = {
            "query_index": i,
            "query_text": query,
            "correct_label": label_gt,
            "predicted_label": pred_label,
            "top_indices": top_indices,
            "top_labels": [source_set[i]["label"] for i in top_indices],
            "scores": scores,
            "mode": "semantic_ratio_source"
        }
        all_results.append(result)

    acc = correct / len(query_set)
    print(f"\nFinal Accuracy: {acc:.4f}")

    with open(f"./key_threshold/key_{threshold}.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
