from utils import obtain_data
from retreival import *
import json

source_set, train_set, query_set = obtain_data()
query_set = train_set
for method in ["bm25_true","bm25_false", "bert","key"]:
    print(f"conducting {method} retrieval")
    N = len(query_set)
    cor = 0
    all_results = [] 

    for i in range(N):
        query = query_set[i]['text']
        print("conducting: ", i)
        if method == "bm25_true":
            top_indices, best_scores = BM25_retrieval(query, source_set, top_k=5, data_augmentation=True)
        elif method == "bm25_false":
            top_indices, best_scores = BM25_retrieval(query, source_set, top_k=5, data_augmentation=False)
        elif method == "bert":
            top_indices, best_scores = BERT_retrieval(query, source_set, top_k=5)
        elif method == "key":
            top_indices, best_scores = key_retreival(query, source_set, top_k=5)
        correct_ans = query_set[i]['label']
        pred = source_set[int(top_indices[0])]["label"]
        if pred == correct_ans:
            cor += 1

        result = {
            "query_index": i,
            "query_text": query,
            "correct_label": correct_ans,
            "predicted_label": pred,
            "top_indices": [int(idx) for idx in top_indices],
            "top_labels": [source_set[int(idx)]["label"] for idx in top_indices],
            "scores": [float(score) for score in best_scores]
        }
        all_results.append(result)

    print("Accuracy:", cor / N)

    with open(f"./test/{method}.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
