import pandas as pd
import json
from datasets import load_dataset, Dataset

def obtain_data():
    ds = load_dataset("dux-tecblic/symptom-disease-dataset")
    
    source_df = pd.DataFrame(ds["train"])
    source_df = source_df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    split = ds["test"].train_test_split(test_size=0.25, seed=42)
    train_df = pd.DataFrame(split["train"])
    query_df = pd.DataFrame(split["test"])

    train_df = train_df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    query_df = query_df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    source_texts = set(source_df["text"])

    train_df = train_df[~train_df["text"].isin(source_texts)].reset_index(drop=True)
    query_df = query_df[~query_df["text"].isin(source_texts)].reset_index(drop=True)

    train_texts = set(train_df["text"])
    query_df = query_df[~query_df["text"].isin(train_texts)].reset_index(drop=True)

    source_set = Dataset.from_pandas(source_df)
    train_set = Dataset.from_pandas(train_df)
    query_set = Dataset.from_pandas(query_df)
    return source_set, train_set, query_set


def norm_score(score, type):
    with open(f'./train/train_{type}.json', 'r', encoding='utf-8') as f:
        doc = json.load(f)
    all_scores = []
    for query in doc:
        all_scores.extend(query["scores"])
    mean = sum(all_scores) / len(all_scores)
    return score / mean

def normalize_scores(doc):
    all_scores = []
    for query in doc:
        all_scores.extend(query["scores"])
    mean = sum(all_scores) / len(all_scores)
    for query in doc:
        query["norm_scores"] = [s / mean for s in query["scores"]]
    return doc

def combine_results(doc1, doc2, doc3, k1=0.4, k2=0.4, top_k=5):
    assert 0 <= k1 <= 1 and 0 <= k2 <= 1 and k1 + k2 <= 1, "k1 + k2 must be ≤ 1"
    weights = (k1, k2, 1 - k1 - k2)

    doc1 = normalize_scores(doc1)
    doc2 = normalize_scores(doc2)
    doc3 = normalize_scores(doc3)

    correct_count = 0
    for idx in range(len(doc1)):
        correct_label = doc1[idx].get("correct_label")
        labels = set(doc1[idx]["top_labels"]) | set(doc2[idx]["top_labels"]) | set(doc3[idx]["top_labels"])
        label_weighted_scores = {}

        for label in labels:
            score1 = score2 = score3 = 0
            if label in doc1[idx]["top_labels"]:
                pos = doc1[idx]["top_labels"].index(label)
                score1 = doc1[idx]["norm_scores"][pos]
            if label in doc2[idx]["top_labels"]:
                pos = doc2[idx]["top_labels"].index(label)
                score2 = doc2[idx]["norm_scores"][pos]
            if label in doc3[idx]["top_labels"]:
                pos = doc3[idx]["top_labels"].index(label)
                score3 = doc3[idx]["norm_scores"][pos]

            weighted_score = weights[0] * score1 + weights[1] * score2 + weights[2] * score3
            label_weighted_scores[label] = weighted_score

        sorted_items = sorted(label_weighted_scores.items(), key=lambda x: x[1], reverse=True)
        predicted_label = sorted_items[0][0]

        if predicted_label == correct_label:
            correct_count += 1

    accuracy = correct_count / len(doc1)
    return accuracy



def compute_topk_accuracy_fusion(doc1, doc2, doc3, k1=0.5, k2=0.4, top_k_eval=5):
    assert k1 + k2 <= 1.0
    k3 = 1 - k1 - k2
    weights = (k1, k2, k3)

    doc1 = normalize_scores(doc1)
    doc2 = normalize_scores(doc2)
    doc3 = normalize_scores(doc3)

    total = len(doc1)
    hit = [0 for _ in range(top_k_eval)]

    for idx in range(total):
        correct_label = doc1[idx]["correct_label"]
        labels = set(doc1[idx]["top_labels"]) | set(doc2[idx]["top_labels"]) | set(doc3[idx]["top_labels"])
        label_weighted_scores = {}

        for label in labels:
            score1 = score2 = score3 = 0
            if label in doc1[idx]["top_labels"]:
                pos = doc1[idx]["top_labels"].index(label)
                score1 = doc1[idx]["norm_scores"][pos]
            if label in doc2[idx]["top_labels"]:
                pos = doc2[idx]["top_labels"].index(label)
                score2 = doc2[idx]["norm_scores"][pos]
            if label in doc3[idx]["top_labels"]:
                pos = doc3[idx]["top_labels"].index(label)
                score3 = doc3[idx]["norm_scores"][pos]

            weighted_score = weights[0]*score1 + weights[1]*score2 + weights[2]*score3
            label_weighted_scores[label] = weighted_score

        sorted_labels = [label for label, _ in sorted(label_weighted_scores.items(), key=lambda x: x[1], reverse=True)]

        for k in range(top_k_eval):
            if correct_label in sorted_labels[:k+1]:
                hit[k] += 1

    topk_accuracies = [hit[k] / total for k in range(top_k_eval)]
    return topk_accuracies
