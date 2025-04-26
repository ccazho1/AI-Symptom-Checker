import random
import torch
import re
import nltk
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from datasets import load_dataset
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s/\-\+]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word != '.']
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def get_paraphrases(query, model_name="humarin/chatgpt_paraphraser_on_T5_base", num_return_sequences=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_text = f"paraphrase: {query} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=256)
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        top_k=2,
        top_p=0.8,
        temperature=0.8,
        max_length=128,
        num_return_sequences=num_return_sequences,
        early_stopping=True
    )
    return [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in output_ids]


def BM25_retrieval(query, source_set, top_k=5, data_augmentation=True, seed=42):
    set_seed(seed)

    if data_augmentation:
        aug_queries = get_paraphrases(query)
        aug_queries.append(query)
        tokenized_query = []
        for q in aug_queries:
            tokenized_query += preprocess(q)
    else:
        tokenized_query = preprocess(query)

    tokenized_docs = [preprocess(doc) for doc in source_set["text"]]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(tokenized_query)

    sorted_indices = np.argsort(bm25_scores)[::-1]

    seen_labels = set()
    selected_indices = []
    for idx in sorted_indices:
        idx = int(idx)
        label = source_set[idx]['label']
        if label not in seen_labels:
            selected_indices.append(idx)
            seen_labels.add(label)
        if len(selected_indices) == top_k:
            break

    # print(f"\nTop-{top_k}:\n")
    # for idx in selected_indices:
    #     print(f"[{idx}] Score: {bm25_scores[idx]:.4f} | Label: {source_set[idx]['label']}")
    #     print(f"→ {source_set[idx]['text'][:100]}...\n")
    return selected_indices, [bm25_scores[i] for i in selected_indices]


def BERT_retrieval(query, source_set, model_name='all-MiniLM-L6-v2',top_k=5):
    model = SentenceTransformer(model_name)
    doc_embeddings = model.encode(source_set["text"], convert_to_tensor=False)
    query_embedding = model.encode([query])[0].reshape(1, -1)
    cos_scores = cosine_similarity(query_embedding, doc_embeddings)[0]  # 一维向量
    
    sorted_indices = np.argsort(cos_scores)[::-1]
    seen_labels = set()
    selected_indices = []
    for idx in sorted_indices:
        idx = int(idx)
        label = source_set[idx]['label']
        if label not in seen_labels:
            selected_indices.append(idx)
            seen_labels.add(label)
        if len(selected_indices) == top_k:
            break

    # print(f"\nTop-{top_k}:\n")
    # for idx in selected_indices:
    #     print(f"[{idx}] Score: {cos_scores[idx]:.4f} | Label: {source_set[idx]['label']}")
    #     print(f"→ {source_set[idx]['text'][:100]}...\n")
    return selected_indices, [cos_scores[i] for i in selected_indices]


def key_retreival(query, data, top_k=5, threshold=0.9):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    ner_model_name = "d4data/biomedical-ner-all"
    tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    def extract_medical_keywords(text):
        ner_results = ner_pipeline(text)
        return [ent["word"].lower() for ent in ner_results if ent["entity_group"].lower() == 'sign_symptom']

    if not os.path.exists("source_embeddings.pt"):
        print("Generating source embeddings from data...")
        source_embeddings = []
        for entry in tqdm(data):
            keywords = extract_medical_keywords(entry["text"])
            if keywords:
                emb = embedder.encode(keywords, convert_to_tensor=True)
            else:
                emb = torch.empty((0, 384))
            source_embeddings.append(emb)
        torch.save(source_embeddings, "source_embeddings.pt")
        print("Saved source embeddings to source_embeddings.pt")
    else:
        print("Using cached source_embeddings.pt")
        source_embeddings = torch.load("source_embeddings.pt")

    query_keywords = extract_medical_keywords(query)
    # print("Query symptoms:", query_keywords)
    if not query_keywords:
        return [], []

    query_emb = embedder.encode(query_keywords, convert_to_tensor=True)

    ranked = []
    for i, emb in enumerate(source_embeddings):
        if emb.shape[0] == 0:
            continue
        cos_sim = util.cos_sim(query_emb, emb)
        matched = (cos_sim >= threshold).any(dim=1)
        ratio = matched.sum().item() / len(query_keywords)
        ranked.append((i, ratio, emb.shape[0]))

    ranked.sort(key=lambda x: (-x[1], x[2]))
    seen_labels = set()
    filtered = []
    for idx, ratio, _ in ranked:
        label = data[idx]["label"]
        if label not in seen_labels:
            seen_labels.add(label)
            filtered.append((idx, ratio))
        if len(filtered) >= top_k:
            break

    top_indices = [idx for idx, _ in filtered]
    scores = [float(score) for _, score in filtered]

    return top_indices, scores