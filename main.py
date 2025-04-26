from utils import *
from retreival import *

query = str(input("Please enter your symptoms: "))
# query = "I'm having a hard time breathing. I'm not feeling well, and I'm sweating a lot. I have a lot of mucous in my throat and my chest hurts. My breathing is labored, and the phlegm I'm coughing up is a weird color."

def retreive_symptom(query, best_coeff=[0.5, 0.4, 0.1]):
    source_set,_,_ = obtain_data()
    top_labels = []
    top_scores = []
    for method in ["bm25", "bert","key"]:
        print("conducting: ", method)
        if method == "bm25":
            top_indices, best_scores = BM25_retrieval(query, source_set, top_k=5, data_augmentation=True)
        elif method == "bert":
            top_indices, best_scores = BERT_retrieval(query, source_set, top_k=5)
        elif method == "key":
            top_indices, best_scores = key_retreival(query, source_set, top_k=5)
        top_labels.append([source_set[int(idx)]["label"] for idx in top_indices])
        top_scores.append([norm_score(score, method) for score in best_scores])

    labels = set(top_labels[0]) | set(top_labels[1]) | set(top_labels[2])
    label_weighted_scores = {}
    for label in labels:
        score1 = score2 = score3 = 0
        if label in top_labels[0]:
            score1 = top_scores[0][top_labels[0].index(label)]
        if label in top_labels[1]:
            score2 = top_scores[1][top_labels[1].index(label)]
        if label in top_labels[2]:
            score3 = top_scores[2][top_labels[2].index(label)]

        weighted_score = best_coeff[0] * score1 + best_coeff[1] * score2 + best_coeff[2] * score3
        label_weighted_scores[label] = weighted_score
        
    sorted_items = sorted(label_weighted_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:3]

sympotms = retreive_symptom(query)
with open("mapping.json", "r") as f:
    mapping = json.load(f)

print("Following are potential diseases and the patient may have (Ranked by decreasing probability):\n")
i = 1
for symptom, score in sympotms:
    name = mapping[str(symptom)]["disease"]
    output = mapping[str(symptom)]["output"]
    print(f"{i}. {name}")
    print(f"\tPotential casuses and treatments: output: {output}")
    i+=1