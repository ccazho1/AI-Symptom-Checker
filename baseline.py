from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
import os
import re

gen_model_name = "gpt2"
nli_model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

input_path = "output/output_evaluate.json"
output_path = "output/nli_baseline.json"

gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
gen_model.eval()

nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_model.to(device)
nli_model.to(device)

def predict_nli(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = nli_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    labels = ["entailment", "contradiction"]
    pred = labels[torch.argmax(probs)]
    return pred, probs[0].tolist()

def extract_diseases_and_explanations(text):
    pattern = r"\d+\.\s+(.*?)\n\s*Potential casuses and treatments:\s*(.+)"
    return re.findall(pattern, text, re.DOTALL)

with open(input_path, "r", encoding="utf-8") as f:
    loaded_results = json.load(f)

nli_results = []

for result in loaded_results:
    original_text = result["output_text"]
    diseases_with_expl = extract_diseases_and_explanations(original_text)
    
    for disease, reference_expl in diseases_with_expl:
        prompt = f"Disease: {disease}\nDescribe its possible cause and treatment:\n"
        inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)
        outputs = gen_model.generate(inputs["input_ids"], max_length=100, do_sample=True, temperature=1.05, top_p=0.9)
        generated_expl = gen_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

        label, prob = predict_nli(reference_expl.strip(), generated_expl)

        nli_results.append({
            "disease": disease.strip(),
            "reference_output": reference_expl.strip(),
            "gpt_output": generated_expl,
            "nli_label": label,
            "nli_probabilities": {
                "entailment": prob[0],
                "contradiction": prob[1]
            }
        })

os.makedirs("output", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nli_results, f, indent=2, ensure_ascii=False)
