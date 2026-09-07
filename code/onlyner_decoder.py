import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import functools

print = functools.partial(print, flush=True)

# =====================================================================
# 1. CONFIGURATION & MODEL PARAMETERS
# =====================================================================
MODEL_MAP = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-4-E4B-it", 
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it"
}

# =====================================================================
# 2. PROMPT BUILDING UTILITIES WITH GUIDELINES INJECTION
# =====================================================================

def load_few_shot_context(n_shots: int) -> str:
    """Loads example datasets from a JSON file and constructs the few-shot prompt context."""
    json_path = "few_shot_prompt_examples.json"
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Running zero-shot configuration.")
        return ""
    
    with open(json_path, "r", encoding="utf-8") as f:
        all_examples = json.load(f)
        
    few_shot_str = "A continuación se presentan ejemplos de referencia con el formato esperado:\n\n"
    
    for idx, example in enumerate(all_examples[:n_shots]):
        text = example.get("text", "")
        expected_output = example.get("expected_output", {})
        del expected_output["classification"]
        
        few_shot_str += f"### Ejemplo {idx + 1}\n"
        few_shot_str += f"Texto Clínico:\n{text}\n"
        few_shot_str += f"Resultado Esperado (JSON):\n{json.dumps(expected_output, ensure_ascii=False, indent=2)}\n\n---\n\n"
        
    return few_shot_str

def build_system_prompt() -> str:
    """Constructs the comprehensive system prompt."""
    return (
        """
        Eres un asistente de IA experto en extraer entidades (NER) y clasificar factores determinantes de la salud (SDOH) en texto clínico en español.

        Extrae un array de strings bajo la clave "ner_entities" con palabras clave específicas de SDOH de estas categorías:
            "sdoh_community_present",
            "sdoh_community_absent",
            "sdoh_education",
            "sdoh_economics",
            "sdoh_environment",
            "behavior_alcohol",
            "behavior_tobacco",
            "behavior_drug"
        Incluye modificadores si aplica (ej. combinaciones con QUANTITY, TEMPORALITY o NEGATION).

        Responde EXCLUSIVAMENTE con un objeto JSON válido con este esquema exacto:
        {
        "ner_entities": [<pares de palabra y categoría extraídos>],
        }
        """
    )

def build_user_prompt(few_shot_context: str, clinical_text: str) -> str:
    return (
        f"{few_shot_context}"
        "Analiza el siguiente texto clínico objetivo aplicando de manera rigurosa las directrices del sistema, "
        "y genera tu respuesta estructurada:\n\n"
        f"Texto Objetivo:\n{clinical_text}\n\n"
        "Respuesta JSON:"
    )

# =====================================================================
# 3. VALIDATION LOOP
# =====================================================================
def validate_and_parse_json(raw_output: str) -> dict:
    """Isolates, sanitizes, and verifies JSON compliance."""
    if "```json" in raw_output:
        raw_output = raw_output.split("```json")[-1].split("```")[0]
    elif "```" in raw_output:
        raw_output = raw_output.split("```")[1].split("```")[0]
        
    data = json.loads(raw_output.strip())
    
    if "ner_entities" not in data:
        raise KeyError("Missing primary root keys ('ner_entities').")
            
    return data

# =====================================================================
# 4. INFERENCE EXECUTION PIPELINE
# =====================================================================
def run_benchmark(model_key: str, target_texts: list, n_shots: int = 1, max_retries: int = 3, batch_size: int = 8):
    print(f"Loading tokenizer and model for: {MODEL_MAP[model_key]}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_key])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_key], 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    few_shot_context = load_few_shot_context(n_shots)
    system_instruction = build_system_prompt()
    
    results = [None] * len(target_texts)
    
    for batch_idx in range(0, len(target_texts), batch_size):
        batch_upper = min(batch_idx + batch_size, len(target_texts))
        batch_slice = target_texts[batch_idx:batch_upper]
        
        print(f"\nProcessing batch {batch_idx // batch_size + 1}: items {batch_idx} to {batch_upper - 1}")
        
        batch_states = []
        for local_idx, text in enumerate(batch_slice):
            global_idx = batch_idx + local_idx
            user_instruction = build_user_prompt(few_shot_context, text)
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction}
            ]
            batch_states.append({
                "global_idx": global_idx,
                "text": text,
                "messages": messages,
                "success": False,
                "prediction": None,
                "last_response": ""
            })
            
        for attempt in range(max_retries):
            active_items = [item for item in batch_states if not item["success"]]
            if not active_items:
                break
                
            prompts = [
                tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
                for item in active_items
            ]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1024, 
                    temperature=0.1 if attempt == 0 else 0.4, 
                    do_sample=True if attempt > 0 else False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            new_tokens = generated_ids[:, prompt_len:]
            responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            
            for item, response in zip(active_items, responses):
                item["last_response"] = response
                try:
                    parsed_json = validate_and_parse_json(response)
                    item["prediction"] = parsed_json
                    item["success"] = True
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[Texto {item['global_idx']}] Attempt {attempt + 1} failed validation: {e}. Appending retry prompt...")
                    item["messages"].append({"role": "assistant", "content": response})
                    item["messages"].append({
                        "role": "user", 
                        "content": "Error: Tu respuesta no cumplió con las especificaciones del esquema o el formato JSON es inválido. Genera de nuevo la estructura de forma estricta."
                    })
                    
        for item in batch_states:
            results[item["global_idx"]] = {
                "index": item["global_idx"],
                "text": item["text"],
                "success": item["success"],
                "prediction": item["prediction"] if item["success"] else {"error": "Failed to resolve after retries", "raw": item["last_response"]}
            }
            
    return results

if __name__ == "__main__":
    # Test batch cases
    test = pd.read_csv("test.csv")
    texts = list(test["text_sp"])
    
    for model_name in MODEL_MAP.keys():
        print(f"\n==================== RUNNING BENCHMARK: {model_name} ====================")

        for n_shot in [4]:

            print(f"\n==================== Nº of shots: {n_shot} ====================")
            try:
                test["output"] = run_benchmark(model_key=model_name, target_texts=texts, n_shots=n_shot)
                test.to_csv(f"test_onlyner_{n_shot}_{model_name}", index=False)
            except Exception as e:
                print(e)
