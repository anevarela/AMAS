import os
import json
import pandas as pd
from openai import OpenAI
import functools

client = OpenAI()

print = functools.partial(print, flush=True)

# =====================================================================
# 1. CONFIGURATION & MODEL PARAMETERS
# =====================================================================

NER_CATEGORIES = [
    "sdoh_community_present",
    "sdoh_community_absent",
    "sdoh_education",
    "sdoh_economics",
    "sdoh_environment",
    "behavior_alcohol",
    "behavior_tobacco",
    "behavior_drug"
]

schema = {
    "type": "object",
    "properties": {
        "ner_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_text": {
                        "type": "string",
                        "description": "The exact text snippet identified as an entity."
                    },
                    "category": {
                        "type": "string",
                        "enum": NER_CATEGORIES,
                        "description": "The SDOH category for this entity."
                    }
                },
                "required": ["entity_text", "category"],
                "additionalProperties": False
            }
        }
    },
    "required": ["ner_entities"],
    "additionalProperties": False
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

        Extrae una lista bajo la clave "ner_entities".
        Cada elemento debe ser un array:

        ["texto extraído", "categoría"]

        La categoría debe ser exactamente una de las siguientes:

        - sdoh_community_present
        - sdoh_community_absent
        - sdoh_education
        - sdoh_economics
        - sdoh_environment
        - behavior_alcohol
        - behavior_tobacco
        - behavior_drug
        Incluye modificadores si aplica (ej. combinaciones con QUANTITY, TEMPORALITY o NEGATION).

        Devuelve únicamente un objeto JSON válido que cumpla exactamente el esquema proporcionado. No añadas texto explicativo ni bloques Markdown.
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
# 4. INFERENCE EXECUTION PIPELINE
# =====================================================================
def run_benchmark(target_texts: list, n_shots: int = 1):
    print(f"Loading GPT...")
    
    few_shot_context = load_few_shot_context(n_shots)
    
    results = []
    
    for i, text in enumerate(target_texts):
        print(f"{i+1}/{len(target_texts)}")
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "system",
                    "content": build_system_prompt()
                },
                {
                    "role": "user",
                    "content": build_user_prompt(few_shot_context, text)
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "sdoh_extraction",
                    "schema": schema,
                    "strict": True
                }
            }
        )

        results.append(json.loads(response.output_text))
            
    return results

if __name__ == "__main__":
    # Test batch cases
    test = pd.read_csv("test.csv")
    texts = list(test["text_sp"])
    
    for n_shot in [3]:
        print(f"\n==================== Nº of shots: {n_shot} ====================")
        try:
            outputs = run_benchmark(target_texts=texts, n_shots=n_shot)
            test["output"] = [
                json.dumps(o, ensure_ascii=False)
                for o in outputs
            ]
            test.to_csv(f"test_onlyner_{n_shot}_gpt.csv", index=False)
        except Exception as e:
            print(e)