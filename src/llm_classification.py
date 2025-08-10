import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

import google.generativeai as genai
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import src.log_config  # noqa: F401

logger = logging.getLogger(__name__)

# --- Configuração ---
load_dotenv(override=True)
BATCH_SIZE = 50
MAX_WORKERS = 10  # Número máximo de chamadas simultâneas
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
modelname = "gemini-2.0-flash"
model = genai.GenerativeModel(modelname)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3n:e2b"  # gemma3:1b
OLLAMA_BATCH_SIZE = 20


def generate_prompt(batch_data, few_shot_examples=None):
    prompt = """
        Você é um especialista em análise de sentimentos para avaliações de e-commerce.
        Sua tarefa é classificar o sentimento do cliente como 'positivo', 'negativo' ou 'neutro'.

        ---
        DEFINIÇÕES IMPORTANTES:
        - **Positivo**: O cliente expressa satisfação, alegria ou elogia o produto/serviço.
        - **Negativo**: O cliente expressa insatisfação, raiva ou decepção.
        - **Neutro**: O cliente descreve o produto sem emoção clara.
        ---

        Use os seguintes exemplos como guia:
    """
    for sentiment, examples in few_shot_examples.items():
        for i, example_text in enumerate(examples):
            prompt += f'\n- Exemplo {sentiment.upper()} {i+1}: "{example_text}"'

    prompt += f"""
        Sua resposta deve ser um objeto JSON com a chave 'classificacoes'.
        Cada item deve conter 'id' e 'sentimento'.

        Avaliações para classificar:
        {json.dumps(batch_data, ensure_ascii=False)}
    """
    return prompt


# Função para processar um único lote
def classify_batch(batch, few_shot_examples):
    try:
        prompt = generate_prompt(batch, few_shot_examples)
        response = model.generate_content(prompt)
        cleaned = response.text.strip().replace("```json", "").replace("```", "")
        batch_results = json.loads(cleaned).get("classificacoes", [])
        return {
            item["id"]: item.get("sentimento", "erro_parse") for item in batch_results
        }
    except Exception as e:
        logger.error(f"[Erro no lote]: {e}")
        return {item["id"]: "erro_api/json" for item in batch}


# Função principal paralela
def run_gemini_classification(texts_to_classify, few_shot_examples=None):
    reviews_with_ids = [
        {"id": i, "text": str(text)} for i, text in enumerate(texts_to_classify)
    ]
    batches = [
        reviews_with_ids[i : i + BATCH_SIZE]
        for i in range(0, len(reviews_with_ids), BATCH_SIZE)
    ]

    all_results = {}
    logger.info(
        f"Iniciando classificação paralela em {len(batches)} lotes com até {MAX_WORKERS} threads..."
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_batch, batch, few_shot_examples): batch
            for batch in batches
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Classificando"
        ):
            result = future.result()
            all_results.update(result)
            # time.sleep(1)

    # Ordena os resultados finais
    final_sentiments = [
        all_results.get(i, "erro_desconhecido") for i in range(len(texts_to_classify))
    ]
    return final_sentiments


def resample_few_shot(
    df: pd.DataFrame, is_absolute: bool = True, n: int = 2, frac: float = None
) -> Tuple[pd.DataFrame, dict]:
    """Realiza uma reamostragem nos dados para criar um conjunto de dados para
    compor o aprendiado few-shot. É possivel realizar a reamostragem de forma
    absoluta (quantidade fixa de amostras por classe) ou relativa (uma fração
    das amostras por classe).

    Notes
    -----
    Note que se a reamostragem for relativa, a porcentagem de amostras é
    construída a partir do número total de amostras de cada class e não
    do número total de amostras do dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem reamostrados.
    is_absolute : bool, optional
        Se True, realiza a reamostragem absoluta, caso contrário, relativa.
        By default, True.
    n : int, optional
        Número de amostras a serem mantidas por classe. By default 2.
    frac : float, optional
        Porcentagem de amostras a serem mantidas por classe. By default None.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Tupla contendo o DataFrame reamostrado e um dicionário com exemplos
        de few-shot para cada classe.

    """

    if is_absolute or frac is None:
        if frac is None:
            logger.info("Usando amostragem absoluta, pois frac é None.")
        df_few_shot = df.groupby("annotation", group_keys=False).apply(
            lambda x: x.head(n)
        )
    else:
        if not (0 < frac < 1):
            raise ValueError("Valor de frac deve estar entre 0 e 1.")
        df_few_shot = df.groupby("annotation", group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=RANDOM_STATE)
        )

    dict_fewShot_examples = {
        "positivo": df_few_shot[df_few_shot["annotation"] == "positivo"][
            "review_text"
        ].tolist(),
        "negativo": df_few_shot[df_few_shot["annotation"] == "negativo"][
            "review_text"
        ].tolist(),
        "neutro": df_few_shot[df_few_shot["annotation"] == "neutro"][
            "review_text"
        ].tolist(),
    }

    return df_few_shot, dict_fewShot_examples


def generate_ollama_prompt(batch_data, few_shot_examples=None):
    prompt = """Você é um especialista em análise de sentimentos. 
    Classifique cada texto como 'positivo', 'negativo' ou 'neutro'.\n\n"""

    if few_shot_examples:
        prompt += "Exemplos:\n"
        for sentiment, examples in few_shot_examples.items():
            for example in examples[:1]:
                prompt += f'Texto: "{example}"\nSentimento: {sentiment}\n\n'
    prompt += "Agora classifique:\n"
    for item in batch_data:
        prompt += f"ID {item['id']}: \"{item['text']}\"\n"
    prompt += '\nResponda no formato JSON: {"classificacoes": [{"id": 0, "sentimento": "positivo"}]}'
    return prompt


def classify_batch_ollama(batch, few_shot_examples):
    prompt = generate_ollama_prompt(batch, few_shot_examples)
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result_text = response.json()["response"]
        # Extrai JSON da resposta
        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = result_text[start:end]
            results = json.loads(json_str).get("classificacoes", [])
            return {item["id"]: item.get("sentimento", "neutro") for item in results}
        return {item["id"]: "neutro" for item in batch}
    except Exception as e:
        logger.error(f"Erro no lote OLLAMA: {e}")
        return {item["id"]: "erro" for item in batch}


def run_ollama_classification(texts_to_classify, few_shot_examples=None):
    reviews_with_ids = [
        {"id": i, "text": str(text)} for i, text in enumerate(texts_to_classify)
    ]
    batches = [
        reviews_with_ids[i : i + OLLAMA_BATCH_SIZE]
        for i in range(0, len(reviews_with_ids), OLLAMA_BATCH_SIZE)
    ]
    all_results = {}

    logger.info(
        f"Iniciando classificação paralela em {len(batches)} lotes com até "
        f"{MAX_WORKERS} threads."
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_batch_ollama, batch, few_shot_examples): batch
            for batch in batches
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Classificando"
        ):
            result = future.result()
            all_results.update(result)

    # Ordena os resultados finais
    final_sentiments = [
        all_results.get(i, "erro_desconhecido") for i in range(len(texts_to_classify))
    ]
    return final_sentiments
