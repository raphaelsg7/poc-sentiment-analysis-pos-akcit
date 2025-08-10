import logging
import os
import urllib.request
import zipfile
from typing import List

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import src.log_config  # noqa: F401
from src.utils import create_dir

logger = logging.getLogger(__name__)


def download_word2vec() -> str:
    """Realiza o download do modelo Word2Vec pré-treinado em dados em português.
    O modelo é baixado do diretório do Núcleo Interinstitucional de
    Linguística Computacional (NILC) da USP.

    Returns
    -------
    str
        Caminho para o modelo Word2Vec baixado.
    """

    url = "http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip"
    zip_path = "models/skip_s300.zip"
    model_path = "models/skip_s300.txt"

    create_dir("models")

    if not os.path.exists(model_path):
        logger.info("Baixando modelo pré-treinado do Word2Vec.")
        try:
            urllib.request.urlretrieve(url, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("models/")

            os.remove(zip_path)
        except Exception as e:
            logger.error(f"Erro no download: {e}")

            return None
    else:
        logger.info("Modelo Word2Vec já existe em disco.")

    return model_path


def load_word2vec() -> KeyedVectors:
    """Carrega o modelo Word2Vec pré-treinado.

    Returns
    -------
    KeyedVectors
        Modelo Word2Vec carregado.
    """

    model_path = download_word2vec()
    if model_path and os.path.exists(model_path):
        logger.info("Carregando modelo Word2Vec.")
        w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        logger.info("Word2Vec carregado!")
    else:
        logger.error("Erro ao carregar Word2Vec.")

    return w2v_model


def create_word2vec_vectors(
    texts: List[str], model: KeyedVectors, vector_size: int = 300
) -> np.ndarray:
    """A partir de uma lista de textos, cria os vetores médios de palavras
    usando o modelo Word2Vec.

    Parameters
    ----------
    texts : List[str]
        Lista de textos a serem vetorizados.
    model : KeyedVectors
        Modelo Word2Vec a ser utilizado.
    vector_size : int, optional
        Tamanho dos vetores de saída, by default 300

    Returns
    -------
    np.ndarray
        Vetores médios de palavras para os textos fornecidos.
    """

    if model is not None:
        vectors = []
        for text in texts:
            words = text.split()
            word_vectors = [model[word] for word in words if word in model]

            vectors.append(
                np.mean(word_vectors, axis=0) if word_vectors else np.zeros(vector_size)
            )

        return np.array(vectors)
    else:
        logger.error("Modelo Word2Vec não carregado. Retornando vetor nulo.")
        return np.zeros((len(texts), vector_size))


def vectorize_with_word2vec(
    texts: List[str], mode: str = "mean", vector_size: int = 300
) -> np.ndarray:
    """Função agregadora para vetorização de textos usando o modelo Word2Vec.

    Parameters
    ----------
    texts : List[str]
        Lista de textos a serem vetorizados.
    mode : str, optional
        Modo de agregação a ser utilizado, by default "mean".
    vector_size : int, optional
        Tamanho dos vetores de saída, by default 300.

    Returns
    -------
    np.ndarray
        Vetores resultantes da vetorização.
    """

    if mode == "mean":
        w2v_model = load_word2vec()
        if w2v_model is None:
            vectorized_data = np.zeros((len(texts), vector_size))
            logger.error("Modelo Word2Vec não carregado. Retornando vetor nulo.")
        else:
            vectorized_data = create_word2vec_vectors(texts, w2v_model, vector_size)
    else:
        logger.warning(f"Modo '{mode}' ainda não implementado.")

    return vectorized_data


def vectorize_with_bert(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """Realiza o carregamento do modelo BERT e a vetorização de textos com
    esse modelo. O modelo é carregado do Hugging Face Hub e os textos são
    vetorizados em lotes para otimizar o uso de memória.

    Parameters
    ----------
    texts : List[str]
        Lista de textos a serem vetorizados.
    batch_size : int, optional
        Tamanho dos lotes para vetorização, by default 32

    Returns
    -------
    np.ndarray
        Vetores resultantes da vetorização.
    """

    model_name = "neuralmind/bert-large-portuguese-cased"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError:
        logger.error(f"Erro: Modelo '{model_name}' não encontrado")
        return np.array([])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    model.to(device)
    model.eval()

    all_vectors = []

    dataloader = DataLoader(
        texts,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,  # Ajustar conforme os núcleos do processador
        pin_memory=True,  # Acelera a transferência de dados CPU -> GPU
    )
    logger.info("Iniciando a vetorização.")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Vetorizando textos"):

            # Tokeniza o lote inteiro
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.autocast("cuda" if device.type == "cuda" else "cpu"):
                outputs = model(**inputs)

            batch_vectors = outputs.last_hidden_state[:, 0, :].cpu()
            all_vectors.append(batch_vectors)

            # all_vectors.extend(batch_vectors)
    final_vectors = torch.cat(all_vectors, dim=0).numpy()
    logger.info("Vetorização concluída.")

    return final_vectors
