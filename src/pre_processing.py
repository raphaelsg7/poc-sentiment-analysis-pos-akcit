import logging
import re
from typing import List

import nltk
import pandas as pd
import spacy
import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandarallel import pandarallel

import src.log_config  # noqa: F401

logger = logging.getLogger(__name__)

pandarallel.initialize(progress_bar=True)
nlp = None


def setup_nltk() -> None:
    """Configura o NLTK para uso no pipeline de pré-processamento."""

    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.DownloadError:
        logger.warning("Punkt tokenizer não encontrado. Iniciando download.")
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except nltk.downloader.DownloadError:
        logger.warning("Stopwords corpus não encontrado. Iniciando download.")
        nltk.download("stopwords")


def setup_spacy() -> None:
    """Baixa o modelo spaCy em português, se necessário, para lematização.

    Notes
    -----
    A função carrega/baixa o modelo md (medium) de português do spaCy.
    """

    global nlp
    try:
        nlp = spacy.load("pt_core_news_md")
    except OSError:
        logger.warning("Modelo spaCy não encontrado. Iniciando download.")
        spacy.cli.download("pt_core_news_md")
        nlp = spacy.load("pt_core_news_md")
        logger.info("Modelo spaCy baixado com sucesso.")


def clean_text(text: str) -> str:
    """Realiza a limpeza do texto, removendo elementos desnecessários e aplicando lematização.

    Parameters
    ----------
    text : str
        Texto a ser limpo.

    Returns
    -------
    str
        Texto limpo e lematizado.
    """

    # Garante que o texto é uma string
    if not isinstance(text, str):
        logger.warning("O texto fornecido não é uma string. Retornando string vazia.")
        return ""

    text = text.lower()  # coloca tudo minusculo
    text = re.sub(r"#\w+", "", text)  # remove hashtags
    text = re.sub(r"[^\w\s]", "", text)  # remove pontuação
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = unidecode.unidecode(text)  # Remove acentuação

    filtered_tokens = remove_stopwords(text)
    lemmatized_tokens = lemmatize_text(filtered_tokens)

    if not lemmatized_tokens:
        return " ".join(filtered_tokens)
    else:
        return " ".join(lemmatized_tokens)


def remove_stopwords(text: str) -> List[str]:
    """Remove stopwords de um texto.

    Parameters
    ----------
    text : str
        Texto a ter as stopwords removidas.

    Returns
    -------
    list
        Lista de palavras sem stopwords.
    """

    # Garante que o texto é uma string
    if not isinstance(text, str):
        logger.warning("O texto fornecido não é uma string. Retornando lista vazia.")
        return []

    tokens = word_tokenize(text, language="portuguese")
    stop_words = set(stopwords.words("portuguese"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return filtered_tokens


def lemmatize_text(tokens: list) -> List[str]:
    """Realiza a lematização dos tokens usando spaCy.

    Parameters
    ----------
    tokens : list
        Lista de tokens a serem lematizados.

    Returns
    -------
    List[str]
        Lista de tokens lematizados.
    """

    global nlp
    if nlp is None:
        logger.warning("spaCy não inicializado. Retornando tokens originais.")
        return tokens

    text = " ".join(tokens)
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_space]
    return lemmatized_tokens


def run_preprocessing_pipeline(file_path: str) -> pd.DataFrame:
    """Aplica o pipeline de pré-processamento de texto a um arquivo cujo
    caminnho é fornecido.

    Parameters
    ----------
    file_path : str
        Caminho para o arquivo a ser processado.

    Returns
    -------
    pd.DataFrame
        DataFrame processado.
    """

    logger.info("Iniciando o pipeline de pré-processamento")

    setup_nltk()
    setup_spacy()

    try:
        logger.info(f"Lendo arquivo: {file_path}")
        df_data = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.exception("ERRO: Arquivo de dados não encontrado.")
        return pd.DataFrame() 

    # Remove linhas em branco da coluna "review_text"
    df_data = df_data.dropna(subset=["review_text"])
    df_data = df_data[df_data["review_text"].str.strip() != ""]

    # Filtra o conjunto de dados para apenas ratings entre 1 e 5
    df_data = df_data[df_data["rating"].isin([1, 2, 3, 4, 5])]

    # Cria a coluna "annotation"
    df_data["annotation"] = df_data["rating"].apply(
        lambda x: 1 if x in [4, 5] else (0 if x == 3 else -1)
    )
    mapeamento_sentimentos = {-1: "negativo", 0: "neutro", 1: "positivo"}
    df_data["annotation"] = df_data["annotation"].map(mapeamento_sentimentos)

    df_data = df_data[["original_index", "review_text", "annotation"]].copy()

    # Aplica a limpeza na coluna "review_text"
    logger.info("Limpando os textos da base de dados.")
    df_data["review_text_processed"] = df_data["review_text"].parallel_apply(
        clean_text
    )

    # Remove as linhas duplicadas baseadas na coluna "review_text_processed"
    df_data = df_data.drop_duplicates(subset=["review_text_processed"]).copy()
    df_data.reset_index(drop=True, inplace=True)

    logger.info("Pipeline de pré-processamento concluído.")
    return df_data


# TODO : adicionar um passo para se encontrar o padrao Letra...Letra, transformar
# para Letra Letra. Ou seja, trocar os pontos por espaços
