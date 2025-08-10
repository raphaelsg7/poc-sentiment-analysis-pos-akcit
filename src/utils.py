import logging
import os

import pandas as pd

import src.log_config  # noqa: F401

logger = logging.getLogger(__name__)


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """Persiste em disco, um DataFrame em um arquivo CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser salvo em disco.
    output_path : str
        Caminho no qual o arquivo CSV será salvo.
    """

    if df is None:
        logger.warning("Nenhum DataFrame para salvar")
        return

    # Garante que o diretório do arquivo exista antes de salvar
    dir_path = os.path.dirname(output_path)
    if dir_path:  # evita problema se for salvar no diretório atual
        create_dir(dir_path)

    try:
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Arquivo salvo com sucesso em '{output_path}'")
    except Exception as e:
        logger.exception(f"Ocorreu um erro ao salvar o arquivo: {e}")


def create_dir(dir_path: str) -> None:
    """Cria um diretório se ele não existir.

    Parameters
    ----------
    dir_path : str
        Caminho do diretório a ser criado.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Diretório criado: {dir_path}")
