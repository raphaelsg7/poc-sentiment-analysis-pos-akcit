import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import src.llm_classification as llm_cls
import src.log_config  # noqa: F401
from src.prediction import calculate_metrics
from src.utils import create_dir

logger = logging.getLogger(__name__)


def calculate_llm_metrics(
    df_results: pd.DataFrame,
    vectorization_method: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    """Calcula as métricas de desempenho para os resultados da classificação
    feita pelo LLM.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame contendo os resultados da classificação.
    vectorization_method : str
        Método de few-shot learning utilizado na classificação.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], dict]
        Tupla contendo os dados de teste, rótulos de teste e predições,
        e um dicionário com os resultados da classificação.
    """

    if df_results.empty or df_results is None:
        logger.error("O DataFrame de resultados está vazio ou é None.")
        dict_results = {
            "model": llm_cls.modelname,
            "vectorization_method": vectorization_method,
            "undersampling": False,
            "cross_validation": False,
            "accuracy": 0.0,
            "f1_score": 0.0,
        }
        return (None, None, None), dict_results

    # Garante que ambos os textos estejam em minúsculas e sem espaços extras
    y_true = df_results["annotation"].str.lower().str.strip()

    # Detecta automaticamente a coluna de predição
    # (gemini_prediction ou gaia_prediction)
    prediction_cols = [col for col in df_results.columns if col.endswith("_prediction")]
    if not prediction_cols:
        logger.error("Nenhuma coluna de predição encontrada no DataFrame")
        return (None, None, None), {
            "model": "erro",
            "vectorization_method": vectorization_method,
            "undersampling": False,
            "cross_validation": False,
            "accuracy": 0.0,
            "f1_score": 0.0,
        }

    # Usa a primeira coluna de predição encontrada
    prediction_col = prediction_cols[0]
    model_name = prediction_col.replace("_prediction", "")

    y_pred = df_results[prediction_col].str.lower().str.strip()

    # Calcula e imprime as métricas
    accuracy, f1_score_value = calculate_metrics(y_true, y_pred)

    # Para manter a compatibilidade com o formato esperado
    undersampling = False
    cross_validation = False

    logger.info("Resultados da Classificação")
    logger.info(f"\tModelo: {model_name}")
    logger.info(f"\tvectorization_method: {vectorization_method}")
    logger.info(f"\tundersampling: {undersampling}")
    logger.info(f"\tcross_validation: {cross_validation}")
    logger.info(f"\tAcurácia Geral: {accuracy:.2%}")
    logger.info(f"\tF1-Score Geral: {f1_score_value:.2%}")

    dict_results = {
        "model": model_name,
        "vectorization_method": vectorization_method,
        "undersampling": undersampling,
        "cross_validation": cross_validation,
        "accuracy": accuracy,
        "f1_score": f1_score_value,
    }
    return (df_results["review_text"], y_true, y_pred), dict_results


def run_graphical_evaluation(
    df_eval: pd.DataFrame,
    dict_predictions_data: dict,
    dict_best_results: dict,
    plots_dir: str,
) -> None:
    """Função para gerar os gráficos de avaliação dos modelos de classificação.

    Parameters
    ----------
    df_eval : pd.DataFrame
        DataFrame contendo as métricas de avaliação dos modelos.
    dict_predictions_data : dict
        Dicionário contendo os dados de predição dos modelos.
    dict_best_results : dict
        Dicionário contendo os melhores resultados dos modelos.
    plots_dir : str
        Diretório onde os gráficos serão salvos.
    """

    # plots_dir = os.getenv("PLOTS_OUTPUT_DIRECTORY")
    if not os.path.exists(plots_dir):
        create_dir(plots_dir)

    # Acurácia
    plt.figure(figsize=(10, 6))
    plt.bar(df_eval["vectorization_method"], df_eval["accuracy"])
    plt.title("Acurácia dos Modelos")
    plt.xlabel("Método de Vetorização")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    for i, v in enumerate(df_eval["accuracy"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/acuracia_modelos.png", dpi=300, bbox_inches="tight")
    plt.close()

    # F1-Score
    plt.figure(figsize=(10, 6))
    plt.bar(df_eval["vectorization_method"], df_eval["f1_score"])
    plt.title("F1-Score dos Modelos")
    plt.xlabel("Método de Vetorização")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)
    for i, v in enumerate(df_eval["f1_score"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/f1score_modelos.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Matriz de Confusão
    best_method = dict_best_results["vectorization_method"]
    if best_method in dict_predictions_data:
        _, y_test_best, predictions_best = dict_predictions_data[best_method]

        # Gerar matriz de confusão
        cm = confusion_matrix(y_test_best, predictions_best)
        labels = sorted(list(set(y_test_best)))

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(
            f"Matriz de Confusão - Melhor Combinação\n"
            f'{dict_best_results["model"]} + {dict_best_results["vectorization_method"]}'
        )
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(
            f"{plots_dir}/matriz_confusao_melhor_modelo.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()
    else:
        logger.warning(
            f"Dados de predição não disponíveis para {best_method}. "
            "Matriz de confusão não será gerada."
        )

    logger.info(f"Gráficos salvos em: {plots_dir}")


def generate_confusion_matrix(
    dict_results: dict, dict_predictions_data: dict, output_path: str
) -> None:
    """Gera e salva uma matriz de confusão para o conjunto de dados especificado.

    Parameters
    ----------
    y_true : pd.Series
        Rótulos verdadeiros.
    y_pred : pd.Series
        Rótulos preditos.
    labels : list
        Lista de rótulos únicos.
    output_path : str
        Caminho para salvar a matriz de confusão.
    """

    # Cria o diretório se não existir
    if not os.path.exists(output_path):
        create_dir(output_path)
        
    # Procura o best_03 experiment in dict_results
    dict_best = None
    for elem in dict_results:
        if elem["experiment"] == "best_03":
            dict_best = elem
            break

    if dict_best is None:
        dict_best = max(dict_results, key=lambda x: x["f1_score"])

    vect_method = dict_best["vectorization_method"]
    if vect_method in dict_predictions_data:
        logger.info("Gerando matriz de confusão para a melhor combinação preditiva.")
        _, y_true, y_pred = dict_predictions_data[vect_method]
        labels = sorted(list(set(y_true)))

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(
            f"Matriz de Confusão {dict_best['dataset_name'].upper()}\n"
            f'{dict_best["model"].upper()} + {dict_best["vectorization_method"].upper()}'
        )
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(
            f"{output_path}/matriz_confusao_{dict_best['dataset_name'].lower()}.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()
        logger.info(
            f"Matriz de confusão salva para dataset {dict_best['dataset_name'].lower()}"
        )
        return
    else:
        logger.warning(
            f"Dados de predição não disponíveis para {vect_method}. "
            "Matriz de confusão não será gerada."
        )
        return


def generate_metrics_plot(all_results: list, output_path: str) -> None:
    """Gera gráficos de acurácia e F1-score comparando todas as bases de dados
    usando apenas os resultados do experimento 'best_03'.

    Parameters
    ----------
    all_results : list
        Lista de dicionários contendo todas as métricas dos experimentos.
    output_path : str
        Caminho onde os gráficos serão salvos.
    """

    # Filtra apenas os resultados do experimento 'best_03'
    best_03_results = [
        result for result in all_results if result.get("experiment") == "best_03"
    ]

    if not best_03_results:
        logger.warning("Nenhum resultado encontrado para o experimento 'best_03'.")
        return

    # Cria o diretório se não existir
    if not os.path.exists(output_path):
        create_dir(output_path)

    # Extrai dados para os gráficos
    datasets = [result["dataset_name"] for result in best_03_results]
    accuracies = [result["accuracy"] for result in best_03_results]
    f1_scores = [result["f1_score"] for result in best_03_results]

    # Gráfico de Acurácia
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, accuracies, color="#1f77b4")
    plt.title("Acurácia por base de dados")
    plt.xlabel("Base de Dados")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)

    # Adiciona valores nas barras
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/metrica_acuracia_bases.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Gráfico de F1-Score
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, f1_scores, color="#1f77b4")
    plt.title("F1-Score por base de dados")
    plt.xlabel("Base de Dados")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)

    # Adiciona valores nas barras
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/metrica_f1score_bases.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info(f"Gráficos de comparação entre bases salvos em: {output_path}")
