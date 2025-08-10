import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import time
from dotenv import load_dotenv

import src.llm_classification as llm_cls
import src.log_config  # noqa: F401
import src.prediction as am_classif
import src.result_evaluation as eval_cls
import src.vectorization as vect
from src.data_analysis import data_analysis_pipeline
from src.pre_processing import run_preprocessing_pipeline
from src.utils import create_dir, save_dataframe

logger = logging.getLogger(__name__)

load_dotenv(override=True)
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))


def do_am_classif(
    dict_param_grid: dict,
    vectors_word2vec: np.ndarray,
    vectors_bert: np.ndarray,
    cleaned_df: pd.DataFrame,
    raw_dataset_name: str,
) -> Tuple[list, list, dict]:
    """Executa os experimentos de classificação usando Aprendizado de Máquina (AM).

    Parameters
    ----------
    dict_param_grid : dict
        Dicionário contendo os parâmetros para execução dos experimentos.
    vectors_word2vec : np.ndarray
        Vetor de características gerado pelo Word2Vec.
    vectors_bert : np.ndarray
        Vetor de características gerado pelo BERT.
    cleaned_df : pd.DataFrame
        DataFrame contendo os dados limpos para treinamento e teste.    
    raw_dataset_name : str
        Nome do conjunto de dados original.

    Returns
    -------
    Tuple[list, list, dict]
        Tupla contendo os resultados dos experimentos AM, os resultados gerais 
        e um dicionário com as predições.
    """

    best_am_method = None
    best_am_vectors = None
    am_results = []
    results = []
    dict_am_predictions = {}

    for params in dict_param_grid:
        if not params.get("run_am_classification", False):
            continue

        exp_name = params["exp_name"]
        logger.info(f"{exp_name}: Iniciando experimento AM")

        # Determina qual conjunto de vetores usar
        vectorization_method = params["vectorization_method"]
        if vectorization_method == "best_from_exp_01_exp_02":
            # Usa o melhor método determinado anteriormente
            if best_am_method is None:
                logger.warning(
                    f"Melhor método AM ainda não determinado para {exp_name}"
                )
                continue
            vectorization_method = best_am_method
            vectors = best_am_vectors
        elif vectorization_method == "w2v":
            vectors = vectors_word2vec
        elif vectorization_method == "bert":
            vectors = vectors_bert
        else:
            logger.error(f"Método de vetorização desconhecido: {vectorization_method}")
            continue

        # Executa classificação com AM
        predictions_data, metrics = am_classif.do_classification(
            vectors,
            cleaned_df["annotation"],
            vectorization_method,
            undersampling=params.get("undersampling", False),
            cross_validation=params.get("crossvalidation", False),
        )

        # Armazena dados de predição para gráficos
        dict_am_predictions[vectorization_method] = predictions_data

        metrics["experiment"] = exp_name
        metrics["dataset_name"] = raw_dataset_name
        results.append(metrics)
        am_results.append(metrics)

        # Atualiza melhor método AM (para exp_03)
        if len(am_results) == 2:  # Após exp_01 e exp_02
            best_am_exp = max(am_results, key=lambda x: x["f1_score"])
            best_am_method = best_am_exp["vectorization_method"]
            best_am_vectors = (
                vectors_word2vec if best_am_method == "w2v" else vectors_bert
            )
            logger.info(
                f"Melhor AM até agora: {best_am_exp['experiment']} ({best_am_method})"
            )

    # Determina melhor AM final
    if am_results:
        best_am_final = max(am_results, key=lambda x: x["f1_score"])
        best_am_final_copy = best_am_final.copy()
        best_am_final_copy["experiment"] = "best_01"
        results.append(best_am_final_copy)
        logger.info(
            f"best_01: Melhor AM final = {best_am_final['experiment']} (F1: {best_am_final['f1_score']:.4f})"
        )

    return am_results, results, dict_am_predictions


def do_llm_classif(
    dict_param_grid: dict,
    cleaned_df: pd.DataFrame,
    raw_dataset_name: str,
    results_path: str = "",
) -> Tuple[list, list, dict]:
    """Executa os experimentos de classificação usando LLMs.

    Parameters
    ----------
    dict_param_grid : dict
        Dicionário contendo os parâmetros para execução dos experimentos.
    cleaned_df : pd.DataFrame
        DataFrame contendo os dados limpos para treinamento e teste.
    raw_dataset_name : str
        Nome do conjunto de dados original.
    results_path : str, optional
        Caminho para salvar os resultados, by default "".

    Returns
    -------
    Tuple[list, list, dict]
        Uma tupla contendo os resultados dos experimentos, os resultados
        específicos dos LLMs e um dicionário com as predições.
    """

    results = []
    llm_results = []
    dict_llm_predictions = {}

    for params in dict_param_grid:
        if not params.get("run_llm_classification", False):
            continue

        exp_name = params["exp_name"]
        logger.info(f"{exp_name}: Iniciando experimento LLM")

        # Preparar few-shot
        is_absolute = params.get("is_absolute_few_shot", True)
        if is_absolute:
            n_samples = params.get("few_shot_n", 3)
            df_few_shot, dict_few_shot = llm_cls.resample_few_shot(
                cleaned_df, is_absolute=True, n=n_samples
            )
        else:
            frac_samples = params.get("few_shot_frac", 0.001)
            df_few_shot, dict_few_shot = llm_cls.resample_few_shot(
                cleaned_df, is_absolute=False, frac=frac_samples
            )

        # Remove os exemplos do dataframe principal para não
        # serem classificados novamente
        df_for_sampling = cleaned_df.drop(df_few_shot.index).copy()

        # # Seleciona amostras aleatórias para classificação, caso o dataset
        # # seja muito grande
        # # TODO : talvez isso aqui não seja necessário
        # sampling_threshold = 100_000  # LMT_DATA - limite de amostras
        # if len(df_for_sampling) > sampling_threshold:
        #     data_limit = 50_000
        #     logger.info(
        #         f"Conjunto possui mais de {sampling_threshold:,} amostras. "
        #         f"Seleciona {data_limit:,} das amostras para classificação."
        #     )
        #     logger.info(
        #         f"Dimensão do conjunto de dados antes da seleção: "
        #         f"{len(df_for_sampling)}"
        #     )
        #     # LMT_DATA - limitado para 15 amostras, descomentar linha abaixo 
        #     # para comportamento correto do sistema

        #     # df_sample = df_for_sampling.sample(n=data_limit, random_state=RANDOM_STATE)
        #     df_sample = df_for_sampling.sample(n=15, random_state=RANDOM_STATE)
        #     logger.info(
        #         f"Dimensão do conjunto de dados após a seleção: {len(df_sample)}"
        #     )
        # else:
        #     logger.info(
        #         "Conjunto de dados aceitável. Todos os dados serão "
        #         "utilizados para classificação."
        #     )
        #     # LMT_DATA Remover o head ao final - limitado para 15 amostras
        #     # df_sample = df_for_sampling
        #     df_sample = df_for_sampling.head(15)
        #     logger.info(f"Dimensão do conjunto de dados: {len(df_sample)}")

        df_sample = df_for_sampling.copy()  # Para manter compatibilidade com o código.
        del df_for_sampling
        reviews_to_classify = df_sample["review_text"]

        # Executa classificação LLM
        llm_model = params.get("llm_model", "gemini")

        logger.info(f"Classificação com o {llm_model.upper()} iniciada.")
        create_dir(results_path)

        if llm_model == "gemini":
            predictions = llm_cls.run_gemini_classification(
                reviews_to_classify, dict_few_shot
            )
        elif llm_model == "gemma":
            predictions = llm_cls.run_ollama_classification(
                reviews_to_classify, dict_few_shot
            )
        else:
            logger.error(f"Modelo LLM não suportado: {llm_model}")
            continue

        logger.info("Em espera para rodar a próxima iteração")
        time.sleep(70)  
        logger.info(f"Continuação da classificação com o {llm_model.upper()}")

        if predictions:
            df_results = df_sample.drop(columns=["review_text_processed"]).copy()
            df_results[f"{llm_model}_prediction"] = predictions

            results_path = os.getenv("RESULTS_PATH")
            results_llm_path = os.path.join(
                results_path, f"classificacao_{llm_model}_{raw_dataset_name}.csv"
            )
            save_dataframe(df_results, results_llm_path)

            # Atualiza informações do experimento
            vect_method_name = f"fewShot_{'absoluto' if is_absolute else 'relativo'}"

            logger.info(f"Calculando métricas da classificação {llm_model.upper()}.")
            predictions_data, metrics = eval_cls.calculate_llm_metrics(
                df_results, vectorization_method=vect_method_name
            )

            # Armazena dados de predição para gráficos
            dict_llm_predictions[vect_method_name] = predictions_data
            metrics.update(
                {
                    "model": llm_model,
                    "vectorization_method": vect_method_name,
                    "undersampling": False,
                    "cross_validation": False,
                    "experiment": exp_name,
                    "dataset_name": raw_dataset_name,
                }
            )

            results.append(metrics)
            llm_results.append(metrics)
        else:
            logger.warning(
                f"A classificação com o {llm_model.upper()} não foi executada "
                f"devido a um erro."
            )
            continue

    # Determina melhor fewShot com LLM
    if llm_results:
        best_llm_final = max(llm_results, key=lambda x: x["f1_score"])
        best_llm_final_copy = best_llm_final.copy()
        best_llm_final_copy["experiment"] = "best_02"
        results.append(best_llm_final_copy)
        logger.info(
            f"best_02: Melhor combinação LLM = {best_llm_final['experiment']} "
            f"(F1: {best_llm_final['f1_score']:.4f})"
        )

    return llm_results, results, dict_llm_predictions


def run_flexible_experiments(
    dataset_name: str, dict_param_grid: list, enable_analysis: bool = False
) -> Tuple[list, dict]:
    """Executa todos os experimentos de classificação com Aprendizado de 
    Máquina (AM) e LLMs para um conjunto de dados específico.

    Parameters
    ----------
    dataset_name : str
        Nome do conjunto de dados a ser utilizado nos experimentos.
    dict_param_grid : list
        Dicionário contendo os parâmetros para execução dos experimentos.
    enable_analysis : bool, optional
        Habilita a análise e visualização dos dados, by default False.

    Returns
    -------
    Tuple[list, dict]
        Lista com todos os resultados dos experimentos e um dicionário com 
        as métricas.
    """

    raw_dataset_name = dataset_name.split(".")[0]

    raw_data_path = os.getenv("INPUT_RAW_DATA_PATH")
    clean_data_path = os.path.join(
        os.getenv("CLEAN_DATA_PATH"), f"dados_processados_{raw_dataset_name}.csv"
    )
    plots_path = f"{os.getenv('PLOTS_OUTPUT_DIRECTORY')}_{raw_dataset_name}"
    results_path = os.getenv("RESULTS_PATH")

    vectors_path = os.getenv("VECTORS_DATA_PATH")
    vect_w2v_filename = f"w2v_vectors_{raw_dataset_name}.npy"
    vect_bert_filename = f"bert_vectors_{raw_dataset_name}.npy"

    dataSet_results = []

    # Tratamento dos dados
    if not os.path.exists(os.path.join(raw_data_path, dataset_name)):
        logger.error(
            f"Conjunto de dados {dataset_name} não encontrado no caminho"
            " especificado. Verifique se o caminho existe e se os dados estão"
            " corretamente armazenados."
        )
        return None, None
    else:
        if not os.path.exists(clean_data_path):
            logger.info("Pré-processamento dos dados iniciado.")
            cleaned_df = run_preprocessing_pipeline(
                os.path.join(raw_data_path, dataset_name)
            )
            logger.info("Pré-processamento dos dados concluído.")
            if cleaned_df is not None:
                # Salva o resultado do pré-processamento
                save_dataframe(cleaned_df, clean_data_path)
            else:
                logger.error(
                    "Pré-processamento falhou. As etapas seguintes não serão "
                    "executadas."
                )
                return None, None

        # Carregamento dos dados pré-processados
        cleaned_df = pd.read_csv(clean_data_path)

        if cleaned_df is not None:
            if enable_analysis:
                logger.info("Análise e visualização dos dados iniciada.")
                data_analysis_pipeline(df=cleaned_df, output_dir=plots_path)
                logger.info("Análise e visualização dos dados finalizada.")
                logger.info("Somente as análises descritivas foram realizadas.")
                return None, None

            if not os.path.exists(vectors_path):
                create_dir(vectors_path)

            # Vetorização com Word2Vec
            if not os.path.exists(os.path.join(vectors_path, vect_w2v_filename)):
                logger.info("Vetorização com Word2Vec iniciada.")
                vectors_word2vec = vect.vectorize_with_word2vec(
                    texts=cleaned_df["review_text"].tolist()
                )
                np.save(os.path.join(vectors_path, vect_w2v_filename), vectors_word2vec)
                logger.info("Vetorização com Word2Vec finalizada.")
            else:
                logger.info("Vetores Word2Vec já existem! Carregando...")
                vectors_word2vec = np.load(
                    os.path.join(vectors_path, vect_w2v_filename), allow_pickle=True
                )

            # Vetorização com BERT
            if not os.path.exists(os.path.join(vectors_path, vect_bert_filename)):
                logger.info("Vetorização com BERT iniciada.")
                vectors_bert = vect.vectorize_with_bert(
                    texts=cleaned_df["review_text"].tolist()
                )
                np.save(os.path.join(vectors_path, vect_bert_filename), vectors_bert)
                logger.info("Vetorização com BERT finalizada.")
            else:
                logger.info("Vetores BERT já existem! Carregando...")
                vectors_bert = np.load(
                    os.path.join(vectors_path, vect_bert_filename), allow_pickle=True
                )

            logger.info(f"Experimentos para {dataset_name} iniciados.")

            # Reduz o conjunto de dados
            # LMT_DATA
            sample_size = 10_000
            if len(cleaned_df) > sample_size:
                logger.warning(
                    f"O conjunto de dados {raw_dataset_name} possui mais de "
                    f"{sample_size:,} amostras ({len(cleaned_df):,}). "
                    f"Realizando subamostragem."
                )
                np.random.seed(RANDOM_STATE)
                indices = np.random.choice(
                    len(cleaned_df), size=sample_size, replace=False
                )
                cleaned_df = cleaned_df.iloc[indices]
                vectors_word2vec = (
                    vectors_word2vec.iloc[indices]
                    if hasattr(vectors_word2vec, "iloc")
                    else vectors_word2vec[indices]
                )
                vectors_bert = (
                    vectors_bert.iloc[indices]
                    if hasattr(vectors_bert, "iloc")
                    else vectors_bert[indices]
                )

            logger.info("---> EXPERIMENTOS AM:")
            am_results, results_01, dict_am_predictions = do_am_classif(
                dict_param_grid, vectors_word2vec, vectors_bert, cleaned_df, raw_dataset_name
            )
            dataSet_results.extend(results_01)

            logger.info("---> EXPERIMENTOS LLM:")
            llm_results, results_02, dict_llm_predictions = do_llm_classif(
                dict_param_grid, cleaned_df, raw_dataset_name, results_path
            )
            dataSet_results.extend(results_02)

            # Combina dicionários de predições para gráficos
            dict_all_predictions = {**dict_am_predictions, **dict_llm_predictions}

            # ========== COMPARAÇÃO FINAL ==========
            if am_results and llm_results:
                best_am = max(am_results, key=lambda x: x["f1_score"])
                best_llm = max(llm_results, key=lambda x: x["f1_score"])

                if best_am["f1_score"] > best_llm["f1_score"]:
                    winner = best_am.copy()
                    winner_name = best_am["experiment"]
                else:
                    winner = best_llm.copy()
                    winner_name = best_llm["experiment"]

                winner["experiment"] = "best_03"
                dataSet_results.append(winner)
                logger.info(
                    f"best_03: {winner_name} "
                    f"({winner['model']} + {winner['vectorization_method']}) - "
                    f"(F1: {winner['f1_score']:.4f})"
                )

    return dataSet_results, dict_all_predictions


def run_all_experiments() -> None:
    """Executa todos os experimentos para todos os conjuntos 
    de dados especificados.
    """

    # datasets = ["b2w.csv", "utlc_apps.csv", "utlc_movies.csv"]
    datasets = ["b2w.csv"]
    all_results = []
    dict_all_predictions = {}

    dict_param_grid = [
        # Experimentos AM
        {
            "exp_name": "exp_01",
            "vectorization_method": "w2v",
            "run_am_classification": True,
            "crossvalidation": False,
            "run_llm_classification": False,
            "undersampling": False,
        },
        {
            "exp_name": "exp_02",
            "vectorization_method": "bert",
            "run_am_classification": True,
            "crossvalidation": False,
            "run_llm_classification": False,
            "undersampling": False,
        },
        {
            "exp_name": "exp_03",
            "vectorization_method": "best_from_exp_01_exp_02",  # Será determinado dinamicamente
            "run_am_classification": True,
            "crossvalidation": True,
            "run_llm_classification": False,
            "undersampling": False,
        },

        # Experimentos LLM
        {
            "exp_name": "exp_04",
            "run_am_classification": False,
            "run_llm_classification": True,
            "is_absolute_few_shot": True,
            "llm_model": "gemini",
            "few_shot_n": 5,
        },
        {
            "exp_name": "exp_05",
            "run_am_classification": False,
            "run_llm_classification": True,
            "is_absolute_few_shot": False,
            "llm_model": "gemini",
            "few_shot_frac": 0.01,
        },
    ]

    for dataset in datasets:
        raw_dataset_name = dataset.split(".")[0]
        plots_path = f"{os.getenv('PLOTS_OUTPUT_DIRECTORY')}_{raw_dataset_name}"
        logger.info(f"Processando {dataset}...")

        # Executa experimentos comparativos baseados no dict_param_grid
        dataset_results, dict_dataset_predictions = run_flexible_experiments(
            dataset, dict_param_grid, enable_analysis=False
        )
        if not dataset_results:
            logger.warning(f"Nenhum resultado encontrado para {dataset}.")
            continue

        all_results.extend(dataset_results)
        dict_all_predictions[raw_dataset_name] = dict_dataset_predictions

        # Gera matriz de confusão para a melhor combinação preditiva no dataset
        eval_cls.generate_confusion_matrix(
            dataset_results, dict_dataset_predictions, plots_path
        )

    if not all_results:
        logger.warning(
            "Nenhum resultado foi gerado nos experimentos. Talvez "
            "tenha sido executada apenas a análise descritiva."
        )
    else:
        # Gerar gráficos de avaliação
        logger.info("Avaliação gráfica dos resultados iniciada.")
        eval_cls.generate_metrics_plot(
            all_results,
            os.getenv("PLOTS_OUTPUT_DIRECTORY")
        )

        logger.info("Experimentos concluídos com sucesso.")

        # Salva todos os resultados em CSV único
        results_path = os.getenv("RESULTS_PATH")
        csv_geral = os.path.join(results_path, "experimentos_completos.csv")
        df_geral = pd.DataFrame(all_results)
        df_geral.to_csv(csv_geral, index=False)
        logger.info(f"Todos os resultados salvos em {csv_geral}")


def run_all_experiments_multi(number_executions: int) -> None:
    """Executa todos os experimentos para todos os conjuntos
    de dados especificados.
    """

    # datasets = ["b2w.csv", "utlc_apps.csv", "utlc_movies.csv"]
    datasets = ["b2w.csv"]
    all_runs_results = []

    dict_param_grid = [
        # Experimentos AM
        {
            "exp_name": "exp_01",
            "vectorization_method": "w2v",
            "run_am_classification": True,
            "crossvalidation": False,
            "run_llm_classification": False,
            "undersampling": False,
        },
        {
            "exp_name": "exp_02",
            "vectorization_method": "bert",
            "run_am_classification": True,
            "crossvalidation": False,
            "run_llm_classification": False,
            "undersampling": False,
        },
        {
            "exp_name": "exp_03",
            "vectorization_method": "best_from_exp_01_exp_02",  # Será determinado dinamicamente
            "run_am_classification": True,
            "crossvalidation": True,
            "run_llm_classification": False,
            "undersampling": False,
        },

        # Experimentos LLM
        {
            "exp_name": "exp_04",
            "run_am_classification": False,
            "run_llm_classification": True,
            "is_absolute_few_shot": True,
            "llm_model": "gemma",
            "few_shot_n": 5,  # mudar para 5
        },
        {
            "exp_name": "exp_05",
            "run_am_classification": False,
            "run_llm_classification": True,
            "is_absolute_few_shot": False,
            "llm_model": "gemma",
            "few_shot_frac": 0.005,
        },
    ]

    for run in range(number_executions):
        logger.info(f"Iniciando execução {run + 1} de {number_executions}...")

        all_results = []
        dict_all_predictions = {}

        # Processa cada dataset

        for dataset in datasets:
            raw_dataset_name = dataset.split(".")[0]
            plots_path = f"{os.getenv('PLOTS_OUTPUT_DIRECTORY')}_{raw_dataset_name}"
            logger.info(f"Processando {dataset}...")

            # Executa experimentos comparativos baseados no dict_param_grid
            dataset_results, dict_dataset_predictions = run_flexible_experiments(
                dataset, dict_param_grid, enable_analysis=False
            )
            if not dataset_results:
                logger.warning(f"Nenhum resultado encontrado para {dataset}.")
                continue

            all_results.extend(dataset_results)
            dict_all_predictions[raw_dataset_name] = dict_dataset_predictions

            # Gera matriz de confusão para a melhor combinação preditiva no dataset
            eval_cls.generate_confusion_matrix(
                dataset_results, dict_dataset_predictions, plots_path
            )
        
        if all_results:
            df_run = pd.DataFrame(all_results)
            df_run["run"] = run + 1  # Adiciona coluna indicando a rodada
            all_runs_results.append(df_run)

    if not all_runs_results:
        logger.warning(
            "Nenhum resultado foi gerado nos experimentos. Talvez "
            "tenha sido executada apenas a análise descritiva."
        )
    else:
        logger.info("Experimentos concluídos com sucesso.")

        # Concatena todos os resultados
        df_all = pd.concat(all_runs_results, ignore_index=True)

        # Gerar gráficos de avaliação
        logger.info("Avaliação gráfica dos resultados iniciada.")
        eval_cls.generate_metrics_plot(
            df_all.to_dict(orient="records"),
            os.getenv("PLOTS_OUTPUT_DIRECTORY")
        )
        logger.info("Experimentos concluídos com sucesso.")

        # Salva todos os resultados em CSV único
        results_path = os.getenv("RESULTS_PATH")
        csv_geral = os.path.join(results_path, "experimentos_completos_multirun.csv")
        df_all.to_csv(csv_geral, index=False)
        logger.info(f"Todos os resultados salvos em {csv_geral}")

        # Calcula média e desvio padrão das métricas por experimento/base
        metrics = ["accuracy", "f1_score"]  # Adapte conforme suas métricas
        group_cols = ["experiment", "dataset_name", "model", "vectorization_method"]
        df_stats = df_all.groupby(group_cols)[metrics].agg(['mean', 'std']).reset_index()
        csv_stats = os.path.join(results_path, "experimentos_stats.csv")
        df_stats.to_csv(csv_stats, index=False)
        logger.info(f"Médias e desvios padrão salvos em {csv_stats}")


if __name__ == "__main__":
    run_all_experiments()
    # run_all_experiments_multi(3)
