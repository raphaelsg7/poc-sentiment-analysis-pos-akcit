import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import src.log_config  # noqa: F401

logger = logging.getLogger(__name__)
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))


def encode_labels(labels: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """Codifica rótulos de classe em valores numéricos.

    Parameters
    ----------
    labels : pd.Series
        Rótulos de classe a serem codificados.

    Returns
    -------
    Tuple[np.ndarray, LabelEncoder]
        Tupla contendo os rótulos codificados e o LabelEncoder usado
        para codificação.
    """

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    return encoded_labels, label_encoder


def train_model(X_train: np.ndarray, y_train: np.ndarray, model: any) -> any:
    """Treina o modelo passado com os dados de treino.

    Parameters
    ----------
    X_train : np.ndarray
        Dados de treinamento.
    y_train : np.ndarray
        Rótulos de classe para os dados de treinamento.
    model : any
        Modelo a ser treinado.

    Returns
    -------
    any
        Modelo treinado.
    """

    model.fit(X_train, y_train)
    return model


def predict(model: any, X_test: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
    """Realiaza predições a partir do modelo treinado.

    Parameters
    ----------
    model : any
        Modelo treinado passado para realizar as predições.
    X_test : np.ndarray
        Dados de teste para realizar as predições.
    label_encoder : LabelEncoder
        LabelEncoder usado para decodificar as predições.

    Returns
    -------
    np.ndarray
        Rótulos de classe decodificados a partir das predições.
    """

    predictions_encoded = model.predict(X_test)
    predictions_decoded = label_encoder.inverse_transform(predictions_encoded)

    return predictions_decoded


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Realiza o cálculo das métricas de avaliação definidas.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros dos dados de teste.
    y_pred : np.ndarray
        Rótulos das classes preditas pelo modelo.

    Returns
    -------
    Tuple[float, float]
        Tupla contendo a acurácia e o F1-Score das predições.
    """

    accuracy_value = accuracy_score(y_true, y_pred)
    f1_score_value = f1_score(y_true, y_pred, average="weighted")

    return accuracy_value, f1_score_value


def undersample_data(
    X_data: np.ndarray, y_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica a técnica de Random Undersampling para balancear as classes.

    Parameters
    ----------
    X_data : np.ndarray
        Dados do vetor X para serem balanceados.
    y_data : np.ndarray
        Dados do vetor y (rótulos) para serem balanceados.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Dados balanceados do vetor X e rótulos balanceados do vetor y.
    """

    rus = RandomUnderSampler(random_state=RANDOM_STATE)

    X_resampled, y_resampled = rus.fit_resample(X_data, y_data)

    return X_resampled, y_resampled


def do_classification(
    X_data: np.ndarray,
    y_data: np.ndarray,
    vectorization_method: str = "",
    undersampling: bool = False,
    cross_validation: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    """Executa o pipeline de classificação.

    Notes
    -----
    Quando a validação cruzada é ativada, os dados de treinamento são retornados
    no lugar dos dados de teste, pois a validação cruzada não separa os dados
    de teste e treinamento como o método `train_test_split`.

    Parameters
    ----------
    X_data : np.ndarray
        Dados do vetor X para classificação.
    y_data : np.ndarray
        Dados do vetor y (rótulos) para classificação.
    vectorization_method : str, optional
        Método de vetorização a ser utilizado, por padrão "".
    undersampling : bool, optional
        Ativa ou desativa o balanceamento de classes, por padrão False.
    cross_validation : bool, optional
        Ativa ou desativa a validação cruzada, por padrão False.

    Returns
    -------
    Tuple[Tuple[np.ndarray, pd.Series, np.ndarray], dict]
        Tupla contendo os dados de teste, rótulos de teste e predições,
        e um dicionário com os resultados da classificação.
    """

    model = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)

    if cross_validation:
        logger.info("Iniciando etapa de validação cruzada.")
        if undersampling:
            logger.info("Aplicando Random Undersampling para balancear as classes.")
            X_data, y_data = undersample_data(X_data, y_data)

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        predictions = cross_val_predict(model, X_data, y_data, cv=cv, n_jobs=-1)
        accuracy, f1_score_value = calculate_metrics(y_data, predictions)

        logger.info("Resultados da Classificação")
        logger.info(f"\tvectorization_method: {vectorization_method}")
        logger.info(f"\tundersampling: {undersampling}")
        logger.info(f"\tcross_validation: {cross_validation}")
        logger.info(f"\tAcurácia Geral: {accuracy:.2%}")
        logger.info(f"\tF1-Score Geral: {f1_score_value:.2%}")

        dict_results = {
            "model": model.__class__.__name__.lower(),
            "vectorization_method": vectorization_method,
            "undersampling": undersampling,
            "cross_validation": cross_validation,
            "accuracy": accuracy,
            "f1_score": f1_score_value,
        }

        return (None, y_data, predictions), dict_results

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.3, random_state=RANDOM_STATE, shuffle=True
        )

        if undersampling:
            logger.info("Aplicando Random Undersampling para balancear as classes.")
            X_train, y_train = undersample_data(X_train, y_train)

        y_train_encoded, label_encoder = encode_labels(y_train)
        trained_model = train_model(X_train, y_train_encoded, model)
        predictions = predict(trained_model, X_test, label_encoder)
        accuracy, f1_score_value = calculate_metrics(y_test, predictions)

        logger.info("Resultados da Classificação")
        logger.info(f"\tvectorization_method: {vectorization_method}")
        logger.info(f"\tundersampling: {undersampling}")
        logger.info(f"\tcross_validation: {cross_validation}")
        logger.info(f"\tAcurácia Geral: {accuracy:.2%}")
        logger.info(f"\tF1-Score Geral: {f1_score_value:.2%}")

        dict_results = {
            "model": model.__class__.__name__.lower(),
            "vectorization_method": vectorization_method,
            "undersampling": undersampling,
            "cross_validation": cross_validation,
            "accuracy": accuracy,
            "f1_score": f1_score_value,
        }

        return (X_test, y_test, predictions), dict_results
