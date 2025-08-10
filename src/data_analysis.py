# src/visualization.py

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

import src.log_config  # noqa: F401

logger = logging.getLogger(__name__)


def data_analysis_pipeline(
    df: pd.DataFrame, output_dir: str = "plots", top_n_locations: int = 15
):

    # Executa um pipeline completo de analise de dados, gerando e salvando todos os gráficos

    logger.info("Iniciando Pipeline de Visualização de Dados")

    # Garante que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    # Gráfico de Distribuição de Sentimentos
    logger.info("Gerando gráfico de distribuição de sentimentos...")
    plt.figure(figsize=(8, 6))
    sns.countplot(
        x="annotation",
        data=df,
        palette="viridis",
        order=df["annotation"].value_counts().index,
    )
    plt.title("Distribuição dos Sentimentos no Dataset")
    plt.xlabel("Polaridade")
    plt.ylabel("Contagem de reviews")
    plt.savefig(os.path.join(output_dir, "distribuicao_sentimentos.png"))
    # plt.show()
    plt.close()


    # Nuvens de Palavras por Sentimento
    for polarity_description in df["annotation"].unique():
        logger.info(f"Gerando nuvem de palavras para sentimentos '{polarity_description}'...")

        text = " ".join(
            reviews
            for reviews in df[df["annotation"] == polarity_description]["review_text_processed"]
            if isinstance(reviews, str)
        )

        if not text:
            logger.warning(f"Não há dados para gerar a nuvem de palavras de '{polarity_description}'.")
            continue

        wordcloud = WordCloud(
            background_color="white",
            max_words=80,
            contour_width=3,
            contour_color="steelblue",
            width=800,
            height=600,
        ).generate(text)

        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nuvem de Palavras - Review {polarity_description}")
        plt.savefig(os.path.join(output_dir, f"wordcloud_{polarity_description.lower()}.png"))
        # plt.show()
        plt.close()

    # Histograma do Tamanho da Frase (Review) - Ajuste para visualização agrupada
    logger.info("Gerando histograma do tamanho das frases (reviews) com barras agrupadas e visíveis...")
    df_copy = df.copy()
    df_copy["comprimento_frase"] = df_copy["review_text_processed"].astype(str).apply(len)

    plt.figure(figsize=(12, 7))
    sns.histplot(
        df_copy["comprimento_frase"],
        bins=range(0, int(df_copy["comprimento_frase"].quantile(0.99) + 50), 50), # Define bins a cada 50 caracteres até o 99º percentil + um extra
        kde=True,         # Mantém o KDE
        color='skyblue',
        edgecolor='black',
        stat='density'    # Mantém densidade para a linha e barras consistentes
    )
    plt.title("Distribuição do Comprimento das Frases (Reviews)")
    plt.xlabel("Número de Caracteres na Frase")
    plt.ylabel("Densidade")

    max_x_value = df_copy["comprimento_frase"].quantile(0.99) 
    plt.xlim(0, max_x_value + (max_x_value * 0.1)) 

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "histograma_tamanho_frases_agrupado.png"))
    # plt.show()
    plt.close()