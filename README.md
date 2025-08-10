# PoC - Análise de Sentimentos em Reviews em diferentes domínios

Repositório público para armazenamento da POC de desenvolvimento do TCC da Pós-Graduação em NLP, do AKCIT-UFG

## Descrição

Este projeto implementa uma prova de conceito  de um fluxo algorítmicio para análise de sentimentos de avaliações de usuários, comparando diferentes abordagens de vetorização de texto (Word2Vec, BERT) e métodos de classificação (Aprendizado de Máquina e Grandes Modeos de Linguagem - LLM).

## Objetivos

- Comparar a eficácia e robustez de técnicas clássicas de processamento de linguagem natural com uma abordagem baseada em LLMs, na classificação de sentimentos, quanto à polaridade expressa, em textos de avaliações de usuários em diferentes domínios.
- Comparar a eficácia de diferentes métodos de vetorização de palavras, usando uma abordagem de geração de vetores estáticos com vetores contextuais
- Avaliar se o desempenho de LLM é melhor que AM na classificação de sentimentos
- Implementar um fluxo algorítmico reprodutível para a classificação de sentimentos em avaliações de usuários em português

## Arquitetura do Sistema

### Módulos de Código

1. **Pré-processamento** (`src/pre_processing.py`)
   - Limpeza de texto
   - Remoção de stopwords
   - Lematização com spaCy
   - Normalização de texto

2. **Vetorização** (`src/vectorization.py`)
   - Word2Vec pré-treinado em português brasileiro (skip-gram 300 dimensões - NILC-USP)
   - BERT português (neuralmind/bert-large-portuguese-cased)

3. **Classificação** 
   - **AM**: Random Forest (`src/prediction.py`)
   - **LLM**: Gemini 2.0 Flash (`src/llm_classification.py`)

4. **Avaliação e Visualização** (`src/result_evaluation.py`)
   - Métricas: Acurácia, F1-Score
   - Matriz de confusão
   - Gráficos comparativos
  
5. **Fluxo de Experimentos** (`main.py`)
   - Experimentos automatizados e parametrizados
   - Comparação entre diferentes configurações
   - Validação cruzada e undersampling

## Dados ([fonte](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets/data))

- **B2W**: Avaliações de produtos da B2W
- **UTLC Apps**: Avaliações de aplicativos
- **UTLC Movies**: Avaliações de filmes

## Configuração do Ambiente

### Pré-requisitos

- Python 3.11.10
- CUDA (opcional, para aceleração GPU)
- Chave da API Google Gemini
- Dados brutos baixados na pasta `data/raw`.

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/raphaelsg7/poc-sentiment-analysis-pos-akcit.git
cd poc-sentiment-analysis-pos-akcit
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Configure as variáveis de ambiente:**  
Criar um arquivo `.env` na raiz do projeto:
```env
GOOGLE_API_KEY=sua_chave_da_api_gemini
RANDOM_STATE=42
INPUT_RAW_DATA_PATH=./data/raw
CLEAN_DATA_PATH=./data/treated
VECTORS_DATA_PATH=./data/vectors
PLOTS_OUTPUT_DIRECTORY=./plots
RESULTS_PATH=./results
```

## Uso do Sistema

### Fluxo de Experimentos

```python
from exp import run_all_experiments

# Executa todos os experimentos automatizados
results_df = run_all_experiments()
```

### Estrutura de Experimentos

O framework executa 5 experimentos principais:

- **exp_01**: Classificação AM com Word2Vec
- **exp_02**: Classificação AM com BERT  
- **exp_03**: Melhor AM com validação cruzada
- **exp_04**: LLM com few-shot absoluto
- **exp_05**: LLM with few-shot relativo

E 3 comparações:

- **best_01**: Melhor resultado entre exp_01, exp_02 e exp_03
- **best_02**: Melhor resultado LLM
- **best_03**: Melhor resultado Melhor resultado geral (AM vs LLM)

## Estrutura do Projeto

```
poc-sentiment-analysis-pos-akcit/
├── data/
│   ├── raw/                    # Dados brutos
│   ├── treated/                # Dados pré-processados
│   └── vectors/                # Vetores salvos
├── src/
│   ├── data_analysis.py        # Análise exploratória
│   ├── llm_classification.py   # Classificação com LLM
│   ├── log_config.py           # Configuração de logs
│   ├── pre_processing.py       # Pré-processamento
│   ├── prediction.py           # Classificação AM
│   ├── result_evaluation.py    # Avaliação e métricas
│   ├── utils.py                # Utilitários
│   └── vectorization.py        # Vetorização de texto
├── models/                     # Modelos Word2Vec
├── results/                    # Gráficos e Resultados dos experimentos
├── main.py                     # Fluxo principal de execução
├── requirements.txt            # Dependências
└── README.md
```

## Saídas do Sistema

### Arquivos de Resultados

- `experimentos_completos.csv`: Métricas de todos os experimentos
- `classificacao_gemini_{dataset}.csv`: Resultados específicos por LLM

### Visualizações

- **Gráficos de Métricas**: Acurácia e F1-Score por método
- **Matriz de Confusão**: Para o melhor modelo
- **Análise Exploratória**: Distribuição de sentimentos, nuvens de palavras
- **Histogramas**: Comprimento dos textos

## Resultados Esperados

O sistema permite comparar:

1. **Métodos de Vetorização**: Word2Vec vs. BERT
2. **Abordagens**: AM tradicional vs. LLM
3. **Configurações**: Com/sem validação cruzada, diferentes tipos de aprendizado few-shot
4. **Datasets**: Performance across different domains

## Autores

- **Raphael** - Desenvolvimento principal - [@raphaelsg7](https://github.com/raphaelsg7)

- **Hianka** - Desenvolvimento principal - [@hnka15](https://github.com/hnka15)
