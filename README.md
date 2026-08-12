# Experimentos de Biometria de Digitação (IC 2024)

Este repositório contém a infraestrutura de experimentos, algoritmos e scripts de análise para pesquisa em Biometria de Digitação (Keystroke Dynamics), desenvolvidos no âmbito de Iniciação Científica (2024).

O objetivo do projeto é avaliar e comparar o desempenho de diferentes classificadores e estratégias de otimização de hiperparâmetros na identificação e autenticação de usuários baseada em padrões de digitação.

## Estrutura do Projeto

* `lib/`: Módulo principal com a arquitetura dos experimentos.
  * `lib/datasets/`: Abstrações e carregadores dos datasets (CMU e KeyRecs).
  * `lib/estimators/`: Implementação de estimadores customizados (ex.: Algoritmo Estatístico de Magalhães).
  * `lib/runners/`: Executores de experimentos para cada estratégia de tuning (Sem HPO, HPO Global e HPO por Usuário).
  * `lib/repositories/`: Repositório de persistência de resultados em formato Parquet.
  * `lib/utils.py`: Funções utilitárias, divisões de treino/teste e constantes globais.
* `datasets/`: Contém os dados brutos e arquivos de apoio para os benchmarks:
  * `datasets/cmu/`: CMU Keystroke Benchmark Dataset.
  * `datasets/keyrecs/`: KeyRecs Dataset.
* `execute_all.py`: Script para execução sequencial de todos os experimentos configurados.
* `generate_exp_metric_averages.py`: Script para cálculo das médias das métricas dos experimentos.
* `generate_graphs.py`: Gerador de gráficos comparativos de desempenho.
* `generate_hp_table.py`: Gerador de tabelas com os conjuntos de hiperparâmetros selecionados.
* `generate_st_params_hist.py`: Gerador de histogramas para os parâmetros do modelo estatístico.

## Algoritmos Avaliados

1. **Random Forest (RF)**
   * `rf.py`: Execução padrão sem otimização de hiperparâmetros.
   * `rf_global_hp_tuning.py`: Otimização global de hiperparâmetros.
   * `rf_user_hp_tuning.py`: Otimização de hiperparâmetros individual por usuário.
   * `rf_keyrecs.py`: Execução específica para o dataset KeyRecs.

2. **Support Vector Machine (SVM)**
   * `svm.py`: Execução padrão sem otimização.
   * `svm_global_hp_tuning.py`: Otimização global de hiperparâmetros.
   * `svm_user_hp_tuning.py`: Otimização de hiperparâmetros individual por usuário.

3. **Algoritmo Estatístico Melhorado / Magalhães (ST)**
   * `st.py`: Execução do modelo estatístico customizado sem otimização.
   * `st_global_hp_tuning.py`: Otimização global dos parâmetros estatísticos.
   * `st_user_hp_tuning.py`: Otimização dos parâmetros por usuário.

## Datasets

O projeto suporta dois datasets principais de dinâmica de digitação:

* **CMU Keystroke Benchmark**: Dados de tempo de acionamento de teclas para digitação de senha padronizada.
* **KeyRecs Dataset**: Benchmark com sequências estendidas de repetição e sessões.

## Requisitos e Instalação

### Pré-requisitos

* Python 3.10+

### Dependências

As principais dependências do projeto incluem:

* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `fastparquet` / `cramjam` (para armazenamento e leitura dos resultados dos experimentos)

Para instalar todas as dependências:

```bash
pip install -r requirements.txt
```

## Como Executar

### Executando Todos os Experimentos

Para rodar a suíte completa de experimentos:

```bash
python execute_all.py
```

### Executando um Experimento Específico

Você pode rodar qualquer um dos scripts individuais. Por exemplo:

```bash
# Executa Random Forest com tuning por usuário
python rf_user_hp_tuning.py

# Executa SVM com tuning global
python svm_global_hp_tuning.py

# Executa o modelo estatístico baseline
python st.py
```

### Gerando Análises e Gráficos

Após a execução dos experimentos, utilize os scripts de geração de métricas e gráficos:

```bash
# Calcular médias de métricas
python generate_exp_metric_averages.py

# Gerar gráficos comparativos
python generate_graphs.py

# Gerar tabela de hiperparâmetros
python generate_hp_table.py

# Gerar histogramas dos parâmetros do modelo estatístico
python generate_st_params_hist.py
```
