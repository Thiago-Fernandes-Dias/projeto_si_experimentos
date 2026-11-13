# Experimentos de Biometria de Digitação (IC 2024 & Adaptação Biométrica)

Este repositório contém a infraestrutura de experimentos, algoritmos e scripts de análise para pesquisa em Biometria de Digitação (*Keystroke Dynamics*), integrando tanto modelos estáticos com otimização de hiperparâmetros (HPO) quanto abordagens adaptativas baseadas na literatura recente para tratamento de envelhecimento de template (*concept drift*).

O objetivo do projeto é avaliar e comparar o desempenho de diferentes classificadores, estratégias de otimização de hiperparâmetros e mecanismos de adaptação contínua na identificação e autenticação de usuários baseada em padrões de digitação.

---

## Estrutura do Projeto

* `lib/`: Módulo principal com a arquitetura dos experimentos.
  * `lib/datasets/`: Abstrações e carregadores dos datasets (CMU e KeyRecs).
  * `lib/estimators/`: Implementação de estimadores customizados compatíveis com a API Scikit-Learn:
    * `improved_statistical_alg.py`: Algoritmo Estatístico Melhorado de Magalhães et al. (2005) (M2005 estático).
    * `m2005_adaptive_alg.py`: M2005 adaptativo com *Double Parallel* (DB) e *Improved Double Parallel* (IDB com estatísticas online de Welford, Pisani et al., 2015).
    * `hocquet_alg.py`: Classificador estatístico de Hocquet et al. (2007) / Mhenni et al. (2016) com decaimento exponencial de distância e suporte a limiar variável.
    * `self_detector_alg.py`: Algoritmo de Seleção Positiva (*Self-Detector*, Pisani et al., 2015) com *Rank Transform*, similaridade de cosseno e estratégias *Sliding Window*, *Usage Control R* e *Usage Control 2*.
    * `scaled_manhattan_alg.py`: Classificador *Scaled Manhattan* com Desvio Absoluto Médio (MAD), campeão do benchmark de Killourhy & Maxion (2009).
    * `knn_instance_alg.py`: Detector baseado em instâncias k-NN / 1-NN com padronização e pontuação contínua de anomalia.
  * `lib/runners/`: Executores de experimentos para cada estratégia:
    * `experiment_without_hpo_runner.py`: Execução padrão sem tuning.
    * `experiment_with_global_hpo_runner.py`: Otimização global de hiperparâmetros.
    * `experiment_with_user_hpo_runner.py`: Otimização de hiperparâmetros por usuário.
    * `adaptive_stream_runner.py`: Avaliação sequencial em fluxo de dados (*data stream*, Pisani et al., 2015) com adaptação online a amostras aceitas como genuínas.
    * `adaptive_session_runner.py`: Avaliação em sessões discretas com atualização tardia (*delayed template update*) e limiares fixo/individual/variável (Mhenni et al., 2016).
  * `lib/repositories/`: Repositório de persistência de predições e parâmetros em formato Parquet (`results/`).
  * `lib/hp_grids.py`: Grades de hiperparâmetros para os modelos.
  * `lib/utils.py`: Funções utilitárias, divisões de treino/teste e constantes globais.
* `datasets/`: Contém os dados brutos e links para os benchmarks:
  * `datasets/cmu/`: CMU Keystroke Benchmark Dataset (`DSL-StrongPasswordData.csv`).
  * `datasets/keyrecs/`: KeyRecs Dataset (`fixed-text.csv`).
* `download_cmu_dataset.py`: Utilitário para download automático do dataset CMU caso não esteja presente localmente.

---

## Algoritmos Avaliados

### 1. Modelos Clássicos & HPO
1. **Random Forest (RF)**
   * `rf.py`: Execução padrão sem otimização.
   * `rf_global_hp_tuning.py`: Otimização global de hiperparâmetros.
   * `rf_user_hp_tuning.py`: Otimização de hiperparâmetros individual por usuário.
   * `rf_keyrecs.py`: Execução específica para o dataset KeyRecs.

2. **Support Vector Machine (SVM)**
   * `svm.py`: Execução padrão One-Class SVM sem otimização.
   * `svm_global_hp_tuning.py`: Otimização global de hiperparâmetros.
   * `svm_user_hp_tuning.py`: Otimização de hiperparâmetros individual por usuário.

3. **Algoritmo Estatístico Melhorado / Magalhães (ST)**
   * `st.py`: Execução do modelo M2005 baseline.
   * `st_global_hp_tuning.py`: Otimização global dos parâmetros estatísticos.
   * `st_user_hp_tuning.py`: Otimização dos parâmetros por usuário.

### 2. Modelos Estatísticos e Adaptativos da Literatura Recente
4. **Hocquet & Atualização de Limiar Adaptativo (Mhenni et al., 2016)**
   * `hocquet.py`: Execução estática baseline do classificador de Hocquet et al. (2007).
   * `adaptive_mhenni.py`: Avaliação em sessões comparando:
     * Janelas: *None* (estático), *Sliding Window* e *Growing Window*.
     * Políticas de Limiar: *Fixed Threshold*, *Individual Threshold* e *Variable Threshold* ($T_{i+1} = \max(T_{min}, T_i - \exp(-\mu/\sigma))$).

5. **Self-Detector e Adaptação em Fluxo de Dados (Pisani et al., 2015)**
   * `self_detector.py`: Execução estática baseline do modelo imunológico de seleção positiva.
   * `adaptive_pisani.py`: Avaliação sequencial em *data stream* com 30% de ataques de impostores comparando:
     * *Self-Detector (No adapt)*: Linha de base estática.
     * *Self-Detector (Sliding)*: Janela deslizante substituindo o detector mais antigo.
     * *Self-Detector (Usage Control R)*: Remoção do detector menos utilizado com ociosidade esgotada.
     * *Self-Detector (Usage Control 2)*: Eliminação proativa de todos os detectores não utilizados recentemente.
     * *M2005 Double Parallel (DB)*: Fusão de modelo crescente com deslizante.
     * *M2005 Improved Double Parallel (IDB)*: Modelo crescente com estatísticas incrementais de Welford ($O(1)$ de memória).

6. **Modelos One-Class Sugeridos (Baseados em Instâncias & MAD)**
   * `scaled_manhattan.py`: Baseline estático do Scaled Manhattan com Desvio Absoluto Médio (MAD).
   * `adaptive_suggested.py`: Execução em *data stream* adaptativo para:
     * *Scaled Manhattan*: Estático vs. *Sliding Window*.
     * *k-NN Instance Score*: Estático vs. *Sliding Window*.

---

## Datasets

O projeto suporta dois datasets principais de dinâmica de digitação:

* **CMU Keystroke Benchmark**: 51 usuários digitando a senha `.tie5Roanl` 400 vezes em 8 sessões (total de 20.400 amostras).
* **KeyRecs Dataset**: Benchmark com sequências estendidas de repetição e sessões.

---

## Requisitos e Instalação

### Pré-requisitos
* Python 3.10+

### Instalação das Dependências

```bash
python -m venv .venv
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest
```

### Download dos Dados do CMU

Para baixar o arquivo `datasets/cmu/dataset.csv`:

```bash
python download_cmu_dataset.py
```

---

## Como Executar

### 1. Executando os Experimentos Adaptativos dos Artigos

```bash
# Experimento de Mhenni et al. (2016): Sessões com janelas deslizantes/crescentes e limiar adaptativo
python adaptive_mhenni.py

# Experimento de Pisani et al. (2015): Fluxo contínuo (Data Stream) com Self-Detector e M2005IDB
python adaptive_pisani.py

# Experimentos com os modelos One-Class sugeridos (Scaled Manhattan e k-NN)
python adaptive_suggested.py
```

### 2. Executando Baselines Individuais Sem HPO

```bash
python st.py               # Modelo Magalhães / M2005
python hocquet.py          # Modelo Hocquet
python self_detector.py    # Modelo Self-Detector
python scaled_manhattan.py # Modelo Scaled Manhattan
python svm.py              # One-Class SVM
python rf.py               # Random Forest
```

### 3. Executando Otimização de Hiperparâmetros (HPO)

```bash
python st_global_hp_tuning.py    # Tuning global do modelo estatístico
python st_user_hp_tuning.py      # Tuning individual do modelo estatístico
python rf_global_hp_tuning.py    # Tuning global de Random Forest
python rf_user_hp_tuning.py      # Tuning individual de Random Forest
python svm_global_hp_tuning.py   # Tuning global de SVM
python svm_user_hp_tuning.py     # Tuning individual de SVM
```

### 4. Executando Todos os Experimentos Tradicionais

```bash
python execute_all.py
```

### 5. Executando Testes Automatizados

```bash
pytest -v
```

### 6. Gerando Análises, Médias e Gráficos

Após a execução dos experimentos, os resultados em formato Parquet são gravados no diretório `results/`. Para processar e consolidar as métricas:

```bash
# Calcular médias de BACC, FRR e FAR de todos os experimentos em results/
python generate_exp_metric_averages.py

# Gerar gráficos comparativos de desempenho
python generate_graphs.py

# Gerar tabela com os hiperparâmetros selecionados
python generate_hp_table.py

# Gerar histogramas dos parâmetros do modelo estatístico
python generate_st_params_hist.py
```
Os arquivos consolidados serão salvos em `results/exp_metric_averages.csv` e no diretório `charts/`.
