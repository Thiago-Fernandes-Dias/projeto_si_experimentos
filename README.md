# Experimentos de Biometria de Digitação (IC 2024 & Adaptação Biométrica)

Este repositório contém a infraestrutura de experimentos, algoritmos e scripts de análise para pesquisa em Biometria de Digitação (*Keystroke Dynamics*), combinando modelos estatísticos estáticos e abordagens adaptativas baseadas na literatura recente para tratamento de envelhecimento de template (*concept drift*).

O objetivo do projeto é avaliar e comparar o desempenho de diferentes classificadores e mecanismos de adaptação contínua na identificação e autenticação de usuários baseada em padrões de digitação. Os experimentos usam hiperparâmetros fixos definidos nos próprios scripts; a infraestrutura de otimização de hiperparâmetros (HPO), os scripts de Random Forest e SVM e os runners de HPO foram removidos.

---

## Algoritmo de Produção Escolhido: `M2005_IDB`

Os resultados dos 52 experimentos consolidados localmente em `results/exp_metric_averages.csv` (gerado por `generate_exp_metric_averages.py`; a pasta `results/` não é versionada) embasaram a escolha do **`M2005_IDB`** (*Improved Double Parallel*, Pisani et al., IJCNN 2015) como classificador de produção da `projeto_si_ml_api`. O algoritmo combina:

- **Submodelo deslizante:** fila FIFO de 50 amostras genuínas recentes, com média, mediana e desvio-padrão por dígrafo.
- **Submodelo crescente incremental:** média e desvio-padrão online pelo algoritmo de Welford, em memória constante `O(1)` sobre todo o histórico genuíno.
- **Escore final:** média aritmética dos dois submodelos. Nos experimentos, o `AdaptiveM2005Alg` usa `threshold = 0.7` fixo; a ML API de produção aplica a calibração por usuário (`min`/`min_non_outlier`, com teto `0.7`).
  - O `M2005Alg` experimental segue Pisani (2024) sem cap no threshold calibrado; o teto `0.7` é uma decisão da ML API.
- **Treinamento One-Class:** apenas amostras genuínas; `IMPOSTOR` e `UNCERTAIN` não entram na adaptação.

| Algoritmo | BACC CMU | FRR CMU | FAR CMU | BACC KeyRecs | FRR KeyRecs | FAR KeyRecs | BACC médio |
|---|---|---|---|---|---|---|---|
| **`M2005_IDB`** | **0.8909** | **15.24%** | **6.58%** | **0.7980** | **9.03%** | **31.37%** | **0.8445** |
| `M2005_DB` | 0.8907 | 14.64% | 7.22% | 0.7822 | 8.68% | 34.87% | 0.8365 |
| `SelfDet_Sliding` | 0.8655 | 23.41% | 3.49% | 0.8068 | 33.25% | 5.38% | 0.8362 |
| `M2005_Pisani2024_Sliding_Dynamic` | 0.8659 | 7.84% | 18.98% | 0.7219 | 5.28% | 50.33% | 0.7939 |

O `M2005_IDB` lidera em BACC médio (`0.8445`), G-mean (`0.8400`) e equilíbrio `|FRR - FAR|` (`0.1550`), reduzindo a FAR do CMU em aproximadamente 3× em relação ao `M2005_Pisani2024_Sliding_Dynamic`.

---

## Estrutura do Projeto

* `lib/`: Módulo principal com a arquitetura dos experimentos.
  * `lib/datasets/`: Abstrações e carregadores dos datasets (CMU e KeyRecs).
  * `lib/estimators/`: Implementação de estimadores customizados compatíveis com a API Scikit-Learn:
    * `m2005_alg.py`: Algoritmo Estatístico Melhorado de Magalhães et al. (2005) e calibração de limiar específico por usuário (`min` e `min_non_outlier`, Pisani, 2024).
    * `m2005_adaptive_alg.py`: M2005 adaptativo com *Double Parallel* (DB) e *Improved Double Parallel* (IDB com estatísticas online de Welford, Pisani et al., 2015).
    * `m2005.py`: *shim* de compatibilidade que reexporta `M2005`, `M2005Alg` e `DigraphsMetrics`.
    * `hocquet_alg.py`: Classificador estatístico de Hocquet et al. (2007) / Mhenni et al. (2016) com decaimento exponencial de distância, duplo limiar e suporte a limiar variável.
    * `self_detector_alg.py`: Algoritmo de Seleção Positiva (*Self-Detector*, Pisani et al., 2015) com *Rank Transform*, similaridade de cosseno e estratégias *Sliding Window*, *Usage Control R* e *Usage Control 2*.
    * `scaled_manhattan_alg.py`: Classificador *Scaled Manhattan* com Desvio Absoluto Médio (MAD), campeão do benchmark de Killourhy & Maxion (2009).
    * `knn_instance_alg.py`: Detector baseado em instâncias k-NN / 1-NN com padronização e pontuação contínua de anomalia.
  * `lib/runners/`: Executores de experimentos para cada estratégia:
    * `experiment_without_hpo_runner.py`: Execução estática dos baselines, com hiperparâmetros fixos.
    * `adaptive_stream_runner.py`: Avaliação sequencial em fluxo de dados (*data stream*, Pisani et al., 2015) com adaptação online a amostras aceitas como genuínas.
    * `adaptive_session_runner.py`: Avaliação em sessões discretas com atualização tardia (*delayed template update*) e limiares fixo/individual/variável (Mhenni et al., 2016).
  * `lib/repositories/`: Repositório de persistência de predições e parâmetros em formato Parquet (`results/`).
  * `lib/logging_config.py`: Configuração de logs legíveis no console e JSONL por execução (`results/<exp_name>/<timestamp>/execution.log`).
  * `lib/utils.py`: Funções utilitárias, divisões de treino/teste e constantes globais.
* `tests/`: Testes unitários dos estimadores e dos runners adaptativos (`pytest`).
* `datasets/`: Contém os dados brutos e links para os benchmarks:
  * `datasets/cmu/dataset.csv`: CMU Keystroke Benchmark Dataset (baixado de `DSL-StrongPasswordData.csv`).
  * `datasets/keyrecs/dataset.csv`: KeyRecs Dataset (baixado de `fixed-text.csv`).
* `references/`: PDFs dos artigos de referência usados na implementação dos estimadores e protocolos adaptativos.
* `download_datasets.py`: Utilitário único para download automático dos datasets CMU e KeyRecs caso não estejam presentes localmente.

---

## Algoritmos Avaliados

### 1. Modelo Estatístico Melhorado / Magalhães (ST)
* `st.py`: Execução do modelo M2005 baseline com hiperparâmetros fixos.

### 2. Hocquet & Atualização de Limiar Adaptativo (Mhenni et al., 2016)
* `hocquet.py`: Execução estática baseline do classificador de Hocquet et al. (2007).
* `adaptive_mhenni.py`: Avaliação em sessões comparando:
  * Janelas: *None* (estático), *Sliding Window* e *Growing Window*.
  * Políticas de Limiar: *Fixed Threshold*, *Individual Threshold* e *Variable Threshold* ($T_{i+1} = \max(T_{min}, T_i - \exp(-\mu/\sigma))$).

### 3. Self-Detector e Adaptação em Fluxo de Dados (Pisani et al., 2015)
* `self_detector.py`: Execução estática baseline do modelo imunológico de seleção positiva.
* `adaptive_pisani.py`: Avaliação sequencial em *data stream* com 30% de ataques de impostores comparando:
  * *Self-Detector (No adapt)*: Linha de base estática.
  * *Self-Detector (Sliding)*: Janela deslizante substituindo o detector mais antigo.
  * *Self-Detector (Usage Control R)*: Remoção do detector menos utilizado com ociosidade esgotada.
  * *Self-Detector (Usage Control 2)*: Eliminação proativa de todos os detectores não utilizados recentemente.
  * *M2005 Double Parallel (DB)*: Fusão de modelo crescente com deslizante.
  * *M2005 Improved Double Parallel (IDB)*: Modelo crescente com estatísticas incrementais de Welford ($O(1)$ de memória).

### 4. Modelos One-Class Sugeridos (Baseados em Instâncias & MAD)
* `scaled_manhattan.py`: Baseline estático do Scaled Manhattan com Desvio Absoluto Médio (MAD).
* `adaptive_suggested.py`: Execução em *data stream* adaptativo para:
  * *Scaled Manhattan*: Estático (`none`), *Sliding Window* (`sliding`) e *Growing Window* (`growing`).
  * *k-NN Instance Score*: Estático vs. *Sliding Window*.

---

## Funcionamento Detalhado dos Algoritmos e Runners

Todos os algoritmos biométricos customizados operam no paradigma de **Classificação de Classe Única (*One-Class Classification*)**, no qual o cadastro (*enrollment*) e a adaptação contínua utilizam **exclusivamente amostras genuínas do próprio usuário**. Para conter o envenenamento da referência (*template poisoning*) durante a adaptação semi-supervisionada, os estimadores e runners suportam **Duplo Limiar (*Double Thresholding*)**:
* **Limiar de Verificação (`threshold` / `radius`)**: Define se a tentativa de autenticação é aceita (`GENUINE_LABEL = 1`) ou rejeitada (`IMPOSTOR_LABEL = -1`).
* **Limiar de Inclusão (`inclusion_threshold` / `inclusion_radius`)**: Critério mais estrito que filtra quais amostras aceitas na verificação possuem confiança suficiente para atualizar a galeria de referência do usuário (`should_include_sample()`).

### 1. Estimadores (`lib/estimators/`)

#### 1.1. `M2005Alg` (`lib/estimators/m2005_alg.py`)
Baseado em **Magalhães et al. (2005)** e estendido com a configuração de limiares específicos por usuário de **Pisani (SBSeg 2024)**:
* **Referência Biométrica**: Para cada atributo/dígrafo $i \in \{1, \dots, n\}$ da galeria do usuário $G_j$, armazena a média ($\mu_i$), a mediana ($md_i$) e o desvio-padrão amostral ($\sigma_i$).
* **Verificação por Atributo**: Uma amostra de consulta $d$ obtém acerto (*hit*) no atributo $i$ se satisfizer as inequações (1) e (2) de Pisani et al. (2015):
  $$\min(\mu_i, md_i) \cdot \left(0.95 - \frac{\sigma_i}{\mu_i}\right) \le d_i \le \max(\mu_i, md_i) \cdot \left(1.05 + \frac{\sigma_i}{\mu_i}\right)$$
* **Cálculo do Score**: Atributos acertados somam $1.0$ ponto na primeira posição ou quando o atributo anterior falhou, e $1.5$ pontos quando o atributo anterior também acertou (premiando estabilidade rítmica consecutiva). O score final é normalizado em $[0, 1]$:
  $$\text{Score}(d) = \frac{\text{soma}}{1 + 1.5 \cdot (n - 1)}$$
  em que o divisor é o valor máximo para a soma.
* **Adaptação e Calibração de Limiar (`Pisani, 2024`)**:
  * Suporta estratégias de janela `none` (estático), `sliding` (FIFO mantendo o tamanho inicial da galeria) e `growing` (acumulativa).
  * O método `calibrate_threshold()` avalia os scores $S_j$ de todas as amostras da galeria $G_j$ contra a referência atual $\text{ref}_j$ e define o limiar do usuário por:
    * `min`: $T_j = \min(S_j)$
    * `min_non_outlier`: $T_j = \min(S_j^n)$, onde $S_j^n = \{s \in S_j \mid s > \text{mean}(S_j) - 2.5 \cdot \text{std}(S_j)\}$.
  * Quando `dynamic_threshold=True`, o limiar $T_j$ é recalibrado automaticamente a cada atualização da galeria em `adapt()`.

#### 1.2. `AdaptiveM2005Alg` (`lib/estimators/m2005_adaptive_alg.py`)
Implementa as arquiteturas híbridas **Double Parallel (DB)** e **Improved Double Parallel (IDB)** propostas por **Pisani, Lorena & Carvalho (IJCNN 2015)**:
* Mantém dois submodelos M2005 em paralelo para cada usuário:
  1. **Submodelo Deslizante (*Sliding*)**: Gerenciado por uma fila FIFO (`deque`) de tamanho fixo `window_size`, refletindo o comportamento motor recente do usuário.
  2. **Submodelo Crescente (*Growing*)**:
     * Na variante `db` (*Double Parallel*), armazena todas as amostras aceitas em memória e recalcula média, mediana e desvio-padrão em lote.
     * Na variante `idb` (*Improved Double Parallel*), substitui a mediana pela média e atualiza média e variância amostral ($N - 1$ graus de liberdade) incrementalmente via **Algoritmo Online de Welford** (`_WelfordOnlineStats`), garantindo complexidade de memória constante $O(1)$ em fluxos infinitos.
* **Decisão**: O score final de uma amostra é a média aritmética dos scores produzidos pelos submodelos deslizante e crescente:
  $$\text{Score}_{\text{DB/IDB}}(x) = \frac{\text{Score}_{\text{sliding}}(x) + \text{Score}_{\text{growing}}(x)}{2}$$

#### 1.3. `HocquetStatisticalAlg` (`lib/estimators/hocquet_alg.py`)
Implementa o classificador estatístico de **Hocquet et al. (2007)** com o mecanismo de adaptação de **Mhenni et al. (ATSIP 2016)**:
* **Referência e Distância**: Representa o usuário pelos vetores de média $\mu$ e desvio-padrão $\sigma$ da galeria. A distância normalizada entre uma amostra $V$ e a referência é dada por:
  $$d(V, \text{ref}) = 1 - \frac{1}{n} \sum_{i=1}^n \exp\left(-\frac{|V_i - \mu_i|}{\sigma_i}\right)$$
  A amostra é aceita como genuína se $d(V, \text{ref}) \le T_i$.
* **Limiar Variável (`calculate_variable_threshold_decrement`)**: Conforme a Equação (2) de Mhenni et al. (2016), à medida que o usuário domina a digitação da senha ao longo das sessões, o limiar é reduzido progressivamente ao fim de cada sessão:
  $$T_{i+1} = \max\left(T_{\min}, \; T_i - \exp\left(-\frac{\bar{\mu}}{\sigma_{\sigma}}\right)\right)$$
  onde $\bar{\mu}$ é a média do vetor de médias $\mu$ e $\sigma_{\sigma}$ é o desvio-padrão do vetor de desvios-padrão $\sigma$.

#### 1.4. `SelfDetectorAlg` (`lib/estimators/self_detector_alg.py`)
Implementa o algoritmo imunológico de **Seleção Positiva (*Self-Detector*)** e as heurísticas de controle de uso de **Pisani et al. (IJCNN 2015)**:
* **Transformação por Postos e Similaridade de Cosseno**: Cada vetor de digitação passa por uma transformação de ranking (`rank_transform_vector`), substituindo os tempos absolutos pelos seus postos relativos ($1, \dots, n$) normalizados pela norma $L_2$. Isso torna o classificador altamente robusto a variações globais na velocidade de digitação.
* **Casamento (*Matching*)**: Cada amostra da galeria forma um detector com centro $\mathbf{c}_k$ e raio $r$. Uma consulta $\mathbf{x}$ casa com o detector se a distância de cosseno for inferior ao raio:
  $$1 - (\mathbf{c}_k \cdot \mathbf{x}) < r$$
* **Estratégias de Adaptação do Conjunto de Detectores**:
  * `none`: Conjunto estático criado no cadastro.
  * `sliding`: Substitui o detector mais antigo (FIFO) a cada nova amostra aceita.
  * `usage_control_r`: Inspeciona os detectores **do mais recente para o mais antigo**. O primeiro detector a casar tem seu `usage_count` incrementado e seu `recent_usage` restaurado para `max_recent_usage` (10), enquanto os demais têm `recent_usage` decrementado em 1. Só ocorre substituição se houver detectores com `recent_usage == 0`, removendo aquele com menor `usage_count`.
  * `usage_control_2`: Também inspeciona do mais recente para o mais antigo, mas a cada amostra aceita **remove todos os detectores ociosos (`recent_usage == 0`)** e insere o novo detector, adaptando dinamicamente o tamanho da galeria e reduzindo a Taxa de Falsa Aceitação (FAR).

#### 1.5. `ScaledManhattanAlg` (`lib/estimators/scaled_manhattan_alg.py`)
Baseado no detector de anomalias campeão do benchmark de **Killourhy & Maxion (2009)**:
* Calcula o vetor médio $\mu$ e o Desvio Absoluto Médio ($\text{MAD}_i = \frac{1}{|G|} \sum_{x \in G} |x_i - \mu_i|$) na galeria genuína:
  $$d(x, \text{ref}) = \frac{1}{n} \sum_{i=1}^n \frac{|x_i - \mu_i|}{\text{MAD}_i}$$
* Suporta as estratégias de janela `window_strategy="none"` (estático), `window_strategy="sliding"` (janela deslizante FIFO de tamanho `sliding_window_size`) e `window_strategy="growing"` (janela crescente que acumula todas as amostras aceitas e recalcula $\mu$ e $\text{MAD}$ sobre todo o histórico), além de duplo limiar (`inclusion_threshold`).

#### 1.6. `KNNInstanceAlg` (`lib/estimators/knn_instance_alg.py`)
Detector One-Class baseado em instâncias que padroniza os atributos pelo Z-score da galeria atual e calcula a distância euclidiana média aos $k$ vizinhos mais próximos, suportando janela deslizante e duplo limiar.

---

### 2. Executores de Experimentos (`lib/runners/`)

#### 2.1. `ExperimentRunner` (`lib/runners/experiment_runner.py`)
Classe base abstrata responsável por:
* Particionar o dataset por usuário em conjuntos de treino (`_X_genuine_training`, `_X_impostor_training`) e teste (`_X_genuine_test`, `_X_impostors_test`).
* Controlar se o estimador recebe apenas amostras da classe genuína (`use_impostor_samples=False`, padrão One-Class) ou também amostras de impostores (`use_impostor_samples=True`, opção disponível para estimadores supervisionados de duas classes).

#### 2.2. Runner Estático
* **`ExperimentWithoutHPORunner`** (`experiment_without_hpo_runner.py`): Treina um modelo fixo por usuário na primeira sessão e avalia todas as tentativas genuínas e de impostores nas sessões restantes sem atualizar o template.

#### 2.3. `AdaptiveStreamRunner` (`lib/runners/adaptive_stream_runner.py`)
Reproduz o protocolo de avaliação sequencial em **Fluxo de Dados (*Data Stream*)** de **Pisani et al. (2015)** e **Pisani (2024)**:
1. **Treinamento Inicial**: Ajusta o modelo exclusivamente nas amostras de cadastro (1ª sessão) do usuário genuíno e, se configurado (`threshold_method`), calibra o limiar inicial do usuário (`min` ou `min_non_outlier`).
2. **Construção do Fluxo Cronológico (`interleave_preserving_chronology` em `lib/utils.py`)**:
   * Ordena as amostras genuínas de teste por `(session, repetition)` e sorteia amostras de impostores na proporção definida por `impostor_ratio` (padrão: 30% impostores / 70% genuínos).
   * Sorteia as posições do fluxo onde as amostras genuínas e impostoras serão alocadas **preservando estritamente a ordem cronológica interna** de todas as sessões e repetições, condição essencial para medir o efeito real do *concept drift*.
3. **Avaliação e Adaptação Online**:
   * Cada amostra do fluxo é predita pelo modelo atual.
   * Se a amostra for aceita (`predicted_label == GENUINE_LABEL`) **e** aprovada pelo filtro de inclusão (`estimator.should_include_sample`), o método `estimator.adapt()` é invocado imediatamente (e o limiar dinâmico é atualizado se `dynamic_threshold=True`).

#### 2.4. `AdaptiveSessionRunner` (`lib/runners/adaptive_session_runner.py`)
Reproduz o protocolo de **Atualização Tardia em Sessões Discretas (*Delayed Template Update*)** de **Mhenni et al. (2016)**:
1. **Cadastro e Política de Limiar Inicial**: Treina o modelo com as amostras da sessão 1. Se `threshold_policy="individual"`, calibra um limiar específico para o usuário com base na dispersão da sua própria galeria inicial e configura a razão do duplo limiar (`inclusion_threshold_ratio`).
2. **Processamento por Sessão**:
   * Itera sobre as sessões de teste em ordem cronológica ($s = 2, 3, \dots$).
   * Em cada sessão, intercala as amostras genuínas daquela sessão com igual quantidade de ataques impostores via `interleave_preserving_chronology`.
   * Durante a sessão, o template e o limiar permanecem congelados. As consultas aceitas na verificação (`predicted_label == GENUINE_LABEL`) que também passam pelo **limiar de inclusão** (`should_include_sample`) são armazenadas em um buffer de sessão (de forma 100% semi-supervisionada, sem consultar o rótulo verdadeiro `expected_label`).
3. **Atualização ao Fim da Sessão**:
   * Ao encerrar a sessão $s$, todas as amostras retidas no buffer atualizam a galeria do usuário (`sliding` ou `growing`).
   * Se `threshold_policy="variable"`, o limiar de verificação e o limiar de inclusão são reduzidos para a sessão $s+1$ aplicando o decremento $\exp(-\bar{\mu}/\sigma_{\sigma})$ até o piso `minimum_threshold`.

---

## Datasets

O projeto suporta dois datasets principais de dinâmica de digitação:

* **CMU Keystroke Benchmark**: 51 usuários digitando a senha `.tie5Roanl` 400 vezes em 8 sessões (total de 20.400 amostras).
* **KeyRecs Dataset**: Benchmark com sequências estendidas de repetição e sessões.

---

## Requisitos e Instalação

### Pré-requisitos
* Python 3.11+ (recomendado 3.12; o código usa `typing.Self`)
* `pip` e, opcionalmente, um ambiente virtual (`venv` ou `uv`)

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

> O `requirements.txt` foi congelado em ambiente Windows e inclui `pywin32`; em Linux/macOS, remova ou ignore essa linha caso o `pip` não encontre uma wheel compatível.

### Download dos Dados (CMU e KeyRecs)

Os scripts de experimento baixam ambos os datasets automaticamente caso não estejam presentes localmente, mas você também pode baixá-los manualmente (`datasets/cmu/dataset.csv` e `datasets/keyrecs/dataset.csv`):

```bash
python download_datasets.py
```

---

## Como Executar

### 1. Executando os Experimentos Adaptativos dos Artigos

```bash
# Experimento de Mhenni et al. (2016): Sessões com janelas deslizantes/crescentes e limiar adaptativo
python adaptive_mhenni.py

# Experimento de Pisani et al. (2015): Fluxo contínuo (Data Stream) com Self-Detector e M2005_IDB
python adaptive_pisani.py

# Experimentos com os modelos One-Class sugeridos (Scaled Manhattan e k-NN)
python adaptive_suggested.py
```

> Os scripts adaptativos usam, por padrão, 5 execuções com sementes `0..4` (`default_seeds_range` em `lib/utils.py`). Cada execução grava predições e hiperparâmetros no histórico Parquet de `results/<exp_name>/<timestamp>/`, além do log JSONL de execução.

### 2. Executando Baselines Individuais

```bash
python st.py               # Modelo Magalhães / M2005
python hocquet.py          # Modelo Hocquet
python self_detector.py    # Modelo Self-Detector
python scaled_manhattan.py # Modelo Scaled Manhattan
```

### 3. Executando Todos os Baselines Estáticos

```bash
python execute_all.py
```

### 4. Executando Testes Automatizados

```bash
pytest -v
```

### 5. Gerando Análises, Médias e Gráficos

Após a execução dos experimentos, os resultados em formato Parquet são gravados no diretório `results/`. Para processar e consolidar as métricas:

```bash
# Calcular médias de BACC, FRR e FAR de todos os experimentos em results/
python generate_exp_metric_averages.py

# Gerar gráficos comparativos de desempenho
python generate_graphs.py
```
Os arquivos consolidados serão salvos em `results/exp_metric_averages.csv` e no diretório `charts/`.

---

## Logs Estruturados e Configuração

Todos os runners emitem logs estruturados de progresso por experimento, por *seed* e por usuário com o tempo gasto (`elapsed_seconds`) em cada etapa:

- **Console (`stdout`)**: texto legível no formato `timestamp | LEVEL | logger | mensagem | chave=valor ...`.
- **Arquivo (`results/<exp_name>/<YYYY-MM-DD_HH-MM-SS>/execution.log`)**: formato **JSON Lines (`.jsonl`)** com os eventos `experiment_started`, `seed_started`, `user_completed`, `predictions_saved`, `hyperparameters_saved`, `seed_completed` e `experiment_completed`.

O comportamento pode ser ajustado programaticamente via `configure_logging()` em `lib/logging_config.py` ou por variáveis de ambiente:

| Variável de Ambiente | Padrão | Descrição |
| :--- | :--- | :--- |
| `SI_LOG_LEVEL` | `INFO` | Nível mínimo de log (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `SI_LOG_TO_CONSOLE` | `1` | Habilita (`1`/`true`) ou desabilita (`0`/`false`) a saída legível no console. |
| `SI_LOG_TO_FILE` | `1` | Habilita (`1`/`true`) ou desabilita (`0`/`false`) a gravação JSONL em arquivo. |
| `SI_LOG_FILENAME` | `execution.log` | Nome do arquivo de log dentro de `results/<exp_name>/<timestamp>/`. |

---

## Referências

Os artigos usados na implementação estão em `references/`:

- `A study on user-specific threshold configuration for keystroke dynamics in the context of adaptive biometric systems.pdf` (Pisani, 2024) — calibração de limiar `min`/`min_non_outlier` e variantes adaptativas.
- `Adaptive approaches for keystroke dynamics.pdf` (Pisani et al., IJCNN 2015) — *Self-Detector*, *Double Parallel* (DB) e *Improved Double Parallel* (IDB).
- `Keystroke Template Update with Adapted Thresholds.pdf` (Mhenni et al., 2016) — atualização tardia de template, janelas deslizantes/crescentes e limiar variável.
