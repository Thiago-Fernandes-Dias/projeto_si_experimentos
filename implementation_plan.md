# Plano de Implementação da ML API (registro histórico)

> **Status:** documento histórico. A proposta original deste arquivo previa **Random Forest com negativos sintéticos**, cadastro na própria ML API (`/api/register`) e extração de features a partir de `rawTimings`. Essas decisões foram substituídas durante a implementação. As fontes atuais são [`projeto_si_ml_api/requisitos.md`](../projeto_si_ml_api/requisitos.md) e [`projeto_si_ml_api/README.md`](../projeto_si_ml_api/README.md).

## Decisões finais

- **Stack:** Python 3.12, Django 5.2 LTS, Django REST Framework, scikit-learn, PostgreSQL 17 compartilhado com o backend .NET.
- **Classificador de produção:** `M2005_IDB` (`AdaptiveM2005Alg`, `variant="idb"`, `window_size=50`, `threshold=0.7`), treinado de forma **one-class** apenas com amostras genuínas. `M2005_DB` e o `M2005Alg` legado (`M2005_Pisani2024_Sliding_Dynamic`) continuam disponíveis por configuração.
- **Cadastro:** responsabilidade exclusiva do backend .NET (`POST /api/register`). A ML API não cadastra usuários nem lê/verifica senhas; `POST /api/register` na ML API responde `404`.
- **Endpoints da ML API:** `GET /api/health`, `POST /api/enrollment`, `POST /api/authenticate` e o endpoint interno `POST /api/models/rollback`.
- **Contrato de entrada:** somente vetores posicionais `3n-2` (`H`, `DD`, `UD`); não há suporte a mapas `features`, `samples` ou `rawTimings`.
- **Persistência:** a ML API lê `usuarios` e `amostras` e escreve apenas `modelos` + artefatos `.joblib`. Os artefatos ficam em `models_data/<user_id>/v<version>_<sha256>.joblib`, com metadados em `.joblib.meta.json` (hash, versão do scikit-learn, gatilho e estatísticas de treino).
- **Limiar:** calibração por usuário com `min`/`min_non_outlier`, limitada ao teto `0.7` e recalculada a cada adaptação/retreinamento (`dynamic_threshold=True`).
- **Retreinamento:** APScheduler em processo dedicado (`ml-scheduler`), com intervalo de `60` minutos e lock `select_for_update(skip_locked=True)`. Para `AdaptiveM2005Alg` da mesma variante, a adaptação é incremental (novas amostras + Welford + janela deslizante); nos demais casos o modelo é reconstruído. Amostras de `Cadastro` são sempre genuínas; amostras de `Login` só entram se forem `Genuine` e superarem a banda de confiança. `IMPOSTOR` e `UNCERTAIN` são descartadas.
- **Isolamento após troca de senha:** versões com `criado_em < data_mudanca_senha` são ignoradas e `activate_model_version` rejeita rollback para versões pré-troca. Em falha do treinamento, o backend chama `POST /api/models/rollback`.
- **Qualidade:** Ruff para lint e `pytest` para os testes automatizados em `tests/`.

## Alternativas descartadas

- **Random Forest supervisionado com negativos sintéticos:** descartado em favor do classificador estatístico one-class `M2005_IDB`, alinhado aos 52 experimentos comparativos do repositório.
- **Cadastro e verificação de senha na ML API:** removidos; o backend .NET concentra credenciais, JWT, bloqueio e persistência de amostras.
- **`rawTimings`/extração de features na ML API:** removidos; o frontend calcula as features e o backend envia o vetor posicional mascarado.
- **HPO e runners de otimização no fluxo de produção:** fora do escopo da ML API; os experimentos usam hiperparâmetros fixos definidos nos scripts.

## Referências

- [`projeto_si_ml_api/requisitos.md`](../projeto_si_ml_api/requisitos.md)
- [`projeto_si_ml_api/README.md`](../projeto_si_ml_api/README.md)
- [`projeto_si_experimentos/README.md`](README.md)
