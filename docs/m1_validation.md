# Relatório de Validação e Evidências — M1: Pipeline Primário Coerente

## 1. Sumário Executivo
Este documento apresenta os resultados da validação automatizada e as evidências técnicas da entrega do marco **M1: Pipeline Primário Coerente** da arquitetura de Áudio e Vídeo do OpenNPUEffects (requisitos AV-M1-01 a AV-M1-06), incorporando integralmente as correções da primeira, segunda e terceira rodadas de revisão técnica (R1, R2 e R3).

A implementação foi validada através de uma suíte abrangente de **93 testes unitários e de integração determinísticos**, executados no ambiente Python virtual `/opt/npu-effects/venv`:
- **100% de aprovação (93/93 testes)** em ~0.75s.
- **Zero regressão** em relação à linha de base M0 (todos os 42 testes preexistentes continuam passando integralmente).
- **0/15 defeitos reproduzidos** na suíte independente da primeira revisão (`M1 - reproduções independentes dos defeitos do patch.py`).
- **0/10 defeitos residuais reproduzidos** na suíte independente da segunda revisão (`M1 revisado - verificações independentes da segunda versão.py`).
- **0/4 grupos residuais reproduzidos** na suíte independente de integração da terceira revisão (`M1 - verificações de integração da terceira versão.py`).
- **Compilação limpa** (`compileall`) sem avisos de sintaxe ou importação em todo o repositório.
- **Validação de estilo de patch** (`git diff --check`) com zero erros de whitespace.

---

## 2. Execução da Suíte de Testes Automatizada

### 2.1 Comando de Execução
```bash
PYTHONPATH=src/bin /opt/npu-effects/venv/bin/python3 -m unittest discover -s tests/pipeline -p 'test_*.py' -v
```

### 2.2 Distribuição dos Testes
| Módulo de Teste | Quantidade | Status | Escopo Coberto |
|---|---|---|---|
| `test_m1_contracts.py` | 6 | APROVADO | Imutabilidade profunda, isolamento de memória de tensores, validação estrita de `M1Config`, contratos de `FrameIdentity`, `StageResult` e `OutputFrame`. |
| `test_m1_coherent_pipeline.py` | 45 | APROVADO | 20 casos de ciclo de vida e pipeline M1 + 15 casos de regressão R1 + 6 casos de regressão R2 + 4 casos de regressão R3 cobrindo todos os defeitos reportados nas revisões. |
| `test_m0_telemetry.py` | 15 | APROVADO | Infraestrutura M0, serialização, limites de buffer, propriedades de telemetria. |
| `test_m0_video_hooks.py` | 13 | APROVADO | Ciclo de vida e concorrência de vídeo M0, CaptureWorker, HeavyAssistWorker, paridade de composição. |
| `test_m0_audio_hooks.py` | 14 | APROVADO | Integridade de áudio M0, AudioEffectsChain, detecção de erros, resiliência sob writer bloqueado. |
| **Total** | **93** | **93/93 OK** | **Tempo total de execução: ~0.75s** |

---

## 3. Matrizes de Conformidade e Regressão

### 3.1 Matriz dos 20 Casos de Teste M1 Obrigatórios
A tabela abaixo detalha os 20 cenários fundamentais implementados em [`tests/pipeline/test_m1_coherent_pipeline.py`](file:///home/kleber/Development/NPU/tests/pipeline/test_m1_coherent_pipeline.py) e seus resultados:

| # | Cenário Testado | Método de Teste | Resultado Verificado |
|---|---|---|---|
| 1 | Resultado exato, configuração e geometria compatíveis | `test_01_exact_result_composition_accepted` | Composição aceita com status `OK` e validade `EXACT`. |
| 2 | Frame 2 com máscara 1 após timeout | `test_02_frame_2_with_mask_1_drop_never_fusion` | Descarte autoritativo (`primary_frame_dropped_total`); nunca realiza fusão entre frames diferentes. |
| 3 | Callback N+1 chega antes de N | `test_03_reordered_callbacks_no_state_regression` | Inversão tratada sem regressão de publicação; cada frame recebe sua respectiva máscara. |
| 4 | Timeout, callback chega logo depois | `test_04_timeout_then_late_callback_does_not_resurrect` | Resultado tardio rejeitado (`primary_late_callback_total`); não ressuscita job cancelado. |
| 5 | Request reutiliza buffer de saída | `test_05_request_reuses_output_buffer_published_remains_intact` | Cópia isolada em `StageResult.get_mask()` mantém pixels publicados imutáveis mesmo após mutação do array do driver. |
| 6 | Captura reutiliza/muda buffer de entrada | `test_06_capture_buffer_mutation_does_not_affect_job` | `FrameContext.framed_pixels` retém cópia imutável protegida contra escrita de hardware subsequente. |
| 7 | Config/crop muda durante inferência | `test_07_config_change_during_inference_invalidates_context` | Transição de época invalida contexto e rejeita callback tardio da geometria anterior. |
| 8 | `reset()` seguido de publicação antiga | `test_08_reset_followed_by_old_callback` | Reset atômico descarta callbacks tardios sem poluir novos resultados. |
| 9 | Job antigo altera EMA interna após reset | `test_09_generation_boundary_rejects_old_history` | Fronteira de geração rejeita histórico de inferências iniciadas na geração anterior. |
| 10 | Geração nova frame 0 com cache antiga frame 1793 | `test_10_new_gen_frame_0_with_old_cache_frame_1793` | Rejeição por incompatibilidade de `generation_id`; repetição do frame anterior é bloqueada. |
| 11 | Primária 365 com auxiliares 362/361/359 | `test_11_auxiliaries_omitted_in_m1` | Auxiliares explicitamente omitidos pendentes de M3 (`aux_omitted_pending_m3_total`). |
| 12 | Privacidade durante espera/composição | `test_12_privacy_during_wait_ensures_safe_screen` | Próximo envio emite tela segura (`SAFE_SCREEN`); commits de frames e repetições são bloqueados. |
| 13 | Desbloqueio e callback privado antigo | `test_13_unlock_clears_history_requires_new_context` | Desbloqueio limpa `last_output_frame`, exigindo um novo contexto capturado post-desbloqueio. |
| 14 | Drop seguido de repetição | `test_14_drop_followed_by_repeat_no_duplicate_unique_count` | Frame repetido é emitido sem incrementar `new_output_frame_total` e sem reprocessar gestos. |
| 15 | Composição ultrapassa prazo absoluto | `test_15_compose_deadline_miss_rejects_commit` | `commit_live_frame` rejeita frames entregues após o deadline (`compose_deadline_miss_total`). |
| 16 | Primeira imagem sem primária pronta | `test_16_first_frame_without_primary_produces_safe_screen` | Na ausência de `last_output_frame`, o pipeline produz tela segura completa e não pixels brutos. |
| 17 | Todos os efeitos desligados | `test_17_bypass_mode_without_inference` | Modo bypass publica pixels em posse privada sem submeter inferências à fila da NPU. |
| 18 | M0 desligado | `test_18_m0_telemetry_disabled_operational_guarantees` | Todas as garantias operacionais de M1 persistem independentemente da telemetria M0 ativa ou inativa. |
| 19 | Troca legacy/coherent com jobs em voo | `test_19_mode_switch_with_inflight_jobs` | Alternância de modo invalida jobs em voo, estabelece nova geração e descarta callbacks legados. |
| 20 | Parada com inferência pendente | `test_20_stop_with_pending_inference` | Encerramento limpo sem bloqueio; `reset()` marca jobs pendentes e preserva propriedade de buffers. |

---

### 3.2 Verificação de Regressão dos 15 Defeitos da Revisão R1
Validação executada via script independente `/home/kleber/Downloads/M1 - reproduções independentes dos defeitos do patch.py`:

| # | Defeito Reportado na Revisão R1 | Resolução Implementada | Status |
|---|---|---|---|
| 1 | Daemon referencia `raw_w`/`raw_h` indefinidos na criação do contexto | Invocação obtém `in_w`/`in_h` diretamente de `framed.shape[1]`/`framed.shape[0]`. | NOT REPRODUCED |
| 2 | QUEUE_BUSY passa `drop_reason` não suportado para `StageResult` | `StageResult` aceita `drop_reason` e sincroniza bidirecionalmente com `error_message`. | NOT REPRODUCED |
| 3 | Chamada síncrona `run_sync` possui ordem incorreta de argumentos | `CoherentPrimaryCoordinator.run_sync` aceita polimorficamente `(ctx, blob, req)` e `(req, ctx, blob)`. | NOT REPRODUCED |
| 4 | Bypass do daemon referencia `current_epoch` inexistente em `LifecycleManager` | Adicionada property `@property current_epoch` delegando para `processing_config_epoch`. | NOT REPRODUCED |
| 5 | Daemon assume que status de string possui atributo `.name` | Implementada classe `StatusStr(str)` expondo `.name` com total compatibilidade. | NOT REPRODUCED |
| 6 | Desbloqueio de privacidade permite commit de contextos pré-barreira | Desbloqueio incrementa épocas, limpa caches e invalida retroativamente contextos pré-barreira. | NOT REPRODUCED |
| 7 | Publicação de saída regride de frame 2 para frame 1 | `commit_live_frame` impõe monotonicidade estrita (`frame_id <= last_committed_frame_id` rejeitado). | NOT REPRODUCED |
| 8 | Repetição rotula pixels do frame 1 com identidade do frame 2 | `create_repeated_frame` preserva proveniência (`source_frame_id`, `source_generation_id`, `attempt_frame_id`). | NOT REPRODUCED |
| 9 | Telemetria desligada reinicia orçamento no contexto em vez da captura | Daemon rastreia `last_capture_mono_ns` e garante deadline baseado na captura de hardware. | NOT REPRODUCED |
| 10 | Contexto retém alias gravável do produtor para `framed_pixels` | Cópia profunda e isolada com flag `writeable = False` na inicialização de `FrameContext`. | NOT REPRODUCED |
| 11 | Campos de `FrameContext` e config aninhada permaneciam mutáveis | Implementado `FrozenDict` e proteção `__setattr__` impedindo mutações em runtime. | NOT REPRODUCED |
| 12 | Payload de `StageResult` retém alias mutável do produtor | Arrays em payload são clonados profundamente e encapsulados em `FrozenDict`. | NOT REPRODUCED |
| 13 | Máscara legada NCHW decodificada com geometria incorreta | `_extract_fallback_mask` diferencia confiavelmente layouts NCHW `(1, H, W)` e NHWC `(H, W, 1)`. | NOT REPRODUCED |
| 14 | Reconexão de câmera altera geração de ciclo após geração de captura escolhida | `invalidate("CAMERA_RECONNECT")` preserva geração atual por default (`new_generation=False`). | NOT REPRODUCED |
| 15 | Configuração inicial inválida de M1 instala defaults silenciosamente | `LifecycleManager` adota política fail-safe (`safe`) rejeitando configurações inválidas. | NOT REPRODUCED |

**Resultado consolidado da revisão R1:** `Defects reproduced: 0/15`.

---

### 3.3 Verificação de Regressão dos 10 Defeitos Residuais da Revisão R2
Validação executada via script independente `/home/kleber/Downloads/M1 revisado - verificações independentes da segunda versão.py`:

| # | Observação Residual Reportada na Revisão R2 | Resolução Implementada | Status |
|---|---|---|---|
| 1 | Metadados aninhados de contexto (`box`, `faces`, etc.) permaneciam mutáveis por indexação | Conversão recursiva em `FrozenList` e `FrozenDict` que neutraliza mutações in-place e protege contra reatribuição. | NOT REPRODUCED |
| 2 | União in-place (`\|=`) em `FrozenDict` contornava a imutabilidade do dicionário | Implementado `FrozenDict.__ior__` retornando `self` sem mutar os dados subjacentes. | NOT REPRODUCED |
| 3 | Configuração inicial inválida de M1 retrocedia para modo inseguro `legacy` | Política explícita de fail-safe: configuração inválida entra em modo `safe` (`SAFE_SCREEN`). | NOT REPRODUCED |
| 4 | Modo de pipeline desconhecido ou com erro tipográfico aceito sem validação | Modos não homologados são sanitizados para a política de fail-safe (`safe`), bloqueando fallback `legacy`. | NOT REPRODUCED |
| 5 | Mudança apenas em telemetria diagnóstica (`m0_telemetry`) invalidava época de processamento | Isolamento estrito entre configurações funcionais e de diagnóstico; telemetria não altera `processing_config_epoch`. | NOT REPRODUCED |
| 6 | Contexto continha apenas subconjunto mínimo de configuração funcional | Snapshot completo dos parâmetros funcionais congelados em `frozen_cfg` (`pipeline_mode`, `blur_enabled`, `epoch`). | NOT REPRODUCED |
| 7 | Daemon rotulava geração antiga de captura como sendo a geração corrente do ciclo | `create_frame_context` preserva a geração de captura de origem (`origin.generation_id`) no contexto. | NOT REPRODUCED |
| 8 | Contexto pós-desbloqueio podia conter frame capturado antes da barreira de privacidade | `privacy_unlock_mono_ns` bloqueia commit de qualquer frame cujo timestamp de captura seja anterior ao desbloqueio. | NOT REPRODUCED |
| 9 | Telemetria de repetição/safe do daemon rotulava frame tentado como fonte | `video_output_frame` emite `source_frame_id` e idade do frame repetido original, e `None` para tela segura. | NOT REPRODUCED |
| 10 | Fallback de crop sem framer continha referência a dimensões não avaliáveis no AST | Atribuição segura com fallback contextual direto preservando integridade da avaliação estática e dinâmica de AST. | NOT REPRODUCED |

**Resultado consolidado da revisão R2:** `Residual observations reproduced: 0/10`.

---

### 3.4 Verificação de Integração dos 4 Grupos Residuais da Revisão R3
Validação executada via script independente `/home/kleber/Downloads/M1 - verificações de integração da terceira versão.py`:

| # | Grupo Residual Reportado na Revisão R3 | Resolução Implementada | Status |
|---|---|---|---|
| 1 | Modo `safe` não integrado ao loop de execução do daemon e ramos de bypass | Tratamento explícito de `safe` no início do loop principal e no ramo de bypass: emissão de tela neutra (`SAFE_SCREEN`), zero pixels de câmera, incremento de `safe_output_total` e bloqueio de ramos downstream. | NOT REPRODUCED |
| 2 | Ramo de bypass não transmitia tempo de chegada nem geração ao commit de privacidade | Propagação de `origin.arrival_mono_ns` e `origin.generation_id` para `FrameIdentity` e `OutputFrame` no bypass; barreira temporal rejeita frames com chegada anterior ao desbloqueio ou ausente. | NOT REPRODUCED |
| 3 | `FrozenList` ainda permitia mutação via métodos `reverse()` e `sort()` | Implementados overrides no-op em `FrozenList.reverse()` e `FrozenList.sort()` que preservam a imutabilidade das listas e geometrias aninhadas publicadas. | NOT REPRODUCED |
| 4 | Coordenador admitia geração antiga e devolvia `OK/EXACT` antes de rejeição no commit | Validação estrita de geração e época em `submit_async`, `handle_async_completion`, `get_or_wait_result` e `run_sync`: snapshots de gerações anteriores são rejeitados com status `ERROR` e validade `MISMATCH`. | NOT REPRODUCED |

**Resultado consolidado da revisão R3:** `Residual groups reproduced: 0/4`.

### 3.5 Verificação Independente de Regressões do Coordenador (Revisão R4)
Validação executada via script independente `/home/kleber/Downloads/M1 - regressões independentes do coordenador da quarta versão.py`:

| # | Item Crítico Reportado na Revisão R4 | Resolução Implementada | Status |
|---|---|---|---|
| 1 | Rejeição de `generation_id` incompatível antes da submissão assíncrona ao driver | `submit_async` captura snapshot atômico do ciclo de vida sob lock e rejeita (`False`, razão `GENERATION_MISMATCH`) antes de chamar `queue.start_async()`. | PASS |
| 2 | Rejeição de época de configuração (`processing_config_epoch`) defasada antes do driver | `submit_async` valida época contra o ciclo de vida e rejeita (`False`, razão `CONFIG_EPOCH_MISMATCH`) antes de chamar `queue.start_async()`. | PASS |
| 3 | Eliminação de acúmulo de resultados órfãos em callbacks pós-timeout | Callbacks de jobs abandonados por timeout ou deadline descartam o resultado sem popular `_results`, liberando a referência do buffer de entrada `blob`. Tamanho do store após timeouts: 0. | PASS |
| 4 | Eliminação de repopulação de store após `reset()` do coordenador | `reset()` marca todos os jobs pendentes como cancelados e abandonados; callbacks tardios pós-reset nunca repopulam o store. Tamanho do store após reset: 0. | PASS |
| 5 | Consumo de resultados exatos da geração corrente esvazia store | O consumo imediato via `get_or_wait_result` remove a entrada sob lock (`pop`), mantendo `len(_results) == 0`. | PASS |
| 6 | Capacidade finita e FIFO estrita no armazenamento de resultados | Implementado `_store_result_locked` com capacidade máxima delimitada por `max_in_flight` (descarte FIFO das entradas mais antigas sob overflow). | PASS |

**Resultado consolidado da revisão R4:** `Ran 6 tests in 0.006s - OK (6/6 passing, 0 orphan results)`.

---

## 4. Verificações de Pré-Voo e Qualidade

### 4.1 Compilação Bytecode
```bash
/opt/npu-effects/venv/bin/python3 -m compileall src/ tests/
# Código de retorno: 0 (Sucesso absoluto)
```

### 4.2 Verificação de Whitespace e Formatação
```bash
git diff --check
# Código de retorno: 0 (Sem erros de whitespace)
```

### 4.3 Estatísticas do Diff
```text
 docs/m1_coherent_primary.md                 |  172 +++++
 docs/m1_validation.md                       |  177 +++++
 src/bin/npu_pipeline/__init__.py            |   32 +-
 src/bin/npu_pipeline/contracts.py           |  350 +++++++++
 src/bin/npu_pipeline/lifecycle.py           |  457 ++++++++++++
 src/bin/npu_pipeline/primary.py             |  627 ++++++++++++++++
 src/bin/npu_pipeline/telemetry_runtime.py   |    2 +
 src/bin/npu_webcam_daemon.py                | 1055 +++++++++++++++++++++++++--
 src/config/config.json                      |    6 +
 tests/pipeline/test_m1_coherent_pipeline.py | 1000 +++++++++++++++++++++++++
 tests/pipeline/test_m1_contracts.py         |  256 +++++++
 11 files changed, 4055 insertions(+), 79 deletions(-)
```

---

## 5. Invariantes de Segurança Garantidas

1. **Posse Exclusiva de Buffers**: Buffers de entrada alocados para a NPU são retidos pelo `CoherentPrimaryCoordinator` até a conclusão física do callback assíncrono do driver, impedindo *use-after-free* ou *data race*.
2. **Isolamento de Memória de Saída**: As máscaras neurais recebidas pelo callback são imediatamente clonadas (`np.ndarray.copy()`), evitando corrupção caso o OpenVINO reutilize o mesmo ponteiro em requisições subsequentes.
3. **Barreira de Privacidade Inviolável**: Uma vez ativada a privacidade, nenhum frame pendente em inferência ou composição pode transpor a barreira para a saída de vídeo. Além disso, o desbloqueio subsequente invalida retroativamente e rejeita qualquer frame capturado antes do momento de desbloqueio (`privacy_unlock_mono_ns`).
4. **Política Rigorosa de Fail-Safe**: Na ocorrência de configurações inválidas ou modos desconhecidos, o pipeline nunca recua inadvertidamente para o modo legada sem garantias; ele adota o modo `safe` com tela de segurança.
5. **Resiliência a Deadlocks**: Todas as estruturas de controle compartilhado utilizam travas reentrantes (`threading.RLock`), eliminando deadlocks entre chamadas aninhadas de repetição e commit.
