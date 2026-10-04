# OpenNPUEffects — M0: Infraestrutura de Telemetria e Instrumentação

Este documento especifica a infraestrutura de telemetria do **M0 (Instrumentação e Baseline)** no OpenNPUEffects, os esquemas de dados, contadores autoritativos, métricas de tempo, classificação de procedência e operação de diagnóstico.

---

## 1. Princípios e Arquitetura

O marco M0 estabelece **observabilidade sem alteração funcional**:
- **Zero alteração de pixels, DSP ou modelos**: os mesmos algoritmos de inferência, guided filter, blendLinear, filtros de áudio e redes neurais são executados sem modificação na matemática.
- **Produtores desacoplados e não bloqueantes**: o loop de vídeo, os callbacks assíncronos da NPU e o loop de áudio não realizam escrita em disco, serialização JSON por frame ou ordenação de quantis.
- **Separação estrita entre contadores do owner e telemetria best-effort**: contadores de ciclo de vida (`*_total`) pertencem ao owner do recurso e nunca descartam dados. A telemetria diagnóstica possui fila de eventos limitada (`event_capacity`) e buffer circular de amostras (`samples_per_metric`). Se o buffer encher ou houver contenção no coletor, descarta-se diagnóstico, **nunca** frames ou blocos PCM.
- **Transparência de incerteza**: nenhum timestamp de chegada é promovido a tempo de exposição do sensor ou latência de apresentação de hardware.

```
+------------------------------------------------------------------------+
|                          Produtores (Loops)                           |
|  npu_webcam_daemon.py: capture, async queue, heavy assist, composition |
|  npu_audio_daemon.py:  physical read, NPU denoise, DSP, write/flush     |
+------------------------------------------------------------------------+
              │                                      │
              ▼ (contadores atômicos)                ▼ (try_lock / bounded)
   [Contadores Autoritativos]             [BoundedTelemetry Collector]
              │                                      │
              │                                      ▼ (Queue limitada)
              │                           [DiagnosticsTraceWriter]
              │                                      │ (0700 / 0600 JSONL)
              ▼                                      ▼
+────────────────────────────────────────────────────────────────────────+
|           TelemetryRuntime: Relatórios Periódicos e Traces             |
+────────────────────────────────────────────────────────────────────────+
```

---

## 2. Configuração (`config.json`)

A telemetria é configurada de forma idêntica e independente nas seções `"video"` e `"audio"` de `~/.config/npu-effects/config.json`:

```json
"m0_telemetry": {
  "enabled": false,
  "trace_enabled": false,
  "report_interval_seconds": 5.0,
  "event_capacity": 2048,
  "samples_per_metric": 512,
  "trace_every_n": 30,
  "trace_max_bytes": 10485760,
  "trace_backup_count": 2
}
```

### Limites e Validação de Schema
| Campo | Tipo | Limites Permitidos | Padrão | Descrição |
|---|---|---|---|---|
| `enabled` | boolean | `true`, `false` | `false` | Ativa a coleta de métricas e relatórios em memória/log |
| `trace_enabled` | boolean | `true`, `false` | `false` | Ativa a gravação de traces em JSONL rotativo em disco |
| `report_interval_seconds` | float | 1.0 a 60.0 s | 5.0 | Intervalo de emissão do sumário com quantis e contadores |
| `event_capacity` | int | 16 a 16384 | 2048 | Capacidade da fila de eventos em memória |
| `samples_per_metric` | int | 8 a 4096 | 512 | Tamanho da janela circular para quantis (p50, p95, p99) |
| `trace_every_n` | int | 1 a 10000 | 30 | Fator de amostragem de eventos periódicos |
| `trace_max_bytes` | int | 1 MiB a 64 MiB | 10 MiB | Tamanho máximo antes de rotacionar o arquivo JSONL |
| `trace_backup_count` | int | 1 a 3 | 2 | Número de backups rotacionados mantidos |

*Nota sobre Hot-Reload*: alterações na seção `m0_telemetry` do arquivo de configuração são detectadas sem reiniciar o pipeline de vídeo ou áudio. Configurações inválidas ou numéricos fora dos limites sãos tratados com fallback seguro sem lançar exceções no chamador.

---

## 3. Diretório de Diagnóstico e Segurança

Quando `trace_enabled=true`, os arquivos de log JSONL são gravados em:
```text
~/.local/state/npu-effects/diagnostics/
```
Se `XDG_STATE_HOME` estiver definido, o diretório usado será `$XDG_STATE_HOME/npu-effects/diagnostics/`.

### Regras de Segurança
1. **Permissões do Diretório**: `0700` (`rwx------`), pertencente exclusivamente ao UID do processo.
2. **Permissões dos Arquivos**: `0600` (`rw-------`), abertos com `O_CREAT | O_WRONLY | O_APPEND` e permissões nativas `0o600` desde a criação.
3. **Rejeição de Links Simbólicos e Arquivos Especiais**: Rejeita symlinks e exige que o alvo seja arquivo regular (`st_mode & S_IFREG`).
4. **Nomenclatura**: `trace_<stream>_<session_id>.jsonl`.

---

## 4. Contadores Autoritativos (Owner Counters)

Estes contadores são acumuladores estritamente monótonos mantidos pelos proprietários dos recursos, preservados mesmo em caso de reconexão de hardware ou descarte na fila diagnóstica:

### Vídeo (`stream: "video"`)
- `capture_attempt_accumulated`: tentativas de leitura física no V4L2 acumuladas por todo o ciclo de vida do daemon.
- `capture_success_accumulated`: quadros físicos obtidos com sucesso.
- `capture_failure_accumulated`: leituras físicas que retornaram falso/vazio.
- `selected_source_gap`: lacuna de quadros observada na seleção atual.
- `primary_async_submitted_total`: inferências primárias submetidas à `AsyncInferQueue`.
- `primary_async_reused_total`: quadros onde a NPU estava ocupada e foi necessário reutilizar a máscara anterior.
- `primary_async_timeouts_total`: estouros do timeout `primary_async_same_frame_timeout_ms` aguardando a NPU.
- `primary_sync_warmup_total`: inferências síncronas de primeiro quadro (aquecimento do pipeline).
- `primary_sync_config_total`: inferências síncronas executadas por configuração explícita (`async_primary_enabled: false`).
- `heavy_assist_submitted_total`: solicitações enviadas à thread `HeavyAssistWorker`.
- `heavy_assist_observed_total`: consumos de resultados da thread de assistência pesada.
- `processed_frame_total`: quadros que completaram toda a composição.
- `send_call_total`: chamadas à virtual camera `vcam.send()`.
- `send_success_total`: envios bem sucedidos para o dispositivo v4l2loopback.
- `source_frame_new_send_total`: quadros enviados contendo captura física recém-chegada.
- `vcam_not_ready_total`: falhas ou bloqueios no dispositivo de saída virtual.
- `callback_dropped_out_of_order_total`: callbacks assíncronos descartados por chegada fora de ordem (política latest-wins).

### Áudio (`stream: "audio"`)
- `full_input_blocks_total`: blocos completos de entrada observados.
- `input_samples_observed_total`: total de amostras de áudio lidas do microfone.
- `short_read_total`: leituras físicas que retornaram menos que o tamanho de bloco esperado.
- `submitted_model_samples_total`: amostras submetidas à inferência na NPU.
- `normal_output_samples_accepted_total`: amostras de áudio normal processadas e aceitas na saída.
- `silence_output_samples_accepted_total`: amostras de silêncio geradas em standby/desconexão.
- `write_attempt_total`: tentativas de escrita no sink PipeWire.
- `write_or_flush_failure_total`: falhas de escrita ou flush no descritor de reprodução.

---

## 5. Métricas e Histogramas de Duração

Todas as durações são medidas em milissegundos via relógio monotônico nativo (`time.monotonic_ns()`). Para cada métrica, o relatório calcula:
`retained_n`, `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms` e `max_ms`.

### Métricas de Vídeo
- `video.capture_wait_ms`: tempo gasto esperando o V4L2 fornecer o quadro na thread de captura.
- `video.select_wait_ms`: espera na sincronização entre a thread de captura e o loop principal.
- `video.capture_to_select_ms`: intervalo desde a chegada do quadro na captura até sua seleção pelo loop.
- `video.framing_ms`: cálculo de auto-framing e bounding box.
- `video.filters_ms`: aplicação de filtros de imagem (suavização de pele, nitidez, iluminação baixa).
- `primary.preprocess_ms`: redimensionamento para 256x256 e conversão de formato para a NPU.
- `primary.same_frame_wait_ms`: espera do evento do callback assíncrono para o próprio quadro.
- `primary.submit_to_callback_ms`: tempo decorrido desde a submissão assíncrona até a execução do callback (inclui tempo de fila e NPU).
- `primary.async_callback_postprocess_ms`: execução do pós-processamento de máscara na thread do callback.
- `primary.sync_infer_ms`: duração da chamada `infer()` no caminho síncrono.
- `primary.sync_postprocess_ms`: pós-processamento de máscara no caminho síncrono.
- `assist.chair_ms`: duração da inferência de cadeira/objetos na thread de assistência.
- `assist.glasses_ms`: duração da inferência de óculos na thread de assistência.
- `assist.modnet_ms`: duração da inferência MODNet na thread de assistência.
- `composition.temporal_filter_ms`: filtragem temporal com `prev_mask`.
- `composition.guided_filter_ms`: tempo do Fast Guided Filter de alta precisão.
- `composition.blend_ms`: fusão SIMD direta uint8 via `cv2.blendLinear`.
- `video.send_call_ms`: chamada bloqueante `vcam.send()`.
- `video.pacing_wait_ms`: espera no pacing de framerate da câmera virtual (`vcam.sleep_until_next_frame()`).

### Métricas de Áudio
- `audio.read_call_ms`: tempo bloqueado no descritor de leitura física do microfone.
- `audio.feed_prepare_ms`: preparação do tensor de entrada e estados recorrentes para a NPU.
- `audio.infer_call_ms`: execução da inferência `infer()` do modelo `noise-suppression-poconetlike-0001.xml` na NPU.
- `audio.output_and_state_fetch_ms`: recuperação de tensores de saída e atualização dos estados.
- `audio.dsp_ms`: cadeia de estúdio DSP (Low-cut, De-reverb, Studio EQ, De-esser, Gate, Compressor).
- `audio.serialization_ms`: conversão de float32 para bytes.
- `audio.write_call_ms`: tempo exclusivo da chamada `stdin.write()`.
- `audio.flush_call_ms`: tempo exclusivo da chamada `stdin.flush()`.
- `audio.arrival_to_flush_return_ms`: latência total desde a chegada do bloco de áudio até o retorno do flush.

---

## 6. Procedência de Observação (`ObservationOrigin`)

Cada entidade (quadro ou bloco PCM) carrega um identificador imutável:
- `generation_id`: identificador único da sessão de captura física. Toda reconexão gera um novo `generation_id`.
- `frame_id` / `block_id`: contador sequencial iniciado em 1 por geração.
- `arrival_mono_ns`: timestamp de relógio monotônico na chegada física.
- `config_version`: versão da configuração em vigor na chegada.
- `timestamp_quality`: classificado como `"ARRIVAL_ESTIMATE"` (estimativa na borda de software; não exposição de hardware).

### Classificação de Relação (`relation_to`)
- `SAME_FRAME`: procedência idêntica à do quadro de composição atual.
- `OLDER_FRAME`: procedência de um quadro anterior da mesma geração (reuso/cache).
- `OTHER_GENERATION`: procedente de uma conexão de sensor anterior (reconexão física em andamento).
- `OTHER_CONFIG`: procedente de uma versão de configuração anterior.
- `FUTURE_SOURCE`: inconsistência de ordenação detectada.
- `INVALID_TIME`: relógio monotônico regrediu ou é inconsistente.
- `UNKNOWN`: procedência ausente ou indeterminada.

### Classificação de Camadas Auxiliares (`origin_kind`)
- `NEW_CONTRIBUTION`: a inferência auxiliar completou com sucesso para o contexto do frame atual sem histórico prévio.
- `MIXED_HISTORY`: combinação de dados novos com histórico acumulado (ex: filtro temporal EMA 70/30 de cadeira).
- `UNCHANGED_CACHE`: o helper devolveu a máscara do cache sem disparar nova inferência ou por decaimento suave.
- `ZERO_RESULT`: inferência executada, mas nenhum objeto/máscara foi detectado.
- `NONE_RESULT`: camada desativada ou não inicializada.
- `ERROR_FALLBACK`: falha na inferência auxiliar com fallback para o cache.

---

## 7. Eventos de Trace e Anomalias (JSONL)

Os eventos estruturados são serializados com as seguintes assinaturas:

### Seleção de Inferência Primária (`primary_selected`)
```json
{
  "schema_version": 1,
  "stream": "video",
  "session_id": "9d8f...",
  "event": "primary_selected",
  "observed_mono_ns": 182746192837,
  "current_generation_id": "video-gen-a1b2c3d4",
  "current_frame_id": 142,
  "source_generation_id": "video-gen-a1b2c3d4",
  "source_frame_id": 142,
  "source_config_version": 1,
  "source_timestamp_quality": "ARRIVAL_ESTIMATE",
  "source_relation": "SAME_FRAME",
  "source_arrival_age_ms": 14.8,
  "source_frame_gap": 0,
  "reason": "ASYNC_SELECTED"
}
```

### Aplicação Real de Máscara Auxiliar (`assist_applied`)
Emitido exclusivamente nos pontos de fusão real (MODNet na região de cabeça pre-GF, óculos pre-GF, óculos post-GF, cadeira e handheld post-GF):
```json
{
  "schema_version": 1,
  "stream": "video",
  "session_id": "9d8f...",
  "event": "assist_applied",
  "observed_mono_ns": 182746194120,
  "layer": "chair",
  "stage": "fullres_post_gf",
  "current_frame_id": 142,
  "current_generation_id": "video-gen-a1b2c3d4",
  "current_config_version": 1,
  "source_frame_id": 139,
  "source_generation_id": "video-gen-a1b2c3d4",
  "latest_contribution_arrival_age_ms": 98.4,
  "origin_kind": "MIXED_HISTORY",
  "history_present": true,
  "history_age_unknown": true
}
```

### Saída de Quadro (`video_output_frame`)
```json
{
  "schema_version": 1,
  "stream": "video",
  "session_id": "9d8f...",
  "event": "video_output_frame",
  "observed_mono_ns": 182746210000,
  "generation_id": "video-gen-a1b2c3d4",
  "output_frame_id": 142,
  "output_type": "LIVE",
  "dropped": false,
  "drop_reason": null,
  "vcam_delay_ms": null,
  "selected_source_gap": 0,
  "source_frame_id": 142,
  "source_generation_id": "video-gen-a1b2c3d4",
  "arrival_to_send_ms": 32.1,
  "presentation_timestamp_ns": null
}
```

### Bloco de Áudio (`audio_block`)
```json
{
  "schema_version": 1,
  "stream": "audio",
  "session_id": "4e7a...",
  "event": "audio_block",
  "observed_mono_ns": 182746220000,
  "generation_id": "audio-gen-f9e8d7c6",
  "block_id": 85,
  "first_observed_sample_index": 174080,
  "sample_count": 2048,
  "rate": 16000,
  "signal_duration_ms": 128.0,
  "timestamp_quality": "ARRIVAL_ESTIMATE",
  "sink_presentation_latency_ms": null
}
```

### Eventos de Anomalia Dedicados
- `callback_dropped_out_of_order`: callback da NPU descartado por ser mais antigo que o resultado publicado anteriormente.
- `audio_short_read`: leitura do microfone incompleta com `requested_samples`, `actual_samples`, `actual_bytes`.
- `audio_sink_error`: erro de escrita ou flush no descritor de reprodução com `error`.
