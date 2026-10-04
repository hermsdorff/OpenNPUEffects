# Relatório de Linha de Base — M0: Infraestrutura e Baseline

## Sumário Executivo
Este relatório documenta a entrega consolidada do marco **M0: Instrumentação e Baseline** do pipeline de Áudio e Vídeo do OpenNPUEffects (requisitos AV-M0-01 a AV-M0-06), incorporando as correções integrais dos bloqueios de revisão V2 (V2-01, V2-02 e V2-03) e dos pontos residuais da revisão V3 (R1, R2 e simplificação AST), sem implementar M1–M6.

A entrega implementa observabilidade de ciclo de vida, precedência temporal (`ObservationOrigin`), telemetria estritamente limitada em memória e diagnósticos rotativos em disco, sem alterar a matemática dos pixels, DSP, pesos de redes neurais ou as heurísticas do pipeline operacional.

O status dos itens diagnosticados na revisão é verificável por suíte de testes determinística e autocontida:
- **V2-01 (Conexão do HeavyAssistWorker)**: O coletor M0 é injetado na construção do worker e mantido em todos os hot reloads (inclusive desligamento/religamento e alterações de capacidade). O tempo de espera em fila do job (`assist.job_wait_ms`) é registrado antes da inferência auxiliar.
- **V2-02 (Identidade de Captura em assist_applied)**: Os cinco pontos de emissão `assist_applied` utilizam a identidade real de captura (`current_frame_id` e `current_generation_id` derivados diretamente de `cur_cap_id` e `cur_gen_id`), preservando o sequencial de saída no campo dedicado `output_frame_id`. Origem não observada (placeholder, gaps, standby/privacy) resulta em `null`. O cálculo de idade é estritamente condicionado à concordância de gerações.
- **V2-03 (Consumidor Único e Resiliência sob I/O Bloqueado)**: O runtime mantém estritamente no máximo um único consumidor e um único writer por stream, sem acumular threads ou criar múltiplos writers concorrentes para o mesmo arquivo sob reloads com I/O bloqueado. O thread caller de mídia é não bloqueante em qualquer reload ou parada.
- **R1 (Estacionamento em OFF e Reancoragem de Janela)**: Em OFF por hot reload, o consumidor permanece estacionado sem invocar o logger de sumários periódicos. Na transição de volta para ON, a janela de relatório (`_last_report_mono_ns`) e a linha de base dos contadores de proprietário (`_last_owner_snapshots`) são reancoradas ao timestamp exato da reativação, eliminando deltas inflados e durações espúrias.
- **R2 (Retenção e Contabilidade Exata de Eventos em Fila)**: `BoundedTelemetry` e `NullTelemetry` expõem `@property events_queued` leve e thread-safe. A drenagem multipasso em `_ConsumerWorker` preserva coletores com eventos remanescentes através de múltiplos lotes (ex.: 300 eventos em fila de 512 drenam 256 no primeiro lote, permanecem rastreados e completam os 44 no lote seguinte). Descartes provocados pelo limite máximo de 4 coletores pendentes e truncamentos em parada abrupta são devidamente contabilizados em `_dropped_events_total` e agregados em `events_queue_dropped`.
- **Timer Operacional Preexistente**: O timer PERF do callback assíncrono em `npu_webcam_daemon.py` opera com medição real via `perf_counter()` mesmo quando M0 está desligado, assegurando acurácia do `cb_ema` operacional sem overhead diagnóstico desnecessário.

---

## Estrutura do Relatório e Evidências

```text
Revisão e alterações locais:
- Commit base do checkout: HEAD em origin/main (3de726b516ec068b1d7e9ae45d431de09994b10a)
- Módulos de telemetria e infraestrutura:
    * src/bin/npu_pipeline/__init__.py
    * src/bin/npu_pipeline/telemetry.py (BoundedTelemetry, NullTelemetry, ObservationOrigin, provenance_fields)
    * src/bin/npu_pipeline/telemetry_runtime.py (TelemetryRuntime, DiagnosticsTraceWriter, validate_m0_config)
- Suíte de testes automatizados (42 testes, 100% aprovados sem hardware físico):
    * tests/pipeline/m0_fakes.py (fixtures determinísticas de FakeCamera, FakeVirtualCam, FakeInferRequest, FakeClock, FakeAudioPipe, FakeAudioProc)
    * tests/pipeline/test_m0_telemetry.py (15 testes de infraestrutura, limits, serialização e property events_queued)
    * tests/pipeline/test_m0_video_hooks.py (13 testes de ciclo de vida, concorrência e integração real: CaptureWorker, HeavyAssistWorker com telemetria e hot reload, identidade de captura em assist_applied, apply_neural_chair_retention, apply_neural_glasses_retention, fast_guided_filter e caminho legado)
    * tests/pipeline/test_m0_audio_hooks.py (14 testes de integridade, AudioEffectsChain, escrita/flush, detecção de erros, resiliência do runtime sob writer bloqueado com barreiras, estacionamento em off e contabilidade exata de descartes e drenagem multilote)
- Documentação técnica:
    * docs/m0_instrumentation.md (especificação de métricas, esquemas JSONL e anomalias)
    * docs/m0_baseline.md (este documento)
- Daemons instrumentados:
    * src/bin/npu_webcam_daemon.py (contadores monótonos entre reconexões, conexão do HeavyAssistWorker com hot reload e medição de espera de job, 5 pontos de assist_applied com identidade de captura e output_frame_id, restauração do timer PERF de cb_ema em M0 off)
    * src/bin/npu_audio_daemon.py (separação estrita de medição de write vs flush, eventos de anomalia audio_short_read e audio_sink_error, contadores autoritativos, proteção de cleanup)
    * src/config/config.json (seções m0_telemetry em 'video' e 'audio' default off)

Ambiente do host (informações declaradas da máquina de desenvolvimento):
- SO: Linux 6.17.9-200.fc41.x86_64
- Host: Intel(R) Core(TM) Ultra 7 155H (Meteor Lake-P) com NPU integrada
- Python: 3.12.9 no ambiente virtual /opt/npu-effects/venv
- OpenVINO: 2026.4.0
- OpenCV: 5.0.0-dev
- NumPy: 2.5.3
- SciPy: 1.18.1
- pyvirtualcam: 0.15.0 (com shims automáticos nos testes em caso de ausência)
- sounddevice: 0.5.1 (com shims automáticos nos testes em caso de ausência)

Hardware e barramentos (alvos do sistema):
- CPU/NPU: Intel Meteor Lake NPU PCI ID 8086:7d1d
- Dispositivos de Áudio: Intel Meteor Lake-P HD Audio (sof-hda-dsp), PipeWire / WirePlumber
- Câmera física de entrada: /dev/video0 (ou auto-detecção)
- Câmera virtual de saída: /dev/video9 (módulo v4l2loopback)
- Estado dos serviços em produção: Inalterados em /opt (sem commits, pushes, instalação ou reinício forçado de serviços)

Modelos declarados e apontados no código:
- Vídeo Primário: selfie_multiclass.xml (Multiclass Segmentation 256x256) na NPU via AsyncInferQueue
- Vídeo Primário (legado/fallback): selfie_segmentation_static.xml (Segmentation 256x144) síncrono na NPU
- Vídeo Auxiliar 1: chair_instance_segmenter.xml (YOLACT detecção de cadeira/objetos COCO) na thread HeavyAssistWorker
- Vídeo Auxiliar 2: face_parsing_bisenet.xml (BiSeNet CelebAMask-HQ Face Parsing / Hastes de óculos) na thread HeavyAssistWorker
- Vídeo Auxiliar 3: modnet_portrait_matting.xml (MODNet matting fino de cabelo/cabeça) na thread HeavyAssistWorker
- Áudio: noise-suppression-poconetlike-0001.xml (supressão neural PoCoNet-like na NPU, chunk size 2048 amostras a 16 kHz mono)

Telemetria off/on e janelas:
- Desativado (Default, Off): NullTelemetry ativo, zero alocação de buffers, zero escrita em disco. Custo residual restrito a verificações booleanas de guard (`if m0_runtime.enabled`), empacotamento leve de dataclass ObservationOrigin na captura física e desempacotamento de tuplas nos leitores de assistência. O timer preexistente de cb_ema opera com sua medição nativa de perf_counter().
- Ativado (On): BoundedTelemetry com anel circular limitado (padrão 2048 eventos), descarte fail-open sob contenção, janelas de 5.0s (configurável de 1.0s a 60.0s).
- Amostras por métrica: 512 amostras com percentis p50, p95, p99, mean, max calculados sob snapshot seguro fora de lock de aquisição.

Métricas de latência implementadas e validadas:
- Vídeo:
    * video.capture_wait_ms
    * video.select_wait_ms
    * video.capture_to_select_ms
    * video.framing_ms
    * video.filters_ms
    * primary.preprocess_ms
    * primary.same_frame_wait_ms
    * primary.submit_to_callback_ms (tempo total desde submissão à fila assíncrona até execução do callback)
    * primary.async_callback_postprocess_ms
    * primary.sync_infer_ms (inferência síncrona na primária)
    * primary.sync_postprocess_ms (pós-processamento no caminho síncrono)
    * assist.job_wait_ms (tempo de espera do job na fila do HeavyAssistWorker antes do início do processamento)
    * assist.chair_ms
    * assist.glasses_ms
    * assist.modnet_ms
    * composition.temporal_filter_ms (filtro de persistência prev_mask)
    * composition.guided_filter_ms (Fast Guided Filter)
    * composition.blend_ms (cv2.blendLinear)
    * video.send_call_ms
    * video.pacing_wait_ms
- Áudio:
    * audio.read_call_ms
    * audio.feed_prepare_ms
    * audio.infer_call_ms (inferência neural na NPU)
    * audio.output_and_state_fetch_ms
    * audio.dsp_ms (cadeia de filtros Studio DSP)
    * audio.serialization_ms
    * audio.write_call_ms (chamada exclusiva stdin.write())
    * audio.flush_call_ms (chamada exclusiva stdin.flush())
    * audio.arrival_to_flush_return_ms (latência total de bloco)

Alcance da paridade matemática (M0 off vs M0 on):
- Vídeo: Comprovada em test_vid_10 através de comparação bit a bit (np.array_equal) entre as matrizes de pixels resultantes dos blocos algorítmicos fast_guided_filter e cv2.blendLinear executadas com M0 off e M0 on. O alcance desta paridade cobre os estágios algorítmicos instrumentados; a verificação end-to-end sob pipeline contínuo em tempo real depende de execução física.
- Áudio: Comprovada em test_rt_03 através de comparação exata valor a valor (np.array_equal) do sinal float32 PCM processado pela AudioEffectsChain completa (Low-cut, Studio EQ, De-esser, Compressor) com M0 off e M0 on.

Resumo dos testes de software executados (todos passaram em ~0.7s):
- 15 testes de infraestrutura, limites e fila limitada (test_m0_telemetry.py)
- 13 testes de ciclo de vida, concorrência e hooks de vídeo (test_m0_video_hooks.py)
- 14 testes de integridade, áudio, resiliência, estacionamento em off e drenagem multilote (test_m0_audio_hooks.py)
- Total: 42 testes automatizados determinísticos executados sem necessidade de câmera física, NPU ou servidor PipeWire.

Testes que permanecem dependentes de hardware físico:
- Validação de estabilidade térmica e vazão contínua a 30 FPS na NPU física (Intel NPU 8086:7d1d)
- Validação de sincronismo em tempo real com sensor físico de câmera V4L2 (/dev/video0)
- Validação de jitter e latência de loopback no nó de áudio real do PipeWire/WirePlumber

Gate M0: PENDENTE (software e infraestrutura de telemetria validados em 42 testes determinísticos sem hardware; aprovação final depende da validação com câmera física, NPU Intel e PipeWire em hardware real).
```

