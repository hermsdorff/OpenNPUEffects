# Especificação e Operação — M1: Pipeline Primário Coerente

## 1. Visão Geral e Objetivos do Marco M1
O marco **M1: Pipeline Primário Coerente** da arquitetura de Áudio e Vídeo do OpenNPUEffects estabelece o desacoplamento temporal estrito e o fim da mistura assíncrona entre frames de captura e máscaras neurais antigas.

No pipeline legado, frames de vídeo podiam compor a captura do instante $T$ com a máscara primária de segmentação calculada sobre um frame anterior $T - k$ (ou reutilizar caches defasados indefinidamente), gerando descompassos visuais e anomalias de precedência temporal.

O M1 resolve estruturalmente esse problema através da política **`exact_or_drop`**:
- Cada frame capturado possui uma identidade única e imutável (`FrameIdentity`).
- A inferência primária só é composta no frame se e somente se pertencer exatamente ao mesmo frame (`identity.frame_id`, `identity.generation_id`, `identity.processing_config_epoch` e `identity.geometry_id`).
- Caso a inferência atrase além do prazo de decisão (`primary_decision_deadline_mono_ns`), falhe ou sofra timeout, o frame $N$ **nunca** é composto com a máscara de outro frame. O pipeline repete o último frame pronto válido (`REPEATED`) ou emite tela segura (`SAFE_SCREEN`), registrando o descarte autoritativo.

---

## 2. Componentes e Contratos de Dados (`contracts.py`)

Localizado em [`src/bin/npu_pipeline/contracts.py`](file:///home/kleber/Development/NPU/src/bin/npu_pipeline/contracts.py):

### 2.1 `FrameIdentity`
- Estrutura `@dataclass(frozen=True)` imutável.
- Campos de identificação:
  - `session_id`: identificador global da sessão do daemon.
  - `generation_id`: identificador de ciclo contínuo de captura de hardware (alterado a cada reconexão/standby).
  - `frame_id`: identificador monótono do frame de captura.
  - `processing_config_epoch`: versão da configuração de processamento (crop, resolução, modo).
  - `model_id` e `model_version`: identificadores do modelo de segmentação primário.
  - `geometry_id`: assinatura canônica da geometria de entrada, crop e saída (`in_w x in_h -> crop -> out_w x out_h`).
  - `seq`: sequencial opcional para ordenação de submissão.
- Método `is_compatible_with(other)`: valida estritamente a compatibilidade temporal e geométrica sem permitir fuso de frames distintos.

### 2.2 `StageResult`
- Encapsula o resultado de uma etapa de inferência ou pós-processamento.
- Status (`StageResultStatus`): `OK`, `DROPPED`, `TIMEOUT`, `ERROR`, `SKIPPED`, encapsulados via `StatusStr(str)` que preserva atributo `.name` para compatibilidade com enums e serializadores.
- Validade (`StageResultValidity`): `EXACT`, `INTERPOLATED`, `EXTRAPOLATED`, `LATE`, `MISMATCH`, `INVALID`.
- Campo `drop_reason`: aceita motivo de descarte explícito (`QUEUE_BUSY`, `TIMEOUT`, etc.) com sincronização bidirecional transparente com `error_message`.
- Proteção de memória: payload encapsulado em `FrozenDict` com isolamento profundo de arrays NumPy (`.copy()` com `writeable = False`), e o método `get_mask(name)` retorna **sempre uma cópia isolada (`.copy()`)**, impedindo que threads subsequentes ou o reuso de buffers pelo driver OpenVINO mutem a máscara publicada.

### 2.3 `FrameContext`
- Contexto `@dataclass(frozen=True)` gerado no momento da captura e enquadramento.
- Orçamento temporal monótono e finito:
  $$\text{frame\_deadline\_mono\_ns} = \text{arrival\_mono\_ns} + (\text{frame\_deadline\_ms} \times 10^6)$$
  $$\text{primary\_decision\_deadline\_mono\_ns} = \text{frame\_deadline\_mono\_ns} - ((\text{compose\_reserve\_ms} + \text{safety\_margin\_ms}) \times 10^6)$$
- Mantém cópia profunda imutável dos pixels capturados (`framed_pixels.copy()`, com flag `writeable = False`), isolando o pipeline contra sobrescrita pelo worker de captura da câmera.
- Dicionários aninhados (`frozen_cfg`, `quality_flags`) encapsulados em `FrozenDict` e atributos protegidos via `__setattr__`, impedindo mutação in-place por consumidores ou helpers.

### 2.4 `OutputFrame`
- Representa o frame pronto para envio para a câmera virtual (`/dev/video9`).
- O array de imagem é protegido contra escrita acidental (`image.flags.writeable = False`).
- Rastreamento explícito de proveniência temporal: `source_frame_id`, `source_generation_id` e `attempt_frame_id`, garantindo que frames repetidos declarem claramente a qual captura física pertencem.
- Campos `output_type`: `NEW_COMPOSED`, `REPEATED`, `SAFE_SCREEN`, `BYPASS`.
- `presentation_timestamp_ns`: inicializado explicitamente como `None`, reservado para o emissor desacoplado do marco M2.

### 2.5 `M1Config`
- Configuração de prazos e restrições com validação estrita no `__post_init__`:
  - `frame_deadline_ms` (10.0 a 1000.0 ms, default: 250.0 ms).
  - `compose_reserve_ms` (5.0 a 500.0 ms, default: 90.0 ms).
  - `safety_margin_ms` (1.0 a 100.0 ms, default: 10.0 ms).
  - `max_repeat_age_ms` (20.0 a 2000.0 ms, default: 500.0 ms).
  - `max_in_flight` (1 a 4, default: 2).
  - `exact_or_drop` (booleano, default: True).
  - Garante a invariante: $\text{compose\_reserve\_ms} + \text{safety\_margin\_ms} < \text{frame\_deadline\_ms}$.
- Política de Fail-Safe: caso os parâmetros de `m1` no arquivo de configuração sejam inválidos ou corrompidos, ou o modo solicitado seja desconhecido/inválido, o `LifecycleManager` emite log de aviso e entra estritamente em `pipeline_mode = "safe"` (tela segura), impedindo que erros de configuração reativem a política legada de máscaras defasadas.

---

## 3. Gerenciamento de Ciclo de Vida e Barreiras (`lifecycle.py`)

Localizado em [`src/bin/npu_pipeline/lifecycle.py`](file:///home/kleber/Development/NPU/src/bin/npu_pipeline/lifecycle.py):

### 3.1 `LifecycleManager`
Centraliza a autoridade do pipeline para:
- **Transições de ciclo de vida (`invalidate`)**:
  - Standby, reconexão de câmera, mudança de resolução/crop, troca de modo de pipeline.
  - Incrementa atomicamente a `epoch` e limpa a referência a `last_output_frame`.
  - Preserva a `generation_id` de captura em reconexões simples (`CAMERA_RECONNECT`, default `new_generation=False`), gerando nova geração em eventos que afetam o hardware/modo (`STANDBY`, `MODE_SWITCH`).
- **Publicação Monótona Estrita (`commit_live_frame`)**:
  - Impõe ordenação temporal inviolável: rejeita qualquer tentativa de publicação onde `frame_id <= last_committed_frame_id`.
  - Verifica estritamente a compatibilidade de época de processamento (`processing_config_epoch == self.processing_config_epoch`) e deadline absoluto.
- **Autoridade única de repetição de saída (`create_repeated_frame`)**:
  - Permite repetir um `OutputFrame` apenas se a geração e a época de configuração forem idênticas às atuais.
  - Rejeita repetições se o frame tiver idade superior a `max_repeat_age_ms` (default 500 ms) ou se a privacidade estiver ativa.
  - Preserva integralmente a proveniência original (`source_frame_id`, `source_generation_id`, `attempt_frame_id`).
  - Repetições **não** incrementam contadores de frames únicos (`new_output_frame_total`) nem disparam reavaliação de gestos.
- **Barreira de Privacidade (`set_privacy`)**:
  - Quando ativada, estabelece uma barreira imediata: o próximo frame enviado ao consumidor é obrigatoriamente a tela segura (`SAFE_SCREEN`).
  - No desbloqueio, incrementa épocas (`epoch`, `processing_config_epoch`, `privacy_epoch`, `privacy_watermark`) e limpa `last_output_frame` e `last_committed_frame_id`, garantindo que contextos pré-barreira nunca sejam compostos ou publicados.
- **Consistência sob Telemetria Inativa**:
  - Mantém `last_capture_mono_ns` atualizado a cada leitura física da câmera, assegurando que o orçamento temporal seja calculado a partir da chegada do hardware e nunca do instante de alocação de contexto em software.
- **Propriedade `current_epoch`**:
  - Exposta de forma autoritativa como alias para `processing_config_epoch`.
- **11 Contadores Autoritativos (AV-M1-06)**:
  1. `primary_exact_accepted_total`: inferências primárias exatas aceitas e compostas.
  2. `primary_frame_dropped_total`: frames com inferência primária descartada (timeout, queue busy, erro).
  3. `primary_late_callback_total`: callbacks do driver recebidos após expiração do deadline de decisão ou após cancelamento.
  4. `primary_identity_rejected_total`: rejeições por divergência de identidade/geometria.
  5. `generation_rejected_total`: rejeições causadas por geração antiga/desconectada.
  6. `aux_omitted_pending_m3_total`: inferências de modelos auxiliares omitidas no M1.
  7. `compose_deadline_miss_total`: composições que excederam o prazo absoluto de entrega.
  8. `new_output_frame_total`: frames novos exclusivos enviados à câmera virtual.
  9. `output_repeat_total`: frames válidos anteriores repetidos na saída.
  10. `safe_output_total`: telas seguras enviadas.
  11. `privacy_barrier_total`: ativações da barreira de privacidade.

---

## 4. Coordenação da Inferência Primária (`primary.py`)

Localizado em [`src/bin/npu_pipeline/primary.py`](file:///home/kleber/Development/NPU/src/bin/npu_pipeline/primary.py):

### 4.1 `CoherentPrimaryCoordinator`
- **Controle de Admissão**:
  - Limita jobs em voo a `max_in_flight=2`.
  - Verifica o prazo remanescente antes de submeter ao hardware; se o deadline já expirou, a admissão é rejeitada imediatamente (`primary_frame_dropped_total`).
- **Posse dos Buffers de Entrada**:
  - Ao chamar `submit_async(queue, frame_ctx, blob)`, o dicionário interno retém a referência ao array `blob` até que o callback físico do driver termine. Isso impede garbage collection prematuro do buffer enquanto o driver C++ da NPU o processa.
- **Tratamento de Callback Assíncrono (`handle_async_completion`)**:
  - Isola os dados de saída do driver com `.copy()` antes de qualquer reutilização pelo OpenVINO.
  - Verifica se o job foi cancelado ou se pertence a uma `generation_id` ou `epoch` anterior. Se inválido, incrementa `primary_late_callback_total` e descarta o resultado sem tocar no cache de saída.
  - Executa pós-processamento utilizando a geometria e metadados congelados em `FrameContext`.
- **Extração Robusta de Máscara (`_extract_fallback_mask`)**:
  - Suporta tensores 4D `(1, 256, 256, 6)` ou `(1, 1, 256, 256)` e 3D `(256, 256, 6)` ou `(1, 256, 256)`.
  - Extrai confiavelmente o canal de pessoa (índice 1 no modelo multiclasse) sem erros de indexação.
- **Espera e Decisão com Prazo Finito (`get_or_wait_result`)**:
  - Aguarda o evento de conclusão até `primary_decision_deadline_mono_ns`.
  - Se o tempo expirar, marca o job como cancelado, incrementa `primary_frame_dropped_total` e retorna `StageResultStatus.TIMEOUT`.
  - Rejeita qualquer resultado cuja conclusão real tenha ocorrido após o deadline.
- **Modo Síncrono (`run_sync`)**:
  - Utilizado no aquecimento ou quando a fila assíncrona estiver indisponível.
  - Aplica o mesmo limite rígido de deadline e isolamento de cópias.

---

## 5. Integração no Daemon (`npu_webcam_daemon.py`)

A integração preserva integralmente o caminho legado e executa o fluxo coerente quando `pipeline_mode == "coherent"`:
- **Zero Regressão**: o modo default permanece `"legacy"`, executando exatamente o código anterior.
- **Omissão Explícita de Auxiliares no M1**: modelos auxiliares (cadeira, óculos, pose pesada) são omitidos no M1 enquanto aguardam a arquitetura de assistência de M3 (`aux_omitted_pending_m3_total += 1`).
- **Desativação do Filtro Temporal Heurístico**: no modo coerente, a mistura temporal cega de máscaras anteriores (`prev_mask`) é desligada para garantir pureza matemática `EXACT` do resultado.
- **Bypass sem Inferência**: quando os efeitos visuais são desativados, o fluxo coerente emite diretamente a imagem capturada como `BYPASS`, sem submeter inferências desnecessárias à NPU.

---

## 6. Procedimento de Ativação e Reversão

### 6.1 Ativação Experimental (após deploy autorizado)
No arquivo `~/.config/npu-effects/config.json`, na seção `"video"`:
```json
{
  "video": {
    "pipeline_mode": "coherent",
    "m1": {
      "frame_deadline_ms": 250.0,
      "compose_reserve_ms": 90.0,
      "safety_margin_ms": 10.0,
      "max_repeat_age_ms": 500.0,
      "max_in_flight": 2,
      "exact_or_drop": true
    }
  }
}
```

### 6.2 Procedimento de Reversão
Para reverter imediatamente para o pipeline legado com zero impacto operacional:
```json
{
  "video": {
    "pipeline_mode": "legacy"
  }
}
```
O daemon detecta a alteração via hot reload e migra atomicamente para a geração legada sem necessidade de reinício do serviço.
