# Guia Operacional: Coleta de Telemetria e Traces JSONL (M0)

Este guia contém os comandos e a explicação prática para realizar uma rodada de teste com coleta completa de dados da telemetria M0 (resumos no journal, status dos serviços e traces granulares em formato JSONL).

---

## Visão Geral do Fluxo

```text
[1. Ativar Trace] ──> [2. Iniciar Sessão ($OUT / $START)] ──> [3. Usar Câmera & Mic (3–5 min)] ──> [4. Salvar Logs & Copiar Traces] ──> [5. Desativar Trace]
```

---

## Passo 1: Ativar os Traces Granulares no `config.json`

Para que os eventos detalhados por quadro e por bloco de áudio sejam salvos em arquivos `.jsonl`, altere `"trace_enabled": true` no arquivo `~/.config/npu-effects/config.json`.

Você pode fazer isso com um comando rápido no terminal:

```bash
python3 -c '
import json, os
p = os.path.expanduser("~/.config/npu-effects/config.json")
with open(p, "r", encoding="utf-8") as f:
    cfg = json.load(f)
for stream in ("video", "audio"):
    m0 = cfg.setdefault(stream, {}).setdefault("m0_telemetry", {})
    m0["enabled"] = True
    m0["trace_enabled"] = True
    m0["trace_every_n"] = 30  # use 1 para gravar 100% dos quadros
with open(p, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print("✓ Trace ativado com sucesso em ~/.config/npu-effects/config.json")
'
```

> [!NOTE]
> Os daemons possuem **hot-reload**: eles detectam a alteração no arquivo em menos de 1 segundo sem precisar reiniciar os serviços.

---

## Passo 2: Definir a Pasta `$OUT` e Marcar o Início da Rodada

Abra o terminal que você utilizará para controlar a coleta. Execute o bloco abaixo:

```bash
# 1. Garante permissões restritas (somente o seu usuário lê e grava)
umask 077

# 2. Define o caminho da pasta com carimbo de data/hora atual
OUT="$HOME/npu-m0-logs/$(date +%Y-%m-%d_%H-%M-%S)"

# 3. Cria a pasta no disco
mkdir -p "$OUT"

# 4. Limpa traces antigos de sessões anteriores no diretório de runtime
rm -f "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/npu-effects/diagnostics/"*.jsonl* 2>/dev/null || true

# 5. Registra o timestamp exato de início da sessão para o journalctl
START="$(date --iso-8601=seconds)"

printf '\n>>> Coleta iniciada!\n'
printf '>>> Diretório de saída ($OUT): %s\n' "$OUT"
printf '>>> Horário de início  ($START): %s\n\n' "$START"
```

### O que esses comandos fazem?
- `umask 077`: assegura que a pasta criada e os arquivos salvos não sejam acessíveis por outros usuários da máquina.
- `OUT=...`: armazena em uma variável de ambiente do terminal o caminho da pasta exclusiva da rodada (ex.: `/home/kleber/npu-m0-logs/2026-10-04_13-45-00`).
- `START=...`: guarda o instante de início para que o comando posterior do `journalctl` traga exatamente os logs ocorridos a partir desse momento, descartando eventos do passado.

> [!IMPORTANT]
> **Mantenha esse terminal aberto** enquanto realiza o teste. As variáveis `$OUT` e `$START` residem na memória desse terminal.

---

## Passo 3: Executar o Teste Prático (3 a 5 minutos)

1. Abra o aplicativo de sua preferência que utilize os dispositivos virtuais:
   - **Vídeo**: *Intel NPU Enhanced Webcam*
   - **Áudio**: *Intel_NPU_ClearVoice_Microphone* (ou padrão do sistema)
2. Use por **3 a 5 minutos** com seus efeitos habituais:
   - Movimente a cabeça, incline-se e mova as mãos em frente à câmera.
   - Fale normalmente em frases contínuas e faça pausas curtas.
   - Tente provocar o momento em que o rastro visual ou chiado ocorria anteriormente.

---

## Passo 4: Salvar os Logs e Copiar os Traces JSONL

No **mesmo terminal** do Passo 2, execute o bloco a seguir:

```bash
# 1. Salva o journal completo dos dois serviços desde o instante $START
journalctl --user \
  -u npu-webcam.service -u npu-audio.service \
  --since "$START" --no-pager -o short-iso \
  > "$OUT/servicos.log"

# 2. Salva o snapshot de status e métricas recentes dos daemons
systemctl --user status \
  npu-webcam.service npu-audio.service --no-pager \
  > "$OUT/status.txt" 2>&1

# 3. Copia os arquivos de trace JSONL gerados na sessão
TRACE_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/npu-effects/diagnostics"
if [ -d "$TRACE_DIR" ]; then
    cp -a "$TRACE_DIR"/*.jsonl* "$OUT/" 2>/dev/null || true
fi

printf '\n✓ Coleta concluída com sucesso!\n'
printf 'Arquivos salvos em: %s\n\n' "$OUT"
ls -lh "$OUT"
```

---

## Passo 5: Desativar os Traces JSONL (Recomendado)

Como os traces granulares gravam eventos contínuos em disco, desative-os após o teste para evitar consumo de I/O e espaço:

```bash
python3 -c '
import json, os
p = os.path.expanduser("~/.config/npu-effects/config.json")
with open(p, "r", encoding="utf-8") as f:
    cfg = json.load(f)
for stream in ("video", "audio"):
    if "m0_telemetry" in cfg.get(stream, {}):
        cfg[stream]["m0_telemetry"]["trace_enabled"] = False
with open(p, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print("✓ Trace desativado em ~/.config/npu-effects/config.json (resumos M0-SUMMARY continuam ativos)")
'
```

---

## Estrutura Final da Pasta Coletada

Ao final da execução, sua pasta `$OUT` conterá os seguintes arquivos:

| Arquivo | Descrição |
| :--- | :--- |
| `servicos.log` | Journal completo do systemd (avisos, erros e linhas periódicas `M0-SUMMARY`). |
| `status.txt` | Saída do `systemctl --user status` com PIDs, memória, threads e último resumo. |
| `video_trace.jsonl` | Linhas JSON estruturadas com timestamps, IDs de captura, submissões e atrasos de quadro. |
| `audio_trace.jsonl` | Linhas JSON estruturadas com blocos de áudio, tempos de inferência e DSP. |
