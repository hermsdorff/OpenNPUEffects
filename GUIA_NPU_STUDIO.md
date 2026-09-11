# 🚀 Guia Completo: Intel NPU AI Studio (Webcam & Áudio)
### Samsung Galaxy Book 4 Ultra — Intel Core Ultra 9 185H (Intel AI Boost) — KDE neon 24.04

Este documento reúne todas as informações sobre a arquitetura implantada, como utilizar no dia a dia, todos os efeitos disponíveis e o catálogo completo de configurações via interface gráfica (**Cameractrls**) e terminal (**`npu-ctl`**).

---

## 📋 Sumário
1. [O que foi Feito (Visão Geral da Arquitetura)](#1-o-que-foi-feito)
2. [Como Usar no Dia a Dia](#2-como-usar-no-dia-a-dia)
3. [Catálogo de Efeitos de Vídeo (IA & Visão Computacional)](#3-catálogo-de-efeitos-de-vídeo)
4. [Catálogo de Efeitos de Áudio (NPU & Pacote de Estúdio Vocal)](#4-catálogo-de-efeitos-de-áudio)
5. [Controle pela Interface Gráfica (Cameractrls)](#5-controle-pela-interface-gráfica-cameractrls)
6. [Controle por Linha de Comando (`npu-ctl`)](#6-controle-por-linha-de-comando-npu-ctl)
7. [Seleção de Dispositivos de Entrada (Câmeras e Microfones)](#7-seleção-de-dispositivos-de-entrada)
8. [Arquivo de Configuração (`config.json`)](#8-arquivo-de-configuração)
9. [Persistência em Atualizações do Sistema e Restauração](#9-persistência-em-atualizações-do-sistema)

---

## 1. O que foi Feito

Configuramos uma suíte completa de processamento de vídeo e áudio acelerada por hardware na **NPU (Neural Processing Unit) integrada do processador Intel Core Ultra 9**, operando de forma 100% nativa e transparente no Linux:

```
[Dispositivos Físicos]
 ├─ Webcam Física (EMEET PIXY / Notebook IPU6) ──► npu-webcam.service (OpenVINO + PTZ) ──► /dev/video72 (v4l2loopback 1080p)
 └─ Microfone Físico (USB / PIXY / Notebook)    ──► npu-audio.service  (PoCoNet + DSP)   ──► PipeWire (npu_clearvoice)
                                                           ▲
                                                   Intel AI Boost NPU
                                                   (/dev/accel/accel0)
```

### Principais Componentes Configurados:
1. **Drivers Intel NPU**:
   - Firmware oficial `intel-fw-npu` (`vpu_37xx_v1.bin`) carregado pelo kernel.
   - Driver Level Zero UMD (`intel-level-zero-npu` e `intel-driver-compiler-npu`).
   - Dispositivo `/dev/accel/accel0` integrado ao OpenVINO Runtime.
2. **Câmera Virtual V4L2 (`v4l2loopback`)**:
   - Módulo de kernel configurado em `/etc/modprobe.d/v4l2loopback.conf` gerando o dispositivo `/dev/video72` com nome amigável `"Intel NPU Enhanced Webcam"`.
   - Suporte a 6 buffers de streaming (`max_buffers=6`), garantindo compatibilidade total com o Cameractrls e entrega fluida em 1080p @ 30 FPS.
3. **Serviços em Segundo Plano (`systemd --user`)**:
   - `npu-webcam.service`: Daemon de vídeo com inicialização automática, standby on-demand e controle de gimbal PTZ.
   - `npu-audio.service`: Daemon de áudio com hot-reload automático ao alterar configurações.
4. **Gerenciador de Linha de Comando (`npu-ctl`)**:
   - Utilitário instalado em `~/.local/bin/npu-ctl` para ajuste de qualquer parâmetro em tempo real.
5. **Integração Visual com Cameractrls**:
   - Aba exclusiva "Intel NPU" no Cameractrls com sliders, switches, janela redimensionada confortavelmente e barra de rolagem sempre visível.

---

## 2. Como Usar no Dia a Dia

### Nos Aplicativos de Chamada e Vídeo (Meet, Teams, Zoom, Discord, Navegadores):
1. **Vídeo**:
   - Em qualquer aplicativo, selecione como câmera:
     👉 **`Intel NPU Enhanced Webcam`** (ou `/dev/video72`).
   - Ela entregará imagem Full HD 1080p com todos os efeitos ativos.
2. **Microfone**:
   - Selecione como dispositivo de entrada:
     👉 **`Intel NPU ClearVoice Microphone`** (`npu_clearvoice`).
   - Ele já é o microfone padrão do sistema no PipeWire.

### Comportamento Inteligente On-Demand (Standby & PTZ):
* Quando **nenhuma** aplicação estiver usando a câmera, ela entra em repouso após 3 segundos: o LED desliga, o sensor físico é liberado e a **EMEET PIXY** inclina a lente para baixo (tilt `-324000`), garantindo privacidade física total.
* Assim que você abrir uma reunião ou o preview do Cameractrls, a câmera acorda em menos de **0,3 segundos**, retorna o tilt para a posição frontal e começa a transmitir de imediato.

---

## 3. Catálogo de Efeitos de Vídeo

Todos os efeitos de vídeo rodam com aceleração de hardware na NPU e CPU com vetores AVX-512/AVX2, mantendo a latência abaixo de 33ms (30 FPS contínuos).

| Efeito | O que faz | Como ajustar |
| :--- | :--- | :--- |
| **Desfoque de Fundo (AI Blur)** | Isola a pessoa usando segmentação neural na NPU e aplica desfoque bokeh no fundo. | `npu-ctl blur [on/off/0-100]` ou slider no Cameractrls |
| **Modo Retrato Óptico (Depth-Aware)** | Simula lente DSLR física com desfoque de profundidade gradativo (bokeh mais forte em áreas distantes). | `npu-ctl blur-mode [standard/portrait]` |
| **Transição Suave de Borda (Feathering)** | Suavização contínua (*Hermite Smoothstep*) eliminando recortes duros ou bordas brancas. | `npu-ctl feather [10-80]` ou slider no Cameractrls |
| **Preservação de Cadeira (Modelo IA NPU)** | Identifica a cadeira (escritório/gamer/sofá) usando rede neural dedicada na NPU (MobileNetV3 LRASPP) e retém o encosto no primeiro plano. | `npu-ctl chair [on/off/10-100]` ou no Cameractrls |
| **Fundo Virtual (Substituição de Imagem)** | Substitui o cenário real por qualquer imagem da pasta `~/.config/npu-effects/backgrounds/`. | `npu-ctl background /caminho/foto.jpg` |
| **Efeito Parallax 3D no Fundo** | Desloca o plano de fundo sutilmente de acordo com o movimento da cabeça, criando sensação tridimensional realista. | `npu-ctl parallax [on/off/10-100]` |
| **Reconhecimento de Gestos (AI)** | Identifica gestos em tempo real (**👍 Joinha**, **✌️ Vitória**, **🖐️ Mão Aberta**) para reações visuais animadas ou atalho de microfone. | `npu-ctl gesture [on/off/reaction/mute/all]` ou no Cameractrls |
| **Preservação de Objetos na Mão** | Impede que celulares, canecas, canetas ou documentos segurados sumam ou fiquem borrados pelo desfoque. | `npu-ctl objects [on/off/10-100]` ou no Cameractrls |
| **Transição Suave de Ausência (Fade)** | Dissolve suavemente (*crossfade*) o vídeo ao entrar e sair da tela de privacidade, eliminando cortes secos. | `npu-ctl privacy-fade [on/off]` ou no Cameractrls |
| **Filtros Artísticos Criativos (AI)** | Estilização inspirada no Windows Studio Effects: **Ilustrado (HQ)**, **Animado (Cartoon)** e **Aquarela**. | `npu-ctl artistic [off/illustrated/animated/watercolor]` |
| **Grading de Cor Cinematográfico (LUTs)** | Paletas de cores de cinema: **Teal & Orange**, **Vintage Aquecido**, **Noir P&B** e **Vibrant Pop**. | `npu-ctl color [none/teal_orange/vintage/noir/vibrant]` |
| **Luz de Recorte Virtual (Rim Light)** | Iluminação de três pontos de cinema/estúdio nas bordas do cabelo e ombros (Âmbar, Branco ou Azul). | `npu-ctl rim [on/off/10-100] [warm/white/cool]` |
| **Supressão de Reflexo de Tela (Glare)** | Neutraliza tons azulados projetados pelo monitor sobre o rosto e armações de óculos no escuro. | `npu-ctl glare [on/off/10-100]` |
| **Auto-Framing (Individual & Grupo)** | Enquadra suavemente uma pessoa ou ajusta a abertura para englobar todas as pessoas na sala (*Group Framing*). | `npu-ctl framing [on/off]` / `npu-ctl framing-mode [single/group]` |
| **Contato Visual (Natural & Teleprompter)** | Realinha as íris para a câmera. O modo **Teleprompter** aplica correção extra para leitura de anotações na tela. | `npu-ctl eyecontact [off/natural/teleprompter]` |
| **Suavização Facial & Denoise** | Filtro bilateral que uniformiza a pele e elimina ruídos de sensor sem borrar olhos e cabelos. | `npu-ctl smooth [on/off/10-100]` |
| **Super-Resolução & Nitidez** | Algoritmo adaptativo que realça micro-contrastes em olhos, tecidos e cabelos. | `npu-ctl sharpen [on/off/10-100]` |
| **Luz de Estúdio (Ring Light)** | Simula iluminação frontal difusa e suave direcionada ao rosto. | `npu-ctl studio [on/off/10-100]` |
| **Modo Pouca Luz Adaptativo** | Curva de ganho inteligente que clareia cenas noturnas sem lavar tons de preto. | `npu-ctl lowlight [on/off/10-100]` |
| **Modo Privacidade por Ausência** | Substitui o vídeo por tela de pausa elegante ao se afastar da câmera por mais de 3 segundos. | `npu-ctl privacy [on/off]` |
| **Standby On-Demand Automático** | Desliga o sensor físico e o LED da webcam quando não há leitores consumindo o `/dev/video72`. | `npu-ctl standby [on/off]` |
| **Estacionamento PTZ Motorizado** | Inclina a câmera EMEET PIXY para baixo (`tilt = -324000`) em standby e reabre ao acordar. | `npu-ctl ptz-park [on/off]` |

---

## 4. Catálogo de Efeitos de Áudio

O fluxo de áudio combina o modelo neural **Intel PoCoNet** executado na NPU com uma cadeia profissional de processamento digital de sinal (DSP de Estúdio Vocal):

```
Microfone Físico
  └──► Low-Cut (80Hz)
        └──► NPU PoCoNet Denoise
              └──► De-Reverb
                    └──► Smart Noise Gate (VAD)
                          └──► Vocal Presence EQ
                                └──► Compressor / AGC (-18dB)
                                      └──► De-Esser (5.2-7.5kHz)
                                            └──► Nó Virtual PipeWire (npu_clearvoice)
```

| Efeito | O que faz | Como ajustar |
| :--- | :--- | :--- |
| **Cancelamento de Ruído IA (PoCoNet)** | Rede neural na NPU que filtra latidos, aspirador, vento, tráfego e ruído elétrico, deixando apenas a voz humana pura. | `npu-ctl mic-noise [on/off]` |
| **Pacote de Estúdio Vocal (Studio Mic)** | Ativa em conjunto toda a cadeia de pós-processamento acústico de padrão broadcast. | `npu-ctl studio-mic [on/off]` |
| **Filtro Passa-Altas / Low-Cut (80Hz)** | Elimina frequências subsônicas inaudíveis: vibrações de mesa, batidas, passos no chão e ruído de digitação em teclado mecânico. | `npu-ctl mic-lowcut [on/off]` |
| **Portão de Ruído Inteligente (Smart Gate)** | Garante **silêncio digital absoluto** nas pausas de respiração ou quando você não estiver falando. Usa detecção de voz (VAD) com envelope suave para não cortar o início das frases. | `npu-ctl mic-gate [-30 a -60 dB]` |
| **Equalizador Vocal de Presença (Vocal EQ)** | Atenua a frequência de 250 Hz (eliminando o som de voz abafada/caixa) e reforça 3.500 Hz (trazendo clareza, ar e inteligibilidade). | `npu-ctl mic-eq [on/off]` |
| **Compressor Vocal & AGC** | Nivela automaticamente o volume da sua fala para o padrão de transmissão (-18 dBFS). Se você falar baixo ou se afastar, o ganho sobe; se falar alto, o limitador atua suavemente sem distorcer. | `npu-ctl mic-comp [-12 a -28 dB]` |
| **De-Esser Dinâmico** | Reduz sibilâncias estridentes e chiados incômodos nas letras "S", "CH" e "X" na faixa entre 5,2 kHz e 7,5 kHz. | `npu-ctl mic-deesser [on/off]` |
| **Desreverberação de Sala (De-Reverb)** | Subtrai reflexões acústicas e eco de salas vazias ou ambientes com piso frio e paredes lisas. | `npu-ctl mic-dereverb [10-100]` |

### Ferramentas de Teste e Monitoramento de Voz:
* **Escutar a própria voz ao vivo**:
  ```bash
  npu-ctl mic-listen on   # Liga o retorno no fone de ouvido para ouvir os efeitos
  npu-ctl mic-listen off  # Desliga o retorno
  ```
* **Gravar e reproduzir um teste rápido**:
  ```bash
  npu-ctl audio-test 5    # Grava 5 segundos da NPU e toca para você avaliar
  ```

---

## 5. Controle pela Interface Gráfica (Cameractrls)

O aplicativo oficial de controle de câmera foi aprimorado com uma interface dedicada:

1. Abra o aplicativo **Cameractrls** no menu Iniciar ou terminal.
2. Selecione **`Intel NPU Enhanced Webcam (/dev/video72)`**.
3. **Melhorias de Layout Aplicadas**:
   - **Tamanho padrão confortável**: A janela abre em 520x680px com altura mínima travada em 580px (não abre minúscula).
   - **Barra de rolagem permanente**: A scrollbar vertical fica sempre visível, permitindo navegar por todos os controles sem precisar maximizar a janela.
4. **Controles na Aba "Intel NPU"**:
   - Todos os botões liga/desliga e controles deslizantes para desfoque, enquadramento, suavização, iluminação, standby, recolhimento PTZ e efeitos de áudio.
5. **Pré-visualização**:
   - Clique no ícone de câmera (Preview) no canto superior direito para ver a imagem 1080p e acompanhar as mudanças em tempo real.

---

## 6. Controle por Linha de Comando (`npu-ctl`)

O utilitário `npu-ctl` permite controlar tudo via terminal:

```bash
# === STATUS E INFORMAÇÕES ===
npu-ctl status                  # Status completo da NPU, dispositivos e efeitos
npu-ctl logs [cam|audio]        # Exibe logs em tempo real dos serviços

# === CONTROLE DE SERVIÇOS ===
npu-ctl restart                 # Reinicia ambos os serviços (vídeo e áudio)
npu-ctl start | npu-ctl stop    # Inicia ou para os serviços

# === VÍDEO & VISÃO COMPUTACIONAL ===
npu-ctl blur on | off           # Liga ou desliga desfoque
npu-ctl blur 45                 # Ajusta intensidade do desfoque (0 a 100%)
npu-ctl blur-mode standard|portrait # Desfoque uniforme vs Retrato óptico com profundidade
npu-ctl feather 40              # Calibra largura da borda suave (10 a 80px)
npu-ctl background /caminho/img # Define plano de fundo virtual (ou none para limpar)
npu-ctl parallax on | off | 40  # Efeito Parallax 3D tridimensional no fundo
npu-ctl gesture on | off | reaction | mute | all # Reconhecimento de gestos (👍, ✌️, 🖐️)
npu-ctl objects on | off | 60   # Preservação de objetos segurados na mão
npu-ctl chair on | off | 50     # Preservação de cadeira de escritório / gamer
npu-ctl privacy-fade on | off   # Transição suave (fade in/out) na ausência
npu-ctl artistic off|illustrated|animated|watercolor [10-100] # Filtros criativos
npu-ctl color none|teal_orange|vintage|noir|vibrant [10-100]  # Cores de cinema (LUTs)
npu-ctl rim on | off | 50 warm|white|cool # Luz de recorte de estúdio (Rim/Hair Light)
npu-ctl glare on | off | 50     # Supressão de reflexo azul de tela de monitor
npu-ctl framing on | off        # Enquadramento automático inteligente
npu-ctl framing-mode single|group # Enquadramento individual vs Enquadramento de grupo
npu-ctl framing smooth 3        # Velocidade do enquadramento (1=cinema, 10=ágil)
npu-ctl eyecontact off|natural|teleprompter # Contato visual com modo Teleprompter
npu-ctl smooth 35               # Suavização de pele e redução de ruído (10 a 100)
npu-ctl studio on | 50 | off    # Ring Light virtual (luz de estúdio)
npu-ctl lowlight on | 40 | off  # Compensador de pouca luz
npu-ctl sharpen on | 35 | off   # Realce de nitidez e super-resolução
npu-ctl privacy on | off        # Modo privacidade por ausência

# === ENERGIA, STANDBY & GIMBAL PTZ ===
npu-ctl standby on | off        # Standby automático inteligente
npu-ctl ptz-park on | off       # Recolhimento físico da câmera em standby
npu-ctl ptz-delay 1.0           # Atraso (s) após desligar o sensor antes de acionar os motores
npu-ctl ptz-tilt -324000 0      # Configura valores de tilt de repouso e trabalho
npu-ctl ptz-test -324000        # Move o motor PTZ imediatamente para teste

# === ÁUDIO DE ESTÚDIO & VOZ ===
npu-ctl mic-noise on | off      # Supressão de ruído neural PoCoNet
npu-ctl studio-mic on | off     # Pacote completo de estúdio vocal
npu-ctl mic-gate -45            # Portão de silêncio absoluto (-30 a -60 dB)
npu-ctl mic-comp -18            # Compressor vocal alvo (-12 a -28 dB)
npu-ctl mic-eq on | off         # Equalizador vocal de presença
npu-ctl mic-deesser on | off    # Atenuador de sibilâncias (S/CH)
npu-ctl mic-dereverb 40         # Desreverberação de sala com eco (10-100%)
npu-ctl mic-lowcut on | off     # Corte de vibrações de digitação e mesa (80Hz)
npu-ctl mic-listen on | off     # Retorno de voz em tempo real no fone
npu-ctl audio-test 5            # Gravação e reprodução de teste de 5s
```

---

## 7. Seleção de Dispositivos de Entrada

Você pode escolher exatamente qual webcam e qual microfone físico alimentam os modelos de IA:

### Webcams:
```bash
# Listar todas as webcams físicas disponíveis e ver a ativa:
npu-ctl cam-source

# Selecionar dispositivo:
npu-ctl cam-source pixy        # Usa a EMEET PIXY (/dev/video1)
npu-ctl cam-source notebook    # Usa a webcam integrada do notebook (/dev/video0)
npu-ctl cam-source auto        # Seleção automática
```
> Quando a webcam integrada do Galaxy Book 4 Ultra voltar a funcionar após atualizações de kernel da Intel/KDE neon (driver IPU6), basta executar `npu-ctl cam-source notebook` para que o daemon passe a aplicar todos os efeitos nela.

### Microfones:
```bash
# Listar todos os microfones físicos detectados no PipeWire:
npu-ctl mic-source

# Selecionar dispositivo:
npu-ctl mic-source pixy        # Microfone da EMEET PIXY
npu-ctl mic-source usb         # Microfone USB dedicado
npu-ctl mic-source notebook    # Microfone embutido no Galaxy Book
npu-ctl mic-source auto        # Automático (prioriza USB, depois interno)
```

---

## 8. Arquivo de Configuração

Todas as configurações são mantidas no arquivo JSON:
👉 `~/.config/npu-effects/config.json`

O daemon de áudio recarrega este arquivo a quente (**sem interromper chamadas**) assim que ele é modificado. O daemon de vídeo aplica a maioria dos parâmetros a quente a cada ciclo de inferência.

Exemplo da estrutura do `config.json`:
```json
{
  "device": "NPU",
  "video_device": "auto",
  "input_device": "auto",
  "effects": {
    "blur": { "enabled": true, "strength": 40 },
    "virtual_background": { "enabled": false, "image_path": "" },
    "auto_framing": { "enabled": true, "smoothness": 3.0, "deadzone": 0.10 },
    "skin_smoothing": { "enabled": true, "strength": 35 },
    "sharpen": { "enabled": false, "strength": 35 },
    "studio_light": { "enabled": false, "intensity": 40 },
    "low_light": { "enabled": false, "gain": 40 },
    "eye_contact": { "enabled": false },
    "feather": 40,
    "standby_on_idle": true,
    "idle_timeout_sec": 3.0,
    "ptz_park_enabled": true,
    "ptz_standby_tilt": -324000,
    "ptz_wakeup_tilt": 0
  },
  "audio_effects": {
    "noise_suppression": { "enabled": true },
    "studio_voice": { "enabled": true },
    "low_cut": { "enabled": true, "cutoff_hz": 80 },
    "smart_gate": { "enabled": true, "threshold_db": -45.0 },
    "vocal_eq": { "enabled": true },
    "compressor": { "enabled": true, "target_db": -18.0 },
    "deesser": { "enabled": true },
    "dereverb": { "enabled": true, "strength": 0.4 }
  }
}
```

---

## 9. Persistência em Atualizações do Sistema

* **Atualizações do KDE neon / Ubuntu (`pkcon`, `apt upgrade`):**
  - **100% preservadas.** O sistema de efeitos e modelos residem centralmente em `/opt/npu-effects/` e as preferências pessoais no diretório de cada usuário (`~/.config/npu-effects/`). Atualizações de pacotes do sistema operacional não sobrescrevem essas configurações.
* **Módulo do Kernel (`v4l2loopback`):**
  - Configurado via DKMS e `/etc/modprobe.d/v4l2loopback.conf`. Sempre que um novo kernel for instalado, o módulo é recompilado automaticamente.
* **Se você reinstalar ou atualizar o Cameractrls via GitHub:**
  - Caso clone uma versão limpa do Cameractrls, você pode reinstalar a aba da NPU e os ajustes de janela instantaneamente com o comando:
    ```bash
    npu-ctl restore-cameractrls
    ```
  - Backups de segurança dos arquivos originais ficam salvos com extensão `.npu_backup`.

---

## 10. Suporte Multi-Usuário (Compartilhamento da NPU)

A infraestrutura foi desenhada para que **todos os usuários do computador** compartilhem a mesma instalação, sem desperdiçar disco:
- **Compartilhado (única instalação):** Ambiente virtual Python, modelos de IA e binários em `/opt/npu-effects/`.
- **Individual por usuário:** Cada usuário tem seu arquivo de efeitos (`~/.config/npu-effects/config.json`) e fundos virtuais (`~/.config/npu-effects/backgrounds/`).

### Como liberar o uso para outro usuário do computador:
```bash
# 1. Liberar para um usuário específico:
sudo npu-ctl setup-user nome_do_usuario
# Ou:
sudo ./scripts/07_setup_users.sh nome_do_usuario

# 2. Liberar para TODOS os usuários existentes de uma só vez:
sudo npu-ctl setup-all-users
# Ou:
sudo ./scripts/07_setup_users.sh --all
```

Ao fazer login na conta do novo usuário:
- O atalho **"Cameractrls"** já estará no menu de aplicativos.
- O comando `npu-ctl` estará disponível no terminal.
- Basta executar `npu-ctl start` para iniciar os efeitos de áudio e vídeo na sessão dele!


---

*Desenvolvido e calibrado sob medida para o Samsung Galaxy Book 4 Ultra.*
