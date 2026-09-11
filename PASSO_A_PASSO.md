# 🛠️ Passo a Passo Completo: Instalação e Configuração da Intel NPU
### Samsung Galaxy Book 4 Ultra — Intel Core Ultra 9 185H — KDE neon / Ubuntu 24.04

Este documento descreve detalhadamente o passo a passo técnico de como toda a infraestrutura da NPU (Intel AI Boost) foi configurada para processamento de **Webcam Full HD 1080p** e **Áudio de Estúdio Broadcast**, permitindo a qualquer momento reproduzir, reinstalar ou auditar cada etapa do sistema.

---

## 📂 1. Estrutura de Arquivos Deste Diretório

A pasta foi montada de forma modular e autocontida:

```
OpenNPUEffects/
├── install_all.sh                    # Script mestre de instalação completa (1 clique)
├── uninstall.sh                      # Script mestre de desinstalação e limpeza
├── PASSO_A_PASSO.md                  # Este manual técnico passo a passo
├── GUIA_NPU_STUDIO.md                # Manual do usuário com catálogo de efeitos e comandos
├── scripts/
│   ├── 01_install_npu_drivers.sh     # Drivers Level Zero e pacotes UMD da NPU
│   ├── 02_setup_v4l2loopback.sh      # Câmera virtual /dev/video72 com 6 buffers
│   ├── 03_setup_dependencies.sh      # Dependências de sistema e venv Python
│   ├── 04_download_models.sh         # Download e preparação dos modelos neurais (Áudio e Vídeo)
│   ├── 05_setup_pipeline_daemons.sh  # Daemons de streaming, npu-ctl e serviços systemd
│   ├── 06_setup_cameractrls.sh       # Integração na interface gráfica Cameractrls
│   └── 07_setup_users.sh             # Configuração de permissões e perfil multi-usuário
├── src/
│   ├── bin/                          # Código-fonte dos daemons e do utilitário npu-ctl
│   │   ├── npu_webcam_daemon.py      # Daemon de vídeo com IA na NPU e controle PTZ
│   │   ├── npu_audio_daemon.py       # Daemon de áudio com IA na NPU e DSP vocal
│   │   └── npu-ctl                   # Ferramenta de linha de comando CLI
│   ├── config/
│   │   └── config.json               # Configuração padrão de parâmetros e efeitos
│   ├── systemd/
│   │   ├── npu-webcam.service        # Unit de inicialização do serviço de vídeo
│   │   └── npu-audio.service         # Unit de inicialização do serviço de áudio
│   ├── v4l2loopback/
│   │   └── v4l2loopback.conf         # Configuração do módulo de kernel
│   └── cameractrls/                  # Arquivos integrados para a interface Cameractrls
│       ├── cameractrls.py            # Backend com classe IntelNPUCtrls
│       ├── cameractrlsgtk.py         # GUI com scrollbar fixa e altura confortável
│       ├── cameractrls               # Launcher executável no PATH
│       └── hu.irl.cameractrls.desktop # Atalho do menu de aplicativos
├── models/                           # Modelos de IA pré-otimizados
│   ├── audio/                        # Intel PoCoNet FP16 para remoção de ruído
│   └── video/                        # YuNet (rosto) + Selfie Segmentation + YOLO11-seg (Cadeira/Objetos)
└── drivers/                          # Pacotes .deb oficiais dos drivers da Intel NPU
    ├── libze1_1.28.2.deb
    ├── intel-fw-npu_*.deb
    ├── intel-driver-compiler-npu_*.deb
    └── intel-level-zero-npu_*.deb
```

---

## ⚡ 2. Como Executar a Instalação

### Opção A: Instalação Completa Automática (Recomendado)
Para rodar todas as etapas em sequência com verificações automáticas:
```bash
git clone https://github.com/hermsdorff/OpenNPUEffects.git
cd OpenNPUEffects
./install_all.sh
```

### Opção B: Instalação Passo a Passo Manual
Você pode executar cada etapa individualmente na ordem numérica:
```bash
cd scripts
./01_install_npu_drivers.sh
./02_setup_v4l2loopback.sh
./03_setup_dependencies.sh
./04_download_models.sh
./05_setup_pipeline_daemons.sh
./06_setup_cameractrls.sh
./07_setup_users.sh [usuario|--all]
```

---

## 🔍 3. Detalhamento de Cada Etapa

---

### ETAPA 1: Drivers da Intel NPU e Nível de Hardware (`01_install_npu_drivers.sh`)

#### O que faz:
1. **Verificação do Kernel**: Garante que o módulo `intel_vpu` (driver do kernel Linux que se comunica com o coprocessador Meteor Lake) esteja carregado.
2. **Instalação dos Pacotes Oficiais da Intel**:
   - `libze1` (v1.28.2): Carregador da biblioteca OneAPI Level Zero.
   - `intel-fw-npu`: Firmware oficial binário que é injetado no chip da NPU (`vpu_37xx_v1.bin`).
   - `intel-driver-compiler-npu`: Compilador de microcódigo que converte grafos OpenVINO para instruções dos núcleos de processamento neural SHAVE.
   - `intel-level-zero-npu`: Driver em espaço de usuário (`libze_intel_npu.so.1`).
3. **Permissões de Usuário e Grupos**:
   - O dispositivo de hardware da NPU é registrado pelo kernel em `/dev/accel/accel0` com permissão de grupo `render`.
   - O script adiciona o usuário aos grupos `render` e `video` (`sudo usermod -aG render,video $USER`).
4. **Regra UDEV**: Cria `/etc/udev/rules.d/10-intel-vpu.rules` para garantir que `/dev/accel/accel*` sempre receba permissão de leitura/escrita para o grupo `render` sem exigir sudo.

#### Como auditar/verificar:
```bash
# Verificar presença do dispositivo da NPU:
ls -l /dev/accel/accel0

# Verificar se o driver do kernel está ativo:
dmesg | grep -i vpu
```

---

### ETAPA 2: Câmera Virtual V4L2 (`02_setup_v4l2loopback.sh`)

#### O que faz:
1. Instala o módulo de kernel dinâmico `v4l2loopback-dkms` e utilitários `v4l-utils`.
2. Cria a configuração persistente em `/etc/modprobe.d/v4l2loopback.conf`:
   ```ini
   options v4l2loopback devices=3 video_nr=70,71,72 card_label="Iriun Webcam","OBS Virtual Cam","Intel NPU Enhanced Webcam" exclusive_caps=1,1,1 max_buffers=6
   ```
   - **`video_nr=70,71,72`**: Reserva números fixos para evitar conflito com webcams físicas USB que usam `/dev/video0` e `/dev/video1`.
   - **`card_label`**: Define o nome amigável *"Intel NPU Enhanced Webcam"*, facilitando a seleção no Google Meet, Zoom, Teams, etc.
   - **`max_buffers=6`**: **Crítico!** O software Cameractrls (`cameraview.py:509`) exige estritamente 6 buffers de streaming. Sem essa configuração, o Cameractrls falhava com erro de *insufficient buffer memory*.
   - **`exclusive_caps=1`**: Evita que navegadores tentem abrir a câmera como dispositivo de gravação/saída em vez de captura.
3. Adiciona `v4l2loopback` em `/etc/modules-load.d/v4l2loopback.conf` para carregar automaticamente em todo boot.

#### Como auditar/verificar:
```bash
v4l2-ctl --device=/dev/video72 --all
```

---

### ETAPA 3: Dependências e Ambiente Virtual Python (`03_setup_dependencies.sh`)

#### O que faz:
1. Instala ferramentas de sistema e bibliotecas matemáticas: `python3-venv`, `python3-pip`, `build-essential`, `libopenblas-dev`, `ffmpeg`, `libportaudio2`.
2. Cria um ambiente virtual Python centralizado e compartilhado em:
   👉 `/opt/npu-effects/venv/` (com permissão de execução para todos os usuários)
3. Instala as versões compatíveis dos pacotes Python:
   - **`openvino`**: Runtime oficial da Intel com backend nativo para a NPU Meteor Lake.
   - **`opencv-python-headless`**: Visão computacional rápida com aceleração AVX2/AVX-512.
   - **`sounddevice`**: Captura e reprodução de áudio em tempo real com baixa latência via ALSA/Pulse/PipeWire.
   - **`pyvirtualcam`**: Injeção direta de quadros no `/dev/video72` em formato YUV420p sem overhead de conversão externa.
   - **`numpy` & `scipy`**: Operações matriciais vetorizadas para filtros de áudio e interpolação de imagem.

#### Como auditar/verificar:
```bash
/opt/npu-effects/venv/bin/python3 -c "import openvino as ov; print(ov.Core().available_devices)"
# Saída esperada: ['CPU', 'GPU', 'NPU']
```

---

### ETAPA 4: Download e Preparação dos Modelos de IA (`04_download_models.sh`)

#### O que faz:
Configura os modelos centralmente em `/opt/npu-effects/models/` (compartilhado em modo leitura para todos os usuários):

1. **Modelo de Áudio — Intel PoCoNet (`noise-suppression-poconetlike-0001`)**:
   - Rede neural recorrente de cancelamento de ruído e isolamento vocal treinada pela Intel.
   - Processa blocos de áudio de 128ms em ~30ms na NPU, garantindo processamento em tempo real com grande margem de folga.
   - Armazenado em formato nativo OpenVINO IR (arquivo `.xml` de topologia + `.bin` de pesos em FP16).
2. **Modelo de Vídeo — Rastreamento de Rosto (`face_detection_yunet_2023mar.onnx`)**:
   - Detector facial ultraleve (OpenCV Zoo) com suporte a rotação e pontos de referência (olhos, nariz e boca).
   - Executa em menos de 1,2ms por quadro, alimentando a lógica de auto-framing inteligente.
3. **Modelo de Vídeo — Segmentação de Pessoas (`selfie_segmentation_static`)**:
   - Modelo de segmentação de corpo inteiro com formato de entrada estático `[1, 3, 144, 256]`, desenhado especificamente para a arquitetura de núcleos da NPU.
   - Executa a inferência de desfoque em apenas **~2.4 ms por quadro (~400 FPS)** na NPU.

---

### ETAPA 5: Daemons de Processamento e Serviços de Segundo Plano (`05_setup_pipeline_daemons.sh`)

#### O que faz:
1. **Instalação dos Daemons**:
   - Instala em `/opt/npu-effects/bin/` os daemons [`npu_webcam_daemon.py`](src/bin/npu_webcam_daemon.py) e [`npu_audio_daemon.py`](src/bin/npu_audio_daemon.py).
   - Capturam preferências individuais de cada usuário em `~/.config/npu-effects/config.json`.
2. **Instalação do utilitário `npu-ctl`**:
   - Instala link simbólico em `/usr/local/bin/npu-ctl` (disponível no PATH de qualquer usuário da máquina).
3. **Serviços Systemd Globais do Usuário**:
   - Instala `npu-webcam.service` e `npu-audio.service` em `/etc/systemd/user/`.
   - Como estão em `/etc/systemd/user/`, qualquer usuário pode gerenciá-los diretamente sem precisar duplicar arquivos.
   - Habilita e inicializa os serviços para o usuário atual (`systemctl --user enable --now ...`).

#### Como auditar/verificar:
```bash
# Ver status dos serviços:
systemctl --user status npu-webcam.service npu-audio.service

# Ver painel de controle interativo:
npu-ctl status
```

---

### ETAPA 6: Integração com a Interface Gráfica Cameractrls (`06_setup_cameractrls.sh`)

#### O que faz:
1. Instala as dependências GTK necessárias (`python3-gi`, `gir1.2-gtk-3.0`, `gir1.2-gtk-4.0`).
2. Instala a base do Cameractrls em `/opt/npu-effects/cameractrls/`.
3. Injeta a aba exclusiva **"Intel NPU"** com controles de Blur, Ring Light, Low Light, Denoise e Auto-Framing vinculados a `~/.config/npu-effects/config.json`.
4. Aprimora a interface GTK com barra de rolagem permanente e tamanho padrão confortável.
5. Configura o launcher global `/usr/local/bin/cameractrls` e o atalho de menu em `/usr/share/applications/hu.irl.cameractrls.desktop` (visível para todos os usuários).

---

### ETAPA 7: Configuração Multi-Usuário (`07_setup_users.sh`)

#### O que faz:
1. Adiciona o usuário aos grupos de hardware `render` (acesso à NPU `/dev/accel/accel0`) e `video` (acesso às webcams física e virtual).
2. Cria o diretório de configurações do usuário `~/.config/npu-effects/` e a pasta de fundos virtuais `~/.config/npu-effects/backgrounds/`.
3. Permite configurar um usuário específico ou todos os usuários do sistema com um só comando:
   ```bash
   # Configurar um usuário específico:
   sudo ./scripts/07_setup_users.sh nome_do_usuario
   # Ou via npu-ctl:
   npu-ctl setup-user nome_do_usuario

   # Configurar TODOS os usuários do computador de uma só vez:
   sudo ./scripts/07_setup_users.sh --all
   # Ou via npu-ctl:
   npu-ctl setup-all-users
   ```

---

## 🎯 4. Como Testar e Validar Tudo

Após a execução dos scripts, execute os testes abaixo:

### 1. Testar o Vídeo 1080p com NPU:
```bash
# Abrir a interface do Cameractrls
cameractrls
```
- No seletor de câmeras, escolha: **`Intel NPU Enhanced Webcam (/dev/video72)`**.
- Clique no ícone de câmera (Preview) no canto superior direito.
- Veja a imagem em Full HD 1080p, com desfoque de fundo e enquadramento automático funcionando.

### 2. Testar o Áudio e Ouvir o Cancelamento de Ruído:
```bash
# Escutar a sua própria voz ao vivo pelo fone com o filtro NPU ativo:
npu-ctl mic-listen on

# Quando terminar de testar:
npu-ctl mic-listen off

# Ou gravar um teste de 5 segundos:
npu-ctl audio-test 5
```

### 3. Testar a Troca de Webcams e Microfones Físicos:
```bash
# Ver câmeras físicas detectadas:
npu-ctl cam-source

# Ver microfones físicos detectados:
npu-ctl mic-source
```

---

## 🔧 5. Resolução de Dúvidas e Problemas Comuns (Troubleshooting)

### A câmera diz "dispositivo ocupado" ao ligar o daemon:
- Outro aplicativo pode estar com o dispositivo físico `/dev/video1` aberto diretamente. Feche aplicativos como Zoom ou Discord, ou execute:
  ```bash
  npu-ctl restart
  ```

### Os controles do Cameractrls sumiram após eu atualizar pelo GitHub:
- Se você clonar uma versão nova do Cameractrls ou reinstalar, basta rodar:
  ```bash
  npu-ctl restore-cameractrls
  ```
  Isso reinjeta automaticamente a aba Intel NPU e os ajustes de janela.

### A webcam interna do Galaxy Book 4 Ultra voltou a funcionar no kernel:
- O driver Intel IPU6 passará a registrar `/dev/video0`. Para mudar a fonte da NPU para ela, basta digitar:
  ```bash
  npu-ctl cam-source notebook
  ```

---

## 🗑️ 6. Como Desinstalar o Open NPU Effects (`uninstall.sh`)

Para remover a suíte, parar os serviços e restaurar o sistema:

```bash
# Desinstalação padrão (remove serviços, /opt/npu-effects e atalhos, preservando configs):
./uninstall.sh

# Desinstalação completa com expurgo (remove também ~/.config/npu-effects e regras de kernel):
./uninstall.sh --purge
```

---

*Para conhecer todos os efeitos disponíveis e comandos do dia a dia, consulte o manual de uso em:*
👉 [**`GUIA_NPU_STUDIO.md`**](GUIA_NPU_STUDIO.md)
