# GPU no VSCode: configuração e print exigido

O projeto foi executado na **GPU do Apple M5 (8 núcleos, Metal 4)** usando o backend **MPS (Metal Performance Shaders)**
do PyTorch. O código também roda em **GPU NVIDIA com CUDA** sem nenhuma alteração: `src/config.py` escolhe
automaticamente CUDA → MPS → CPU.

## Por que Metal e não CUDA?

CUDA é uma plataforma exclusiva das placas NVIDIA. Macs com Apple Silicon (M1–M5) têm GPU integrada da Apple, que é
programada via **Metal**. O PyTorch acessa essa GPU pelo backend `mps`, que cumpre o mesmo papel do CUDA:

| | NVIDIA | Apple Silicon |
|---|---|---|
| API da GPU | CUDA + cuDNN | Metal + Metal Performance Shaders |
| Dispositivo no PyTorch | `torch.device("cuda")` | `torch.device("mps")` |
| Instalação | driver NVIDIA + torch com a versão CUDA certa | só o macOS 12.3+ e o `torch` padrão do PyPI |
| Memória | VRAM dedicada | memória unificada (CPU e GPU compartilham os 16 GB) |
| Monitoramento | `nvidia-smi` | `ioreg` (sem sudo), Monitor de Atividade, `powermetrics` |

## Configurando no Mac (o que foi feito neste projeto)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"   # deve imprimir True
```

No VSCode, instale as extensões **Python** e **Jupyter** (recomendadas automaticamente em `.vscode/extensions.json`)
e selecione o interpretador `.venv`.

## Como gerar o print da GPU dentro do VSCode

1. Abra a pasta do projeto no VSCode.
2. Abra `notebooks/projeto.ipynb`, selecione o kernel `.venv` e rode a célula **"1. GPU instalada e funcionando"**.
   Também dá para usar **Terminal → Run Task → "GPU: informações"**.
   A saída mostra chip, GPU, núcleos, versão do Metal, backend MPS, memória e um benchmark em TFLOPS que prova
   que a GPU está processando.
3. Num segundo terminal do VSCode, rode **Run Task → "GPU: monitor ao vivo"** (`python src/gpu_monitor.py`).
   Ele mostra a utilização da GPU e a memória em uso, atualizadas a cada segundo.
4. Com um treino ou o frontend rodando, tire o print da janela do VSCode mostrando a saída do `gpu_info.py`
   e o monitor com a GPU perto de 100%.
5. Salve como `docs/img/gpu_vscode.png`.

> **Sobre os plugins GPU Monitor / GPU Environment:** eles dependem do `nvidia-smi` e só funcionam com placas NVIDIA.
> No Mac, o `src/gpu_monitor.py` faz o mesmo papel dentro do terminal do VSCode, lendo a utilização direto do driver
> da GPU Apple (`ioreg`). Se alguém do grupo tiver NVIDIA, o mesmo script detecta e passa a usar o `nvidia-smi`.

## Rodando em PC com NVIDIA (CUDA)

1. Instale o driver NVIDIA mais recente e rode `nvidia-smi`. O campo **CUDA Version** é a versão **máxima** de CUDA
   que o seu driver suporta.
2. Confira a compute capability da sua placa em <https://developer.nvidia.com/cuda-gpus> e instale o PyTorch com uma
   versão de CUDA **menor ou igual** à do driver, pelo seletor em <https://pytorch.org/get-started/locally/>.
   Exemplo para CUDA 12.8:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
3. Confirme: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"`.
4. Instale as extensões **GPU Monitor** e **GPU Environment** no VSCode e siga os mesmos passos do print acima.

## Dicas de desempenho no Mac (aprendidas na prática)

- **Deixe na tomada.** Na bateria, o macOS reduz o desempenho da GPU. A primeira tentativa de treino ficou quase
  parada e terminou com o Mac suspenso a 1% de bateria.
- **Use bf16 no treino.** Foi 4× mais rápido que fp32 e teve a mesma loss (`src/config.py`).
- **Batch 4 + acumulação 4** em vez de batch 8 + acumulação 2: o batch 8 fazia o Mac usar swap e ficava 2,7× mais lento,
  com o mesmo batch efetivo de 16.
- **Gradient checkpointing ligado:** sem ele, o treino estoura a memória da GPU (limite de ~11,8 GB em 16 GB de RAM).
