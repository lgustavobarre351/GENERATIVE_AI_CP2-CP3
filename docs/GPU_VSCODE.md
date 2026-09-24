# GPU no VSCode: como gerar o print exigido

O treino é feito em **GPU NVIDIA com CUDA**. Há duas formas de mostrar a GPU funcionando dentro do VSCode.

## Opção A — VSCode conectado ao Google Colab (GPU T4)

Use esta opção se ninguém do grupo tiver placa NVIDIA.

1. No VSCode, instale as extensões **Jupyter** e **Google Colab** (publicada pelo Google).
2. Abra `notebooks/treino_colab.ipynb`.
3. Clique em **Select Kernel → Colab → New Colab Server → GPU → T4** e faça login com a conta Google.
4. Rode as células `!nvidia-smi` e `!python src/gpu_info.py`.
5. Tire o print da janela do VSCode mostrando o notebook, o kernel Colab (T4) e a saída com nome da GPU, versão do CUDA e memória.
6. Salve a imagem em `docs/img/gpu_vscode.png`.

> As extensões *GPU Monitor* / *GPU Environment* leem a GPU **da máquina local**. Com o Colab, a GPU é remota,
> então o comprovante é a saída do `nvidia-smi` e do `gpu_info.py` dentro do VSCode.

## Opção B — PC local com placa NVIDIA

1. **Driver:** instale o driver NVIDIA mais recente e rode `nvidia-smi`. O campo **CUDA Version**, no canto superior direito,
   é a versão **máxima** de CUDA que o seu driver suporta.
2. **CUDA / PyTorch:** confira qual CUDA a sua placa suporta (arquitetura/compute capability) em
   <https://developer.nvidia.com/cuda-gpus> e instale o PyTorch compilado para uma versão de CUDA **menor ou igual** à do driver,
   usando o seletor em <https://pytorch.org/get-started/locally/>. Exemplo para CUDA 12.8:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
   (As wheels do PyTorch já trazem o runtime do CUDA; o CUDA Toolkit completo só é necessário para compilar extensões.)
3. Confirme: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"`.
4. Instale no VSCode as extensões de monitoramento (procure por **GPU Monitor** e **GPU Environment** no Marketplace).
5. Rode `python src/gpu_info.py` no terminal do VSCode, de preferência durante um treino, e tire o print mostrando
   a saída + o painel/barra de status das extensões com uso de GPU e memória.

### Memória necessária (Qwen2.5-1.5B + LoRA, batch 8, seq ≤ 384, gradient checkpointing)

| GPU | VRAM | Funciona? |
|---|---|---|
| T4 (Colab) | 16 GB | Sim (fp16) |
| RTX 3060 / 4060 Ti | 12–16 GB | Sim (bf16) |
| RTX 3050 / 4060 | 8 GB | Sim, com `--batch_size 4 --grad_accum 4` |
| < 6 GB | — | Use `--base_model Qwen/Qwen2.5-0.5B-Instruct` |
