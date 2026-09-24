"""Frontend (Gradio) para o usuário testar o modelo.

    python app/app.py --adapter outputs/r16_lr2e-4          # local
    python app/app.py --adapter outputs/r16_lr2e-4 --share  # no Colab (gera link público)
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import BASE_MODEL, LABELS, SYSTEM_PROMPT  # noqa: E402
from gpu_info import gpu_info  # noqa: E402
from modeling import classify, generate_stream, load_model  # noqa: E402

EXAMPLES = [
    "Produto chegou antes do prazo e funciona perfeitamente. Recomendo!",
    "Péssimo. Parou de funcionar depois de uma semana e a loja não responde.",
    "É ok. Cumpre o que promete, mas o acabamento poderia ser melhor pelo preço.",
    "Comprei para dar de presente, minha mãe adorou, mas a entrega atrasou 10 dias.",
]


def build_ui(model, tok, adapter: str | None, base_model: str = BASE_MODEL):
    has_adapter = adapter is not None

    def adapter_ctx(use_adapter: bool):
        # Com PEFT dá para desligar o LoRA e comparar com o modelo base sem carregar outro modelo
        if has_adapter and not use_adapter:
            return model.disable_adapter()
        return contextlib.nullcontext()

    def to_dict(p):
        return {label: float(p[i]) for i, label in enumerate(LABELS)}

    def run_classify(text: str, compare: bool):
        if not text.strip():
            raise gr.Error("Digite uma avaliação.")
        probs_ft = classify(model, tok, [text])[0]
        probs_base = None
        if has_adapter and compare:
            with adapter_ctx(False):
                probs_base = to_dict(classify(model, tok, [text])[0])
        return to_dict(probs_ft), probs_base

    def run_generate(system, prompt, max_tokens, temperature, top_p, use_adapter):
        if not prompt.strip():
            raise gr.Error("Digite um prompt.")
        with adapter_ctx(use_adapter):
            yield from generate_stream(model, tok, system, prompt, int(max_tokens), temperature, top_p)

    run_info = {}
    if has_adapter and (Path(adapter) / "run_info.json").exists():
        run_info = json.loads((Path(adapter) / "run_info.json").read_text())
        run_info.pop("log_history", None)

    with gr.Blocks(title="Sentimento PT-BR com LLM") as demo:
        gr.Markdown(
            "# Análise de sentimento de avaliações (PT-BR)\n"
            f"Modelo: **{base_model.split('/')[-1]}** {'+ LoRA `' + Path(adapter).name + '`' if has_adapter else '(sem fine-tuning)'}"
            f" · dispositivo: **{model.device}**"
        )
        with gr.Tab("Classificar avaliação"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(label="Avaliação do produto", lines=5,
                                      placeholder="Escreva aqui uma avaliação de produto...")
                    compare = gr.Checkbox(label="Comparar com o modelo base (sem LoRA)", value=has_adapter,
                                          visible=has_adapter)
                    btn = gr.Button("Classificar", variant="primary")
                    gr.Examples(EXAMPLES, inputs=text)
                with gr.Column():
                    out_ft = gr.Label(label="Modelo treinado" if has_adapter else "Modelo base", num_top_classes=3)
                    out_base = gr.Label(label="Modelo base (zero-shot)", num_top_classes=3, visible=has_adapter)
            btn.click(run_classify, [text, compare], [out_ft, out_base])
            text.submit(run_classify, [text, compare], [out_ft, out_base])

        with gr.Tab("Prompt livre"):
            gr.Markdown("Crie seu próprio prompt e converse com o modelo.")
            with gr.Row():
                with gr.Column():
                    system = gr.Textbox(label="Prompt de sistema", value=SYSTEM_PROMPT, lines=3)
                    prompt = gr.Textbox(label="Seu prompt", lines=6,
                                        placeholder="Ex.: Classifique: 'A bateria dura pouco, mas a câmera é ótima.'")
                    with gr.Accordion("Parâmetros de geração", open=False):
                        max_tokens = gr.Slider(8, 1024, value=256, step=8, label="Máx. de tokens")
                        temperature = gr.Slider(0, 1.5, value=0.7, step=0.05, label="Temperatura (0 = determinístico)")
                        top_p = gr.Slider(0.1, 1, value=0.9, step=0.05, label="Top-p")
                        use_adapter = gr.Checkbox(label="Usar adaptador LoRA", value=has_adapter,
                                                  visible=has_adapter)
                    gen_btn = gr.Button("Gerar", variant="primary")
                with gr.Column():
                    answer = gr.Textbox(label="Resposta", lines=14)
            gen_btn.click(run_generate, [system, prompt, max_tokens, temperature, top_p, use_adapter], answer)

        with gr.Tab("Modelo e GPU"):
            gr.JSON(value=gpu_info(), label="GPU")
            if run_info:
                gr.JSON(value=run_info, label="Configuração do treino")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="pasta do adaptador LoRA (ex.: outputs/r16_lr2e-4)")
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--share", action="store_true", help="gera link público (necessário no Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    model, tok = load_model(args.adapter, args.base_model)
    build_ui(model, tok, args.adapter, args.base_model).queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
