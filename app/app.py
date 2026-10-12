"""Frontend (Gradio) para o usuário testar o modelo.

    python app/app.py --adapter outputs/r16_lr2e-4          # http://localhost:7860
    python app/app.py --adapter outputs/r16_lr2e-4 --share  # gera também um link público
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import ASPECTS, BASE_MODEL, SENTIMENTS, SYSTEM_PROMPT  # noqa: E402
from gpu_info import gpu_info  # noqa: E402
from modeling import analyze, generate_stream, load_model, parse_output  # noqa: E402

EXAMPLES = [
    "Chegou em dois dias e o fone é excelente, mas achei caro pelo que oferece.",
    "A caixa veio rasgada e o produto arranhado. Abri chamado e ninguém respondeu até agora.",
    "Preço na média, entrega no prazo normal. O liquidificador é potente e silencioso.",
    "Comprei para dar de presente para minha mãe.",
    "O vendedor foi super atencioso, mas o pedido atrasou mais de uma semana.",
]

COLORS = {"positivo": "#1a9e5c", "negativo": "#d6453d", "neutro": "#8a8f98"}
ICONS = {"positivo": "▲", "negativo": "▼", "neutro": "●"}


def render(raw: str, title: str) -> str:
    """Mostra o resultado como uma lista de aspectos coloridos + a resposta crua do modelo."""
    parsed = parse_output(raw)
    if parsed is None:
        body = "<p style='color:#d6453d'><b>Resposta fora do formato JSON</b></p>"
    elif not parsed:
        body = "<p style='opacity:.7'>Nenhum aspecto mencionado.</p>"
    else:
        rows = "".join(
            f"<tr><td style='padding:6px 12px 6px 0'><b>{a}</b></td>"
            f"<td><span style='background:{COLORS[s]};color:white;padding:3px 10px;border-radius:12px'>"
            f"{ICONS[s]} {s}</span></td></tr>"
            for a, s in parsed.items())
        body = f"<table>{rows}</table>"
    return (f"<div style='padding:4px'><h3 style='margin:0 0 8px'>{title}</h3>{body}"
            f"<p style='margin-top:10px;font-size:12px;opacity:.7'>Resposta do modelo: <code>{raw}</code></p></div>")


def build_ui(model, tok, adapter: str | None, base_model: str = BASE_MODEL):
    has_adapter = adapter is not None

    def adapter_ctx(use_adapter: bool):
        # Com PEFT dá para desligar o LoRA e comparar com o modelo base sem carregar outro modelo na memória
        if has_adapter and not use_adapter:
            return model.disable_adapter()
        return contextlib.nullcontext()

    def run_single(text: str, compare: bool):
        if not text.strip():
            raise gr.Error("Digite uma avaliação.")
        ft = render(analyze(model, tok, [text])[0], "Modelo treinado (LoRA)" if has_adapter else "Modelo base")
        base = ""
        if has_adapter and compare:
            with adapter_ctx(False):
                base = render(analyze(model, tok, [text])[0], "Modelo base (sem treino)")
        return ft, base

    def run_batch(texts: str):
        reviews = [t.strip() for t in texts.splitlines() if t.strip()]
        if not reviews:
            raise gr.Error("Cole uma avaliação por linha.")
        preds = [parse_output(r) or {} for r in analyze(model, tok, reviews)]
        table = pd.DataFrame([{"avaliação": t, **{a: p.get(a, "") for a in ASPECTS}} for t, p in zip(reviews, preds)])
        summary = pd.DataFrame([{"aspecto": a, **{s: sum(p.get(a) == s for p in preds) for s in SENTIMENTS}}
                                for a in ASPECTS])
        return table, summary

    def run_generate(system, prompt, max_tokens, temperature, top_p, thinking, use_adapter):
        if not prompt.strip():
            raise gr.Error("Digite um prompt.")
        with adapter_ctx(use_adapter):
            yield from generate_stream(model, tok, system, prompt, int(max_tokens), temperature, top_p, thinking)

    run_info = {}
    if has_adapter and (Path(adapter) / "run_info.json").exists():
        run_info = json.loads((Path(adapter) / "run_info.json").read_text())
        run_info.pop("log_history", None)

    with gr.Blocks(title="Sentimento por aspecto") as demo:
        gr.Markdown(
            "# Sentimento por aspecto em avaliações de produtos\n"
            f"Modelo **{base_model.split('/')[-1]}** "
            f"{'+ LoRA `' + Path(adapter).name + '`' if has_adapter else '(sem fine-tuning)'} · "
            f"dispositivo **{model.device}** · aspectos: {', '.join(ASPECTS)}"
        )
        with gr.Tab("Analisar avaliação"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(label="Avaliação do produto", lines=5,
                                      placeholder="Ex.: Chegou rápido, mas veio arranhado.")
                    compare = gr.Checkbox(label="Comparar com o modelo base (sem treino)", value=has_adapter,
                                          visible=has_adapter)
                    btn = gr.Button("Analisar", variant="primary")
                    gr.Examples(EXAMPLES, inputs=text)
                with gr.Column():
                    out_ft = gr.HTML()
                    out_base = gr.HTML()
            btn.click(run_single, [text, compare], [out_ft, out_base])
            text.submit(run_single, [text, compare], [out_ft, out_base])

        with gr.Tab("Várias avaliações"):
            gr.Markdown("Cole **uma avaliação por linha** para analisar em lote e ver o resumo por aspecto.")
            texts = gr.Textbox(label="Avaliações", lines=8, value="\n".join(EXAMPLES))
            batch_btn = gr.Button("Analisar todas", variant="primary")
            table = gr.Dataframe(label="Resultado por avaliação", wrap=True,
                                 column_widths=["45%"] + ["11%"] * len(ASPECTS))
            summary = gr.Dataframe(label="Resumo por aspecto")
            batch_btn.click(run_batch, texts, [table, summary])

        with gr.Tab("Prompt livre"):
            gr.Markdown("Crie seu próprio prompt e converse com o modelo.")
            with gr.Row():
                with gr.Column():
                    system = gr.Textbox(label="Prompt de sistema", value=SYSTEM_PROMPT, lines=6)
                    prompt = gr.Textbox(label="Seu prompt", lines=5,
                                        placeholder="Ex.: Avaliação: A bateria dura pouco, mas a câmera é ótima.")
                    with gr.Accordion("Parâmetros de geração", open=False):
                        max_tokens = gr.Slider(8, 2048, value=256, step=8, label="Máx. de tokens")
                        temperature = gr.Slider(0, 1.5, value=0.7, step=0.05, label="Temperatura (0 = determinístico)")
                        top_p = gr.Slider(0.1, 1, value=0.9, step=0.05, label="Top-p")
                        thinking = gr.Checkbox(label="Modo raciocínio do Qwen3 (<think>)", value=False)
                        use_adapter = gr.Checkbox(label="Usar adaptador LoRA", value=has_adapter,
                                                  visible=has_adapter)
                    gen_btn = gr.Button("Gerar", variant="primary")
                with gr.Column():
                    answer = gr.Textbox(label="Resposta", lines=16)
            gen_btn.click(run_generate, [system, prompt, max_tokens, temperature, top_p, thinking, use_adapter],
                          answer)

        with gr.Tab("Modelo e GPU"):
            gr.JSON(value=gpu_info(), label="GPU")
            if run_info:
                gr.JSON(value=run_info, label="Configuração do treino")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="pasta do adaptador LoRA (ex.: outputs/r16_lr2e-4)")
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--share", action="store_true", help="gera link público")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    model, tok = load_model(args.adapter, args.base_model)
    build_ui(model, tok, args.adapter, args.base_model).queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
