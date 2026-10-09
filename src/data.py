"""Gera o dataset sintético de sentimento por aspecto em avaliações de produtos.

Cada avaliação combina de 0 a 4 trechos, um por aspecto, cada trecho com um sentimento,
mais frases "de enchimento" sem aspecto. O rótulo é o JSON {aspecto: sentimento}.

Para medir generalização de verdade, as frases de cada (aspecto, sentimento) são divididas:
75% só aparecem no treino/validação e 25% só aparecem no teste. Além disso existe um teste
manual (data/teste_manual.jsonl), escrito à mão, com linguagem natural fora dos modelos.
"""
import argparse
import json
import random
import re
import unicodedata

from config import ASPECTS, DATA_DIR, SEED

PHRASES = {
    "entrega": {
        "positivo": [
            "a entrega foi super rápida", "chegou antes do prazo", "chegou em dois dias",
            "entregaram bem antes do previsto", "o pedido chegou rapidinho", "a transportadora foi muito ágil",
            "recebi no dia seguinte à compra", "o prazo de entrega foi cumprido com folga",
            "chegou muito antes do esperado", "a entrega foi rápida e sem problemas", "o frete foi rapidíssimo",
            "veio voando, chegou em 24 horas", "os correios entregaram rapidinho",
            "a entrega me surpreendeu de tão rápida", "chegou uma semana antes da data prevista",
            "a entrega foi relâmpago",
        ],
        "negativo": [
            "a entrega atrasou mais de uma semana", "demorou demais para chegar",
            "o pedido chegou com 15 dias de atraso", "a transportadora perdeu meu pacote",
            "o prazo de entrega não foi cumprido", "entregaram no endereço errado", "a entrega foi uma novela",
            "esperei quase um mês pelo pedido", "o rastreio ficou parado por dias",
            "chegou muito depois do prometido", "a entrega foi lenta demais",
            "tive que buscar na agência porque não entregaram", "remarcaram a entrega três vezes",
            "atrasaram a entrega sem nenhum aviso", "o frete demorou uma eternidade",
            "a entrega foi muito demorada",
        ],
        "neutro": [
            "a entrega foi no prazo normal", "chegou no prazo previsto, nem antes nem depois",
            "o tempo de entrega foi o padrão", "a entrega levou uns cinco dias, o normal",
            "o prazo de entrega foi o informado no site", "a entrega foi dentro do esperado",
            "chegou na data prevista", "a entrega foi comum, sem nada de especial",
            "o frete levou o tempo de sempre", "recebi na data estimada", "a entrega foi normal",
            "o prazo de entrega foi o de costume",
        ],
    },
    "qualidade": {
        "positivo": [
            "o produto é de ótima qualidade", "o material é excelente e bem resistente", "funciona perfeitamente",
            "o acabamento é impecável", "a qualidade superou minhas expectativas", "é muito bem feito e durável",
            "o som é incrível", "a tela tem cores lindas", "é muito resistente, já caiu e nada aconteceu",
            "funciona muito bem desde o primeiro dia", "é um produto de primeira linha",
            "o tecido é macio e de qualidade", "a bateria dura o dia inteiro", "está tudo funcionando perfeitamente",
            "a qualidade é excelente", "o produto é robusto e muito bem construído",
        ],
        "negativo": [
            "o produto veio com defeito", "parou de funcionar em uma semana", "o material é frágil e mal acabado",
            "veio todo arranhado", "quebrou no primeiro uso", "a qualidade é péssima",
            "o tecido desbotou na primeira lavagem", "a bateria não dura nada", "esquenta muito e desliga sozinho",
            "veio com peças faltando", "o som é muito ruim e chiado", "a costura soltou em poucos dias",
            "parece de brinquedo de tão frágil", "não funciona direito", "a tela veio trincada",
            "o produto é muito mal feito",
        ],
        "neutro": [
            "a qualidade é razoável", "o produto é ok, nada demais", "a qualidade é mediana",
            "é um produto simples, cumpre o básico", "o material é comum", "a qualidade não é boa nem ruim",
            "a qualidade é a esperada para a categoria", "funciona normalmente", "é um produto padrão",
            "o acabamento é simples", "atende o básico", "o desempenho é o normal para esse tipo de produto",
        ],
    },
    "preço": {
        "positivo": [
            "o preço estava ótimo", "comprei em promoção e valeu muito a pena", "o custo-benefício é excelente",
            "estava bem mais barato que nas outras lojas", "o preço é justo pelo que oferece",
            "paguei pouco por um produto desses", "o valor é muito em conta", "foi uma pechincha",
            "vale cada centavo", "o preço está abaixo do mercado", "o desconto foi excelente",
            "o preço foi muito camarada",
        ],
        "negativo": [
            "o preço é muito alto", "é caro demais pelo que entrega", "não vale o que custa",
            "achei o valor um absurdo", "paguei caro e me arrependi", "o preço é bem salgado",
            "o custo-benefício é ruim", "encontrei bem mais barato em outras lojas", "o valor cobrado é exagerado",
            "está caro para o que oferece", "o preço não compensa", "aumentaram o preço sem motivo",
        ],
        "neutro": [
            "o preço é o mesmo das outras lojas", "o valor é o normal de mercado", "o preço está na média",
            "não é caro nem barato", "o preço é compatível com o mercado", "paguei o preço normal",
            "o valor está dentro do esperado", "custa o mesmo que em qualquer loja",
            "o valor é o padrão para esse tipo de produto", "o preço é o de tabela",
        ],
    },
    "atendimento": {
        "positivo": [
            "o atendimento foi excelente", "o vendedor respondeu rapidinho todas as dúvidas",
            "o suporte resolveu meu problema na hora", "fui muito bem atendido", "o SAC foi atencioso e educado",
            "a loja trocou sem burocracia", "o vendedor foi super prestativo", "o pós-venda é muito bom",
            "me ajudaram em tudo pelo chat", "o atendimento foi nota dez", "responderam meu e-mail no mesmo dia",
            "a equipe de suporte foi muito simpática",
        ],
        "negativo": [
            "o atendimento é péssimo", "o SAC não resolve nada", "ninguém responde as mensagens",
            "o vendedor foi grosseiro", "fiquei horas esperando no chat", "abri três chamados e nenhum foi respondido",
            "o suporte me deixou sem resposta", "a loja se recusou a trocar", "o atendente desligou na minha cara",
            "o pós-venda é horrível", "me trataram com descaso", "prometeram retorno e nunca ligaram",
        ],
        "neutro": [
            "o atendimento foi normal", "o suporte deu uma resposta padrão", "o vendedor respondeu no tempo normal",
            "o atendimento foi comum", "falei com o SAC e foi um atendimento normal",
            "o chat respondeu dentro do horário comercial", "o atendimento foi protocolar",
            "o atendimento não foi bom nem ruim", "o atendente seguiu o roteiro de sempre",
            "a resposta do vendedor veio no prazo de costume",
        ],
    },
    "embalagem": {
        "positivo": [
            "a embalagem é muito caprichada", "veio muito bem embalado", "a embalagem é reforçada, com plástico bolha",
            "a caixa chegou intacta", "a embalagem é linda, ótima para presente", "veio tudo muito bem protegido na caixa",
            "embalaram com muito cuidado", "a embalagem veio perfeita", "veio em caixa resistente e lacrada",
            "capricharam na embalagem",
        ],
        "negativo": [
            "a embalagem veio rasgada", "a caixa chegou toda amassada", "veio mal embalado, sem proteção nenhuma",
            "a embalagem estava violada", "a caixa chegou molhada", "o lacre estava rompido",
            "a embalagem é frágil, sem plástico bolha", "veio solto dentro de uma caixa enorme",
            "a embalagem chegou aberta", "a caixa estava destruída",
        ],
        "neutro": [
            "a embalagem é simples", "veio na caixa padrão", "a embalagem é comum", "a embalagem é a de sempre",
            "veio numa embalagem básica", "a caixa é normal, sem nada de especial",
            "a embalagem é de papelão simples", "a embalagem é discreta",
        ],
    },
}

# Frases sem aspecto (contexto ou opinião geral): o modelo deve ignorá-las
FILLERS = [
    "Comprei para dar de presente.", "Comprei para usar no trabalho.", "É a segunda vez que compro.",
    "Minha esposa escolheu.", "Uso há duas semanas.", "Comprei na Black Friday.", "Vou usar na casa de praia.",
    "Estava precisando de um desses.", "Pesquisei bastante antes de comprar.", "Comprei pelo aplicativo.",
    "Recomendo!", "Não recomendo.", "Estou satisfeito.", "Que decepção.", "Nota 10.", "Gostei.",
    "Meu filho adorou.", "Comprei para substituir o antigo.", "Primeira compra nessa loja.", "Era o que eu queria.",
]

CONTRAST = [", mas ", ", porém ", ". Por outro lado, ", ", só que ", "; no entanto, ", ". Pena que "]
SAME = [", e ", ". ", ". Além disso, ", ", ", " e "]
N_ASPECTS_WEIGHTS = [0.05, 0.30, 0.40, 0.20, 0.05]  # probabilidade de 0, 1, 2, 3, 4 aspectos


def split_phrases(rng: random.Random):
    """Separa as frases de cada (aspecto, sentimento): 75% treino/validação, 25% só no teste."""
    train, test = {}, {}
    for aspect, by_sent in PHRASES.items():
        for sent, phrases in by_sent.items():
            shuffled = phrases[:]
            rng.shuffle(shuffled)
            cut = round(len(shuffled) * 0.75)
            train[(aspect, sent)] = shuffled[:cut]
            test[(aspect, sent)] = shuffled[cut:]
    return train, test


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def make_review(rng: random.Random, bank: dict) -> dict:
    n = rng.choices(range(len(N_ASPECTS_WEIGHTS)), N_ASPECTS_WEIGHTS)[0]
    aspects = rng.sample(ASPECTS, n)
    label = {a: rng.choice(["positivo", "negativo", "neutro"]) for a in aspects}

    text = ""
    prev = None
    for aspect in aspects:
        phrase = rng.choice(bank[(aspect, label[aspect])])
        if not text:
            text = phrase
        else:
            text += rng.choice(CONTRAST if label[aspect] != prev else SAME) + phrase
        prev = label[aspect]
    if text:
        text = text[0].upper() + text[1:] + rng.choice([".", ".", ".", "!", ""])
        text = re.sub(r"([.!?] )(\w)", lambda m: m.group(1) + m.group(2).upper(), text)

    # Frases sem aspecto antes ou depois (sempre pelo menos uma se não houver aspecto)
    for _ in range(rng.choices([0, 1, 2], [0.5, 0.4, 0.1])[0] or (0 if aspects else 1)):
        filler = rng.choice(FILLERS)
        if rng.random() < 0.5 or not text:
            text = f"{filler} {text}".strip()
        else:
            text = f"{text if text[-1] in '.!' else text + '.'} {filler}"

    # Ruído de texto real: tudo minúsculo, sem acento
    if rng.random() < 0.25:
        text = text.lower()
    if rng.random() < 0.15:
        text = strip_accents(text)

    ordered = {a: label[a] for a in ASPECTS if a in label}
    return {"text": text, "label": ordered}


def generate(n: int, bank: dict, rng: random.Random, seen: set) -> list[dict]:
    rows = []
    while len(rows) < n:
        row = make_review(rng, bank)
        if row["text"] not in seen:
            seen.add(row["text"])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=400)
    args = ap.parse_args()

    rng = random.Random(SEED)
    train_bank, test_bank = split_phrases(rng)
    seen = set()
    splits = {
        "train": generate(args.train, train_bank, rng, seen),
        "val": generate(args.val, train_bank, rng, seen),
        "test": generate(args.test, test_bank, rng, seen),
    }
    DATA_DIR.mkdir(exist_ok=True)
    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} exemplos -> {path}")
    print("\nExemplos:")
    for row in splits["train"][:5]:
        print(f"  {row['text']}\n    -> {json.dumps(row['label'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
