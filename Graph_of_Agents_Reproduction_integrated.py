import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


BASE = Path.cwd()
DATA_PATH = BASE / "data" / "questions.json"
RESULTS_DIR = BASE / "results"
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


def build_math_question(rng, idx):
    a = rng.randint(6, 30)
    b = rng.randint(2, 12)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        correct = a + b
        stem = f"{a} + {b} 的结果是多少？"
    elif op == "-":
        correct = a - b
        stem = f"{a} - {b} 的结果是多少？"
    else:
        correct = a * b
        stem = f"{a} × {b} 的结果是多少？"

    wrong = set()
    while len(wrong) < 3:
        offset = rng.randint(1, 8)
        candidate = correct + rng.choice([-1, 1]) * offset
        if candidate != correct and candidate >= 0:
            wrong.add(candidate)

    options = [str(correct)] + [str(x) for x in wrong]
    rng.shuffle(options)
    return {
        "id": f"math-{idx}",
        "domain": "math",
        "stem": stem,
        "options": options,
        "answer": options.index(str(correct)),
    }


def build_fixed_questions(domain, rows):
    questions = []
    for idx, row in enumerate(rows, start=1):
        stem, options, correct = row
        questions.append(
            {
                "id": f"{domain}-{idx}",
                "domain": domain,
                "stem": stem,
                "options": options,
                "answer": options.index(correct),
            }
        )
    return questions


CODING = [
    ("Python 中定义函数使用哪个关键字？", ["func", "def", "function", "lambda"], "def"),
    ("Python 中获取列表长度使用哪个函数？", ["size()", "length()", "len()", "count()"], "len()"),
    ("Python 中单行注释使用哪个符号？", ["//", "#", "--", "/*"], "#"),
    ("Python 中输出内容通常使用哪个函数？", ["print()", "echo()", "write()", "output()"], "print()"),
    ("Python 中判断两个值是否相等使用哪个运算符？", ["=", "==", "===", "!="], "=="),
    ("Python 中创建空集合应使用哪个写法？", ["{}", "[]", "set()", "()"], "set()"),
    ("Python 中创建字典使用哪类数据结构？", ["list", "tuple", "set", "dict"], "dict"),
    ("Python 中遍历序列通常使用哪个关键字？", ["loop", "for", "while", "each"], "for"),
    ("Python 中逻辑“与”运算使用哪个关键字？", ["and", "or", "not", "xor"], "and"),
    ("Python 中导入模块使用哪个关键字？", ["include", "require", "import", "using"], "import"),
]

MEDICAL = [
    ("正常成年人的腋下体温大约是多少摄氏度？", ["35.0", "36.5", "37.8", "39.0"], "36.5"),
    ("成人安静时的正常心率范围通常是多少次每分钟？", ["40-50", "60-100", "110-140", "150-180"], "60-100"),
    ("人体进行气体交换的主要器官是什么？", ["肝脏", "肺", "胃", "肾脏"], "肺"),
    ("维生素 C 主要存在于哪类食物中？", ["肉类", "新鲜果蔬", "油炸食品", "奶制品"], "新鲜果蔬"),
    ("人体最大的器官是什么？", ["心脏", "肝脏", "皮肤", "肺"], "皮肤"),
    ("成年人血压测量中，收缩压的正常范围通常是多少 mmHg？", ["90-120", "50-70", "140-160", "180-200"], "90-120"),
    ("人体内负责运输氧气的主要细胞是什么？", ["白细胞", "红细胞", "血小板", "淋巴细胞"], "红细胞"),
    ("糖尿病患者通常需要控制哪种物质的摄入？", ["蛋白质", "糖类", "钙", "铁"], "糖类"),
    ("人体骨骼中，成年人共有约多少块骨？", ["126", "206", "306", "406"], "206"),
    ("哪种维生素与钙吸收密切相关？", ["维生素 A", "维生素 B", "维生素 C", "维生素 D"], "维生素 D"),
]

GENERAL = [
    ("地球绕太阳公转一周大约需要多长时间？", ["一天", "一个月", "一年", "十年"], "一年"),
    ("一年中哪个月份至少有 28 天？", ["一月", "二月", "三月", "每个月"], "每个月"),
    ("水的化学式是什么？", ["CO2", "H2O", "O2", "NaCl"], "H2O"),
    ("下列哪个国家位于亚洲？", ["巴西", "中国", "埃及", "澳大利亚"], "中国"),
    ("太阳从哪个方向升起？", ["西边", "南边", "东边", "北边"], "东边"),
    ("一周通常有多少天？", ["5", "6", "7", "8"], "7"),
    ("1 小时等于多少分钟？", ["30", "60", "100", "120"], "60"),
    ("三角形内角和是多少度？", ["90", "180", "270", "360"], "180"),
    ("世界上面积最大的海洋是哪一个？", ["大西洋", "印度洋", "太平洋", "北冰洋"], "太平洋"),
    ("“GDP”通常用于衡量什么？", ["人口数量", "国土面积", "国内生产总值", "空气质量"], "国内生产总值"),
]


def generate_questions():
    rng = random.Random(20261119)
    questions = []
    for idx in range(1, 11):
        questions.append(build_math_question(rng, idx))
    questions.extend(build_fixed_questions("coding", CODING))
    questions.extend(build_fixed_questions("medical", MEDICAL))
    questions.extend(build_fixed_questions("general", GENERAL))
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    return questions


@dataclass
class ModelCard:
    name: str
    specialty: str
    reliability: float
    cross_domain: float = 0.40


@dataclass
class AgentResponse:
    agent: str
    answer: int
    confidence: float


class MockAgent:
    def __init__(self, card: ModelCard):
        self.card = card

    def relevance(self, domain):
        if self.card.specialty == domain:
            return 1.0
        if self.card.specialty == "general":
            return 0.62
        if self.card.specialty == "reasoning":
            return 0.70
        if self.card.specialty == "none":
            return 0.15
        return self.card.cross_domain

    def answer(self, question, rng):
        domain = question["domain"]
        p_correct = self._correct_probability(domain)
        p_correct = min(0.92, max(0.10, p_correct))
        correct_idx = question["answer"]
        if rng.random() < p_correct:
            answer = correct_idx
        else:
            wrong = [i for i in range(len(question["options"])) if i != correct_idx]
            answer = rng.choice(wrong)
        confidence = min(0.98, max(0.05, p_correct + rng.uniform(-0.08, 0.08)))
        return AgentResponse(self.card.name, answer, confidence)

    def _correct_probability(self, domain):
        if self.card.specialty == domain:
            return self.card.reliability + 0.12
        if self.card.specialty == "general":
            return 0.56
        if self.card.specialty == "reasoning":
            return 0.58 if domain != "medical" else 0.42
        if self.card.specialty == "none":
            return 0.22
        return self.card.cross_domain


def build_agents():
    cards = [
        ModelCard("MathAgent", "math", 0.82),
        ModelCard("CodeAgent", "coding", 0.82),
        ModelCard("MedAgent", "medical", 0.82),
        ModelCard("GeneralAgent", "general", 0.58),
        ModelCard("ReasonAgent", "reasoning", 0.66),
        ModelCard("NoisyAgent", "none", 0.20),
    ]
    return [MockAgent(card) for card in cards]


@dataclass
class SelectedAgent:
    agent: MockAgent
    response: AgentResponse
    relevance: float


def node_sampling(agents, question, responses, k=3):
    scored = []
    for agent in agents:
        relevance = agent.relevance(question["domain"])
        score = relevance + 0.05 * agent.card.reliability
        scored.append((score, agent))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, agent in scored[:k]:
        selected.append(SelectedAgent(agent, responses[agent.card.name], score))
    return selected


def message_passing(selected):
    ordered = sorted(
        selected,
        key=lambda x: (x.relevance, x.response.confidence),
        reverse=True,
    )
    responses = [item.response for item in ordered]

    for _ in range(2):
        for i in range(1, len(responses)):
            upstream = responses[i - 1]
            current = responses[i]
            if upstream.confidence > current.confidence + 0.04:
                current.answer = upstream.answer
                current.confidence = min(0.98, upstream.confidence * 0.92)

        for i in range(len(responses) - 2, -1, -1):
            downstream = responses[i + 1]
            current = responses[i]
            if downstream.confidence > current.confidence:
                current.answer = downstream.answer
                current.confidence = min(0.98, downstream.confidence * 0.95)

    return responses


def graph_pooling(responses, mode="max"):
    if mode == "max":
        return max(responses, key=lambda r: r.confidence).answer
    votes = {}
    for response in responses:
        votes[response.answer] = votes.get(response.answer, 0.0) + response.confidence
    return max(votes.items(), key=lambda item: item[1])[0]


def run_goa(agents, question, responses, k=3, pool_mode="max"):
    selected = node_sampling(agents, question, responses, k=k)
    updated = message_passing(selected)
    return graph_pooling(updated, mode=pool_mode)


def weighted_vote(responses):
    votes = {}
    for response in responses:
        votes[response.answer] = votes.get(response.answer, 0.0) + response.confidence
    return max(votes.items(), key=lambda item: item[1])[0]


def single_best(agents, question, responses):
    best_agent = max(agents, key=lambda a: a.relevance(question["domain"]))
    return responses[best_agent.card.name].answer


def all_six_vote(agents, question, responses):
    return weighted_vote([responses[agent.card.name] for agent in agents])


def random_three_vote(agents, question, responses, rng):
    chosen = rng.sample(agents, 3)
    return weighted_vote([responses[agent.card.name] for agent in chosen])


def seed_for(question_id):
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def accuracy(predictions):
    return sum(p == t for p, t in predictions) / len(predictions)


def macro_f1(predictions):
    classes = sorted({t for _, t in predictions})
    f1s = []
    for c in classes:
        tp = sum(1 for p, t in predictions if p == c and t == c)
        fp = sum(1 for p, t in predictions if p == c and t != c)
        fn = sum(1 for p, t in predictions if p != c and t == c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def plot_comparison(metrics, output_path):
    width, height = 900, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 28)
    title_font = ImageFont.truetype(FONT_PATH, 34)

    methods = list(metrics.keys())
    accs = [metrics[m]["accuracy"] for m in methods]
    max_acc = max(accs) if accs else 1.0
    area = (160, 100, 820, 420)
    gap = (area[2] - area[0]) / len(methods)
    bar_width = gap * 0.58

    draw.text((40, 30), "方法准确率对比", font=title_font, fill=(0, 0, 0))
    for i, method in enumerate(methods):
        acc = accs[i]
        bar_h = int((acc / max_acc) * (area[3] - area[1]))
        x1 = int(area[0] + gap * i + (gap - bar_width) / 2)
        y1 = int(area[3] - bar_h)
        x2 = int(x1 + bar_width)
        y2 = int(area[3])
        color = (234, 67, 53) if method == "goa_max" else (66, 133, 244)
        draw.rectangle([x1, y1, x2, y2], fill=color)
        draw.text((int(x1 + bar_width / 2 - 20), y1 - 36), f"{acc:.3f}", font=font, fill=(0, 0, 0))
        label = method.replace("_", " ")
        label_width = draw.textbbox((0, 0), label, font=font)[2]
        draw.text((int(x1 + bar_width / 2 - label_width / 2), area[3] + 8), label, font=font, fill=(0, 0, 0))

    img.save(output_path)


def main():
    if DATA_PATH.exists():
        questions = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    else:
        questions = generate_questions()

    agents = build_agents()
    method_names = ["single_best", "all_six", "random_three", "goa_max", "goa_mean"]
    predictions_by_method = {name: [] for name in method_names}
    rows = []

    for question in questions:
        rng = random.Random(seed_for(question["id"]))
        responses = {agent.card.name: agent.answer(question, rng) for agent in agents}

        predictions = {
            "single_best": single_best(agents, question, responses),
            "all_six": all_six_vote(agents, question, responses),
            "random_three": random_three_vote(agents, question, responses, rng),
            "goa_max": run_goa(agents, question, responses, k=3, pool_mode="max"),
            "goa_mean": run_goa(agents, question, responses, k=3, pool_mode="mean"),
        }

        for name in method_names:
            predictions_by_method[name].append((predictions[name], question["answer"]))

        rows.append({"id": question["id"], "domain": question["domain"], "true": question["answer"], **predictions})

    metrics = {}
    for name in method_names:
        metrics[name] = {
            "accuracy": round(accuracy(predictions_by_method[name]), 4),
            "macro_f1": round(macro_f1(predictions_by_method[name]), 4),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_header = ["id", "domain", "true"] + method_names
    csv_lines = [",".join(csv_header)]
    for row in rows:
        csv_lines.append(",".join([row["id"], row["domain"], str(row["true"])] + [str(row[name]) for name in method_names]))
    (RESULTS_DIR / "results.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    plot_comparison(metrics, RESULTS_DIR / "comparison.png")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
