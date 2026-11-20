import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .agents import build_agents
from .baselines import all_six_vote, random_three_vote, single_best
from .generate_data import DATA_PATH, main as generate_data
from .goa import run_goa


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


def _seed_for(question_id):
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _accuracy(predictions):
    return sum(p == t for p, t in predictions) / len(predictions)


def _macro_f1(predictions):
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


def _plot(metrics, output_path):
    width, height = 900, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 28)
    title_font = ImageFont.truetype(FONT_PATH, 34)

    methods = list(metrics.keys())
    accs = [metrics[m]["accuracy"] for m in methods]
    max_acc = max(accs) if accs else 1.0
    bar_area_x1, bar_area_y1, bar_area_x2, bar_area_y2 = 160, 100, 820, 420
    bar_width = (bar_area_x2 - bar_area_x1) / len(methods) * 0.58
    gap = (bar_area_x2 - bar_area_x1) / len(methods)

    draw.text((40, 30), "方法准确率对比", font=title_font, fill=(0, 0, 0))
    for i, method in enumerate(methods):
        acc = accs[i]
        bar_h = int((acc / max_acc) * (bar_area_y2 - bar_area_y1))
        x1 = int(bar_area_x1 + gap * i + (gap - bar_width) / 2)
        y1 = int(bar_area_y2 - bar_h)
        x2 = int(x1 + bar_width)
        y2 = int(bar_area_y2)
        color = (66, 133, 244) if method != "goa_max" else (234, 67, 53)
        draw.rectangle([x1, y1, x2, y2], fill=color)
        draw.text((int(x1 + bar_width / 2 - 20), y1 - 36), f"{acc:.3f}", font=font, fill=(0, 0, 0))

        label = method.replace("_", " ")
        label_width = draw.textbbox((0, 0), label, font=font)[2]
        draw.text((int(x1 + bar_width / 2 - label_width / 2), bar_area_y2 + 8), label, font=font, fill=(0, 0, 0))

    img.save(output_path)


def main():
    if not DATA_PATH.exists():
        generate_data()

    questions = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    agents = build_agents()

    method_names = ["single_best", "all_six", "random_three", "goa_max", "goa_mean"]
    predictions_by_method = {name: [] for name in method_names}
    rows = []

    for question in questions:
        rng = random.Random(_seed_for(question["id"]))
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

        rows.append(
            {
                "id": question["id"],
                "domain": question["domain"],
                "true": question["answer"],
                **predictions,
            }
        )

    metrics = {}
    for name in method_names:
        metrics[name] = {
            "accuracy": round(_accuracy(predictions_by_method[name]), 4),
            "macro_f1": round(_macro_f1(predictions_by_method[name]), 4),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_header = ["id", "domain", "true"] + method_names
    csv_lines = [",".join(csv_header)]
    for row in rows:
        csv_lines.append(
            ",".join(
                [row["id"], row["domain"], str(row["true"])]
                + [str(row[name]) for name in method_names]
            )
        )
    (RESULTS_DIR / "results.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    _plot(metrics, RESULTS_DIR / "comparison.png")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
