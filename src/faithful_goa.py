import csv
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def _find_project_root():
    expected_questions = 2_000
    candidates = []
    env_root = os.environ.get("GOV_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    file_value = globals().get("__file__") or globals().get("__vsc_ipynb_file__")
    if file_value:
        file_dir = Path(file_value).resolve().parent
        candidates.extend([file_dir, file_dir.parent])
    current_dir = Path.cwd().resolve()
    candidates.extend([current_dir, *current_dir.parents])
    home_dir = Path.home()
    profile_candidates = [home_dir]
    for drive in ("C:", "D:", "E:", "F:"):
        profile_dir = Path(f"{drive}/Users/{home_dir.name}")
        if profile_dir.exists():
            profile_candidates.append(profile_dir)
    for profile_dir in profile_candidates:
        candidates.append(
            profile_dir
            / "Desktop"
            / "博士在读"
            / "博士课程作业"
            / "前沿作业"
            / "任务2结果"
        )

    expanded_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        expanded_candidates.append(candidate)
        try:
            for child in candidate.iterdir():
                if child.is_dir() and child not in seen:
                    seen.add(child)
                    expanded_candidates.append(child)
        except OSError:
            continue

    for candidate in expanded_candidates:
        data_path = candidate / "data" / "questions.json"
        if not data_path.exists():
            continue
        try:
            question_count = len(json.loads(data_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
        if question_count == expected_questions:
            return candidate

    search_roots = [current_dir]
    for profile_dir in profile_candidates:
        search_roots.extend(
            [
                profile_dir / "Desktop",
                profile_dir / "Documents",
                profile_dir / "OneDrive" / "Desktop",
            ]
        )
    skipped_dirs = {
        "AppData",
        ".cache",
        ".conda",
        ".git",
        ".venv",
        "site-packages",
        "node_modules",
        "venv",
    }
    visited_roots = set()
    for search_root in search_roots:
        if search_root in visited_roots or not search_root.exists():
            continue
        visited_roots.add(search_root)
        base_depth = len(search_root.parts)
        for dir_path, dir_names, file_names in os.walk(search_root):
            current_path = Path(dir_path)
            if len(current_path.parts) - base_depth >= 6:
                dir_names[:] = []
                continue
            dir_names[:] = [
                name
                for name in dir_names
                if name not in skipped_dirs and not name.startswith(".")
            ]
            if current_path.name != "data" or "questions.json" not in file_names:
                continue
            data_path = current_path / "questions.json"
            try:
                question_count = len(
                    json.loads(data_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError):
                continue
            if question_count == expected_questions:
                return current_path.parent

    for candidate in candidates:
        if (candidate / "data" / "questions.json").exists():
            return candidate
        task_dir = candidate / "任务2结果"
        if (task_dir / "data" / "questions.json").exists():
            return task_dir
    return current_dir


ROOT = _find_project_root()
DATA_PATH = ROOT / "data" / "questions.json"
TOTAL_QUESTIONS = 2_000
RESULTS_DIR = ROOT / "results"
CACHE_PATH = RESULTS_DIR / "high_fidelity_cache.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
ANALYSIS_PATH = RESULTS_DIR / "analysis.json"
RESULTS_PATH = RESULTS_DIR / "results.csv"
PLOT_PATH = RESULTS_DIR / "comparison.png"
MANIFEST_PATH = RESULTS_DIR / "experiment_manifest.json"
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

PROMPT_VERSION = "faithful-goa-v1"
META_MODEL = "deepseek-v4-pro"
TOP_K = 3
TAU = 0.05
LETTERS = ("A", "B", "C", "D")
WORKERS = int(os.environ.get("GOV_FAITHFUL_WORKERS", "16"))
METHOD_NAMES = (
    "single_best",
    "all_six",
    "random_three",
    "goa_max",
    "goa_mean",
)
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class AgentResponse:
    answer: int
    confidence: float


def _weighted_vote(responses):
    votes = {}
    for response in responses:
        votes[response.answer] = votes.get(response.answer, 0.0) + response.confidence
    return max(votes.items(), key=lambda item: item[1])[0]


def _accuracy(predictions):
    return sum(predicted == truth for predicted, truth in predictions) / len(predictions)


def _macro_f1(predictions):
    classes = sorted({truth for _, truth in predictions})
    f1_scores = []
    for label in classes:
        true_positive = sum(
            1 for predicted, truth in predictions if predicted == label and truth == label
        )
        false_positive = sum(
            1 for predicted, truth in predictions if predicted == label and truth != label
        )
        false_negative = sum(
            1 for predicted, truth in predictions if predicted != label and truth == label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def _plot(metrics, output_path):
    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 28)
    title_font = ImageFont.truetype(FONT_PATH, 34)

    methods = list(metrics)
    accuracies = [metrics[name]["accuracy"] for name in methods]
    max_accuracy = max(accuracies) if accuracies else 1.0
    left, top, right, bottom = 160, 100, 820, 420
    gap = (right - left) / len(methods)
    bar_width = gap * 0.58

    draw.text((40, 30), "方法准确率对比", font=title_font, fill=(0, 0, 0))
    for index, method in enumerate(methods):
        accuracy = accuracies[index]
        bar_height = int((accuracy / max_accuracy) * (bottom - top))
        x1 = int(left + gap * index + (gap - bar_width) / 2)
        y1 = int(bottom - bar_height)
        x2 = int(x1 + bar_width)
        color = (234, 67, 53) if method == "goa_mean" else (66, 133, 244)
        draw.rectangle([x1, y1, x2, bottom], fill=color)
        draw.text(
            (int(x1 + bar_width / 2 - 20), y1 - 36),
            f"{accuracy:.3f}",
            font=font,
            fill=(0, 0, 0),
        )
        label = method.replace("_", " ")
        label_width = draw.textbbox((0, 0), label, font=font)[2]
        draw.text(
            (int(x1 + bar_width / 2 - label_width / 2), bottom + 8),
            label,
            font=font,
            fill=(0, 0, 0),
        )
    image.save(output_path)

AGENT_MODEL_MAP = {
    "MathAgent": {
        "model": "deepseek-reasoner",
        "domain": "mathematics",
        "task": "mathematical reasoning and calculation",
        "size": "reasoning-oriented model",
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    "CodeAgent": {
        "model": "deepseek-coder",
        "domain": "coding and computer science",
        "task": "programming, algorithms, and software questions",
        "size": "coding-oriented model",
        "temperature": 0.2,
        "max_tokens": 512,
    },
    "MedAgent": {
        "model": "deepseek-v4-pro[1m]",
        "domain": "biomedical and clinical medicine",
        "task": "medical examination question answering",
        "size": "long-context general model",
        "temperature": 0.2,
        "max_tokens": 512,
    },
    "GeneralAgent": {
        "model": "deepseek-chat",
        "domain": "general knowledge",
        "task": "broad knowledge and natural language understanding",
        "size": "general-purpose model",
        "temperature": 0.4,
        "max_tokens": 512,
    },
    "ReasonAgent": {
        "model": "deepseek-v4-pro",
        "domain": "cross-domain reasoning",
        "task": "multi-step reasoning and option elimination",
        "size": "reasoning-oriented model",
        "temperature": 0.3,
        "max_tokens": 512,
    },
    "NoisyAgent": {
        "model": "deepseek-v4-flash",
        "domain": "general knowledge",
        "task": "fast but less reliable question answering",
        "size": "compact fast model",
        "temperature": 1.0,
        "max_tokens": 512,
    },
}

AGENT_ROLES = {
    "MathAgent": (
        "You are a mathematics expert. Solve the problem carefully, "
        "check numerical operations, and choose the best option."
    ),
    "CodeAgent": (
        "You are a computer science expert. Use programming, algorithms, "
        "computer architecture, and software knowledge to choose the answer."
    ),
    "MedAgent": (
        "You are a medical knowledge expert answering a written examination "
        "question. Identify the most accurate option and avoid speculation."
    ),
    "GeneralAgent": (
        "You are a broad general-knowledge assistant. Compare all options "
        "and select the one best supported by established knowledge."
    ),
    "ReasonAgent": (
        "You are a careful cross-domain reasoner. Eliminate inconsistent "
        "options and select the answer that follows from the evidence."
    ),
    "NoisyAgent": (
        "You are a fast, less reliable generalist. Still read the question "
        "carefully and choose your best guess."
    ),
}


def _load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    )


def _save_cache(cache):
    temp_path = CACHE_PATH.with_suffix(".json.tmp")
    _write_json(temp_path, cache)
    temp_path.replace(CACHE_PATH)


def _cache_key(stage, question_id, names=()):
    suffix = "|".join(names)
    return f"{PROMPT_VERSION}|{stage}|{META_MODEL}|{question_id}|{suffix}"


def _extract_json(content):
    decoder = json.JSONDecoder()
    candidates = []
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            candidates.append(candidate)
    for expected_key in ("answer", "indices", "scores"):
        matching = [item for item in candidates if expected_key in item]
        if matching:
            return matching[-1]
    if not candidates:
        raise ValueError(f"invalid JSON response: {content[:300]}")
    return max(candidates, key=len)


def _deepseek_json(model, system_prompt, user_prompt, temperature, max_tokens):
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if not token or not base_url:
        raise RuntimeError("ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL are required")

    last_error = None
    for attempt in range(6):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": token,
                    "Authorization": f"Bearer {token}",
                    "anthropic-version": ANTHROPIC_VERSION,
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "thinking": {"type": "disabled"},
                },
                timeout=120,
            )
            if response.status_code in {429, 500, 502, 503, 504, 529}:
                raise requests.HTTPError(
                    f"retryable status {response.status_code}",
                    response=response,
                )
            response.raise_for_status()
            body = response.json()
            content = "".join(
                block.get("text", "")
                for block in body.get("content", [])
                if block.get("type") == "text"
            )
            payload = _extract_json(content)
            usage = body.get("usage", {})
            return {
                "payload": payload,
                "model": body.get("model", model),
                "prompt_tokens": int(usage.get("input_tokens", 0)),
                "completion_tokens": int(usage.get("output_tokens", 0)),
            }
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"model call failed: {model}") from last_error


def _question_block(question):
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(LETTERS, question["options"])
    )
    return f"Question:\n{question['stem']}\n\nOptions:\n{options}"


def _normalize_answer(payload, question):
    answer = str(payload.get("answer", "")).strip().upper()
    if answer in LETTERS:
        return LETTERS.index(answer)
    for index, option in enumerate(question["options"]):
        if answer == option.strip().upper():
            return index
    match = re.match(r"^([ABCD])[\).:\-\s]", answer)
    if match:
        return LETTERS.index(match.group(1))
    raise ValueError(f"invalid answer: {answer}")


def _normalize_response(entry, question):
    payload = entry["payload"]
    return {
        "answer": _normalize_answer(payload, question),
        "confidence": min(
            0.99,
            max(0.01, float(payload.get("confidence", 0.5))),
        ),
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "model": entry["model"],
    }


def _validate_answer_entry(entry):
    payload = entry.get("payload", {})
    answer = str(payload.get("answer", "")).strip().upper() if isinstance(payload, dict) else ""
    if answer not in LETTERS:
        raise ValueError(f"response has an invalid answer: {answer}")
    return entry


def _sanitize_cache(cache):
    removed = 0
    for key in list(cache):
        stage = key.split("|")[1]
        payload = cache[key].get("payload", {})
        valid = True
        if stage in {"initial", "s2t", "t2s", "pool"}:
            valid = str(payload.get("answer", "")).strip().upper() in LETTERS
        elif stage == "node":
            valid = isinstance(payload.get("indices"), list)
        elif stage == "edge":
            valid = isinstance(payload.get("scores"), list)
        if not valid:
            del cache[key]
            removed += 1
    if removed:
        _save_cache(cache)
        print(f"removed {removed} invalid cached entries", flush=True)


def _model_description(name):
    info = AGENT_MODEL_MAP[name]
    return (
        f"[{name}] domain={info['domain']}; task={info['task']}; "
        f"size={info['size']}; model={info['model']}"
    )


def _node_prompt(question):
    model_lines = "\n".join(
        f"{index}. {_model_description(name)}"
        for index, name in enumerate(AGENT_MODEL_MAP)
    )
    return (
        f"Select exactly {TOP_K} models best suited for this question.\n\n"
        f"{_question_block(question)}\n\n"
        f"Available models:\n{model_lines}\n\n"
        "Selection criteria, in priority order:\n"
        "1. Domain match\n"
        "2. Task specialization\n"
        "3. Include at least one generalist when useful\n"
        "4. Prefer larger or stronger models only when the difference matters\n\n"
        'Return only JSON in the form {"indices": [0, 1, 2]}. '
        "Use exactly three distinct zero-based indices."
    )


def _initial_prompt(question, name):
    return (
        f"{_question_block(question)}\n\n"
        "Provide brief reasoning in 2-3 sentences, then choose the best option. "
        'Return only JSON in the form {"reasoning": "brief reasoning", '
        '"answer": "A", "confidence": 0.0}. '
        "The answer must be one of A, B, C, D."
    )


def _response_text(response):
    answer = LETTERS[response["answer"]]
    reasoning = response["reasoning"] or "No reasoning provided."
    return f"Answer: {answer}\nReasoning: {reasoning}"


def _edge_prompt(question, judge_name, other_names, responses):
    response_lines = []
    for name in other_names:
        response_lines.append(f"[{name}]\n{_response_text(responses[name])}")
    return (
        f"Score the following {len(other_names)} model responses to the question.\n\n"
        f"{_question_block(question)}\n\n"
        "Responses:\n"
        + "\n\n".join(response_lines)
        + "\n\n"
        "Assign a score based on correctness, coherence, and relevance. "
        "Scores must be non-negative and sum to exactly 1.0. "
        'Return only JSON in the form {"scores": [0.5, 0.5]}.'
    )


def _relevance_label(weight):
    if weight > 0.7:
        return "high relevance"
    if weight > 0.4:
        return "moderate relevance"
    return "low relevance"


def _neighbor_lines(neighbors, responses, weights):
    lines = []
    for name, weight in sorted(
        zip(neighbors, weights),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(
            f"[{name}] ({_relevance_label(weight)}, weight={weight:.3f})\n"
            f"{_response_text(responses[name])}"
        )
    return "\n\n".join(lines)


def _source_to_target_prompt(question, target_name, own_response, neighbors, responses, weights):
    return (
        "Refine your answer by considering other models' responses.\n\n"
        f"{_question_block(question)}\n\n"
        f"Your initial response:\n{_response_text(own_response)}\n\n"
        f"Other models' responses, ranked by relevance:\n"
        f"{_neighbor_lines(neighbors, responses, weights)}\n\n"
        "Integrate useful insights, but be critical because some responses may be wrong. "
        'Return only JSON in the form {"reasoning": "brief reasoning", '
        '"answer": "A", "confidence": 0.0}.'
    )


def _target_to_source_prompt(question, source_name, own_response, neighbors, responses, weights):
    return (
        "Other models refined their answers after seeing yours. "
        "Use their improvements to finalize your response.\n\n"
        f"{_question_block(question)}\n\n"
        f"Your initial response:\n{_response_text(own_response)}\n\n"
        f"Updated responses from other models:\n"
        f"{_neighbor_lines(neighbors, responses, weights)}\n\n"
        "Write your final response, incorporating valuable refinements. "
        "Be critical because some responses may be wrong. "
        'Return only JSON in the form {"reasoning": "brief reasoning", '
        '"answer": "A", "confidence": 0.0}.'
    )


def _pool_prompt(question, ordered_names, responses, relevance):
    response_lines = "\n\n".join(
        f"[{name}] (relevance={relevance[name]:.3f})\n{_response_text(responses[name])}"
        for name in ordered_names
    )
    return (
        "Synthesize these model responses into one final answer. "
        "Use the relevance scores as weights and be critical of incorrect content.\n\n"
        f"{_question_block(question)}\n\n"
        f"Model responses:\n{response_lines}\n\n"
        'Return only JSON in the form {"answer": "A", "confidence": 0.0}. '
        "The answer must be one of A, B, C, D."
    )


def _run_stage(tasks, callback, cache, stage_name):
    missing = [task for task in tasks if task["key"] not in cache]
    print(f"{stage_name}: {len(missing)}/{len(tasks)} calls required", flush=True)
    if not missing:
        return

    errors = []
    completed = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(callback, task): task
            for task in missing
        }
        for future in as_completed(futures):
            task = futures[future]
            completed += 1
            try:
                cache[task["key"]] = future.result()
            except Exception as exc:
                errors.append(f"{task['key']}: {exc}")
            if completed % 50 == 0 or completed == len(missing):
                _save_cache(cache)
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"{stage_name}: {completed}/{len(missing)} "
                    f"({rate:.2f} calls/s, {len(errors)} errors)",
                    flush=True,
                )
    if errors:
        raise RuntimeError(
            f"{stage_name} has {len(errors)} failed calls; rerun to retry. "
            f"First error: {errors[0]}"
        )


def _initial_task(task):
    name = task["agent"]
    info = AGENT_MODEL_MAP[name]
    return _validate_answer_entry(_deepseek_json(
        info["model"],
        AGENT_ROLES[name],
        _initial_prompt(task["question"], name),
        info["temperature"],
        info["max_tokens"],
    ))


def _node_task(task):
    entry = _deepseek_json(
        META_MODEL,
        "You are an expert model-routing assistant.",
        _node_prompt(task["question"]),
        0.1,
        256,
    )
    if not isinstance(entry["payload"].get("indices"), list):
        raise ValueError("node selection is missing an indices list")
    return entry


def _edge_task(task):
    info = AGENT_MODEL_MAP[task["agent"]]
    entry = _deepseek_json(
        info["model"],
        AGENT_ROLES[task["agent"]],
        _edge_prompt(
            task["question"],
            task["agent"],
            task["other_names"],
            task["responses"],
        ),
        info["temperature"],
        256,
    )
    if not isinstance(entry["payload"].get("scores"), list):
        raise ValueError("edge scoring is missing a scores list")
    return entry


def _message_task(task):
    info = AGENT_MODEL_MAP[task["agent"]]
    return _validate_answer_entry(_deepseek_json(
        info["model"],
        AGENT_ROLES[task["agent"]],
        task["prompt"],
        info["temperature"],
        info["max_tokens"],
    ))


def _pool_task(task):
    return _validate_answer_entry(_deepseek_json(
        META_MODEL,
        "You are a careful meta-reasoner synthesizing multiple model answers.",
        task["prompt"],
        0.1,
        512,
    ))


def _fallback_nodes(question):
    domain = question["domain"]
    preferred = {
        "math": ["MathAgent", "ReasonAgent", "GeneralAgent"],
        "coding": ["CodeAgent", "ReasonAgent", "GeneralAgent"],
        "medical": ["MedAgent", "ReasonAgent", "GeneralAgent"],
        "general": ["GeneralAgent", "ReasonAgent", "NoisyAgent"],
    }[domain]
    return preferred[:TOP_K]


def _selected_agents(question, node_entry, cache):
    payload = node_entry["payload"]
    indices = payload.get("indices", [])
    names = list(AGENT_MODEL_MAP)
    selected = []
    if isinstance(indices, list):
        for value in indices:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(names) and names[index] not in selected:
                selected.append(names[index])
    if len(selected) != TOP_K:
        selected = _fallback_nodes(question)
    return selected


def _initial_responses(question, cache):
    responses = {}
    for name in AGENT_MODEL_MAP:
        key = _cache_key("initial", question["id"], (name,))
        responses[name] = _normalize_response(cache[key], question)
    return responses


def _edge_scores(question, selected, responses, cache):
    relevance = {name: 0.0 for name in selected}
    for judge in selected:
        key = _cache_key("edge", question["id"], (judge,))
        entry = cache[key]
        others = [name for name in selected if name != judge]
        scores = entry["payload"].get("scores", [])
        try:
            parsed = [max(0.0, float(score)) for score in scores]
        except (TypeError, ValueError):
            parsed = []
        if len(parsed) != len(others):
            parsed = [1.0 / len(others)] * len(others)
        total = sum(parsed)
        if total <= 0:
            parsed = [1.0 / len(others)] * len(others)
            total = 1.0
        for name, score in zip(others, parsed):
            relevance[name] += score / total
    return relevance


def _pipeline_tasks(questions, cache):
    initial_tasks = []
    for question in questions:
        for name in AGENT_MODEL_MAP:
            initial_tasks.append(
                {
                    "key": _cache_key("initial", question["id"], (name,)),
                    "question": question,
                    "agent": name,
                }
            )
    _run_stage(initial_tasks, _initial_task, cache, "initial responses")

    node_tasks = [
        {
            "key": _cache_key("node", question["id"]),
            "question": question,
        }
        for question in questions
    ]
    _run_stage(node_tasks, _node_task, cache, "node sampling")

    selected_by_question = {}
    edge_tasks = []
    for question in questions:
        selected = _selected_agents(
            question,
            cache[_cache_key("node", question["id"])],
            cache,
        )
        selected_by_question[question["id"]] = selected
        responses = _initial_responses(question, cache)
        for judge in selected:
            others = [name for name in selected if name != judge]
            edge_tasks.append(
                {
                    "key": _cache_key("edge", question["id"], (judge,)),
                    "question": question,
                    "agent": judge,
                    "other_names": others,
                    "responses": responses,
                }
            )
    _run_stage(edge_tasks, _edge_task, cache, "edge sampling")
    return selected_by_question


def _run_message_passing(questions, selected_by_question, cache):
    state = {}
    for question in questions:
        selected = selected_by_question[question["id"]]
        responses = _initial_responses(question, cache)
        relevance = _edge_scores(question, selected, responses, cache)
        ordered = sorted(selected, key=lambda name: relevance[name], reverse=True)
        ordered = [name for name in ordered if relevance[name] >= TAU] or ordered[:TOP_K]
        state[question["id"]] = {
            "ordered": ordered,
            "initial": responses,
            "relevance": relevance,
        }

    s2t_tasks = []
    for question in questions:
        item = state[question["id"]]
        ordered = item["ordered"]
        relevance = item["relevance"]
        for target_index in range(1, len(ordered)):
            target = ordered[target_index]
            neighbors = ordered[:target_index]
            denom = sum(relevance[name] for name in neighbors) or 1.0
            weights = [relevance[name] / denom for name in neighbors]
            s2t_tasks.append(
                {
                    "key": _cache_key("s2t", question["id"], (target,)),
                    "question": question,
                    "agent": target,
                    "prompt": _source_to_target_prompt(
                        question,
                        target,
                        item["initial"][target],
                        neighbors,
                        item["initial"],
                        weights,
                    ),
                }
            )
    _run_stage(s2t_tasks, _message_task, cache, "source-to-target")

    s2t_responses = {}
    for question in questions:
        item = state[question["id"]]
        updated = dict(item["initial"])
        for target in item["ordered"][1:]:
            key = _cache_key("s2t", question["id"], (target,))
            updated[target] = _normalize_response(cache[key], question)
        s2t_responses[question["id"]] = updated

    t2s_tasks = []
    for question in questions:
        item = state[question["id"]]
        ordered = item["ordered"]
        relevance = item["relevance"]
        updated = s2t_responses[question["id"]]
        for source_index in range(0, len(ordered) - 1):
            source = ordered[source_index]
            neighbors = ordered[source_index + 1 :]
            denom = sum(relevance[name] for name in neighbors) or 1.0
            weights = [relevance[name] / denom for name in neighbors]
            t2s_tasks.append(
                {
                    "key": _cache_key("t2s", question["id"], (source,)),
                    "question": question,
                    "agent": source,
                    "prompt": _target_to_source_prompt(
                        question,
                        source,
                        item["initial"][source],
                        neighbors,
                        updated,
                        weights,
                    ),
                }
            )
    _run_stage(t2s_tasks, _message_task, cache, "target-to-source")

    final_responses = {}
    for question in questions:
        item = state[question["id"]]
        updated = dict(s2t_responses[question["id"]])
        for source in item["ordered"][:-1]:
            key = _cache_key("t2s", question["id"], (source,))
            updated[source] = _normalize_response(cache[key], question)
        final_responses[question["id"]] = updated
    return state, final_responses


def _run_pooling(questions, state, final_responses, cache):
    pool_tasks = []
    for question in questions:
        item = state[question["id"]]
        pool_tasks.append(
            {
                "key": _cache_key("pool", question["id"]),
                "question": question,
                "prompt": _pool_prompt(
                    question,
                    item["ordered"],
                    final_responses[question["id"]],
                    item["relevance"],
                ),
            }
        )
    _run_stage(pool_tasks, _pool_task, cache, "graph pooling")


def _summarize_question(question, selected, state, final_responses, cache):
    initial = state[question["id"]]["initial"]
    ordered = state[question["id"]]["ordered"]
    relevance = state[question["id"]]["relevance"]
    initial_responses = {
        name: AgentResponse(response["answer"], response["confidence"])
        for name, response in initial.items()
    }
    domain_expert = {
        "math": "MathAgent",
        "coding": "CodeAgent",
        "medical": "MedAgent",
        "general": "GeneralAgent",
    }[question["domain"]]
    predicted = {
        "single_best": initial_responses[domain_expert].answer,
        "all_six": _weighted_vote(list(initial_responses.values())),
    }
    rng = random.Random(question["id"])
    chosen = rng.sample(list(initial_responses.values()), 3)
    predicted["random_three"] = _weighted_vote(chosen)
    predicted["goa_max"] = final_responses[question["id"]][ordered[0]]["answer"]
    pool_entry = cache[_cache_key("pool", question["id"])]
    predicted["goa_mean"] = _normalize_answer(pool_entry["payload"], question)
    return predicted, relevance


def _evaluate(questions, selected_by_question, state, final_responses, cache):
    predictions_by_method = {name: [] for name in METHOD_NAMES}
    rows = []
    per_domain = {}
    for question in questions:
        predicted, relevance = _summarize_question(
            question,
            selected_by_question[question["id"]],
            state,
            final_responses,
            cache,
        )
        for name in METHOD_NAMES:
            predictions_by_method[name].append((predicted[name], question["answer"]))
        rows.append(
            {
                "id": question["id"],
                "domain": question["domain"],
                "source": question["source"],
                "true": question["answer"],
                "selected_agents": "|".join(selected_by_question[question["id"]]),
                "relevance": json.dumps(relevance, ensure_ascii=False),
                **predicted,
            }
        )

    metrics = {
        name: {
            "accuracy": round(_accuracy(predictions_by_method[name]), 4),
            "macro_f1": round(_macro_f1(predictions_by_method[name]), 4),
        }
        for name in METHOD_NAMES
    }
    for domain in ("general", "coding", "medical", "math"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        if not domain_rows:
            continue
        per_domain[domain] = {
            name: round(
                sum(row[name] == row["true"] for row in domain_rows) / len(domain_rows),
                4,
            )
            for name in METHOD_NAMES
        }
    return metrics, rows, per_domain


def _write_outputs(metrics, rows, per_domain, cache):
    METRICS_PATH.write_bytes(
        json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8")
    )
    _write_json(ANALYSIS_PATH, {"by_domain_accuracy": per_domain})
    _plot(metrics, PLOT_PATH)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "id",
                "domain",
                "source",
                "true",
                "selected_agents",
                "relevance",
                *METHOD_NAMES,
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    prompt_tokens = sum(item.get("prompt_tokens", 0) for item in cache.values())
    completion_tokens = sum(item.get("completion_tokens", 0) for item in cache.values())
    _write_json(
        MANIFEST_PATH,
        {
            "prompt_version": PROMPT_VERSION,
            "questions": len(rows),
            "calls": len(cache),
            "top_k": TOP_K,
            "tau": TAU,
            "meta_model": META_MODEL,
            "model_map": AGENT_MODEL_MAP,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "metrics": metrics,
        },
    )


def main():
    questions = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if len(questions) != TOTAL_QUESTIONS:
        raise RuntimeError(
            f"expected {TOTAL_QUESTIONS} questions, got {len(questions)}; "
            f"resolved project root: {ROOT}; data path: {DATA_PATH}. "
            "Set GOV_PROJECT_ROOT to the 任务2结果 directory if needed."
        )

    cache = _load_json(CACHE_PATH, {})
    _sanitize_cache(cache)
    selected_by_question = _pipeline_tasks(questions, cache)
    state, final_responses = _run_message_passing(
        questions,
        selected_by_question,
        cache,
    )
    _run_pooling(questions, state, final_responses, cache)
    metrics, rows, per_domain = _evaluate(
        questions,
        selected_by_question,
        state,
        final_responses,
        cache,
    )
    _write_outputs(metrics, rows, per_domain, cache)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
