import hashlib
import json
import random
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "questions.json"
MANIFEST_PATH = ROOT / "data" / "dataset_manifest.json"

TOTAL_QUESTIONS = 2_000
SAMPLES_PER_SOURCE = 500
SAMPLES_PER_SUBJECT = 100
SEED = 20261125

HF_MIRROR = "https://hf-mirror.com"
USER_AGENT = "Graph-of-Agents-course-reproduction/1.0"

MMLU_SUBJECTS = (
    "philosophy",
    "high_school_world_history",
    "high_school_geography",
    "global_facts",
    "sociology",
)

CEVAL_SUBJECTS = (
    "college_programming",
    "computer_architecture",
    "computer_network",
    "operating_system",
    "discrete_mathematics",
)

MMLU_DATASET = "cais/mmlu"
CEVAL_DATASET = "ceval/ceval-exam"
MEDQA_DATASET = "GBaker/MedQA-USMLE-4-options"
GSM8K_ARCHIVE_URL = (
    "https://codeload.github.com/openai/grade-school-math/zip/refs/heads/master"
)


def _download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                with target.open("wb") as output:
                    shutil.copyfileobj(response, output)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == 4:
                raise RuntimeError(f"failed to download {url}") from exc
            time.sleep(2**attempt)


def _read_parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to rebuild the public dataset: "
            "python -m pip install pyarrow"
        ) from exc
    return pq.read_table(path).to_pylist()


def _shuffle_question(record, options, correct_index):
    indexed_options = list(enumerate(options))
    rng = random.Random(f"{SEED}:{record['id']}")
    rng.shuffle(indexed_options)
    shuffled_options = [option for _, option in indexed_options]
    shuffled_answer = next(
        index
        for index, (source_index, _) in enumerate(indexed_options)
        if source_index == correct_index
    )
    record["options"] = shuffled_options
    record["answer"] = shuffled_answer
    return record


def _download_mmlu_subject(subject, cache_dir):
    url = (
        f"{HF_MIRROR}/datasets/{MMLU_DATASET}/resolve/main/"
        f"{subject}/test-00000-of-00001.parquet?download=true"
    )
    target = cache_dir / f"mmlu-{subject}.parquet"
    _download(url, target)
    return url, _read_parquet(target)


def _download_ceval_subject(subject, cache_dir):
    url = (
        f"{HF_MIRROR}/datasets/{CEVAL_DATASET}/resolve/main/"
        f"{subject}/test-00000-of-00001.parquet?download=true"
    )
    target = cache_dir / f"ceval-{subject}.parquet"
    _download(url, target)
    return url, _read_parquet(target)


def _sample_mmlu(cache_dir):
    questions = []
    source_rows = {}
    urls = []
    seen_stems = set()
    for subject in MMLU_SUBJECTS:
        url, rows = _download_mmlu_subject(subject, cache_dir)
        urls.append(url)
        valid_rows = []
        for row_index, row in enumerate(rows):
            stem = row["question"].strip()
            if (
                stem
                and stem not in seen_stems
                and len(row["choices"]) == 4
                and len({choice.strip() for choice in row["choices"]}) == 4
                and 0 <= int(row["answer"]) < 4
            ):
                seen_stems.add(stem)
                valid_rows.append((row_index, row))
        if len(valid_rows) < SAMPLES_PER_SUBJECT:
            raise RuntimeError(
                f"MMLU/{subject} has only {len(valid_rows)} valid test rows"
            )
        source_rows[subject] = len(rows)
        rng = random.Random(f"{SEED}:mmlu:{subject}")
        for source_index, row in rng.sample(valid_rows, SAMPLES_PER_SUBJECT):
            question = {
                "id": f"mmlu-{subject}-{source_index}",
                "source": "MMLU",
                "source_subject": subject,
                "split": "test",
                "domain": "general",
                "stem": row["question"].strip(),
            }
            questions.append(
                _shuffle_question(
                    question,
                    [choice.strip() for choice in row["choices"]],
                    int(row["answer"]),
                )
            )
    return questions, source_rows, urls


def _sample_ceval(cache_dir):
    questions = []
    source_rows = {}
    urls = []
    seen_stems = set()
    letters = ("A", "B", "C", "D")
    position = {letter: index for index, letter in enumerate(letters)}
    for subject in CEVAL_SUBJECTS:
        url, rows = _download_ceval_subject(subject, cache_dir)
        urls.append(url)
        valid_rows = []
        for row_index, row in enumerate(rows):
            stem = row["question"].strip()
            if (
                stem
                and stem not in seen_stems
                and all(str(row[letter]).strip() for letter in letters)
                and len({str(row[letter]).strip() for letter in letters}) == 4
                and str(row["answer"]).strip().upper() in position
            ):
                seen_stems.add(stem)
                valid_rows.append((row_index, row))
        if len(valid_rows) < SAMPLES_PER_SUBJECT:
            raise RuntimeError(
                f"C-Eval/{subject} has only {len(valid_rows)} valid test rows"
            )
        source_rows[subject] = len(rows)
        rng = random.Random(f"{SEED}:ceval:{subject}")
        for source_index, row in rng.sample(valid_rows, SAMPLES_PER_SUBJECT):
            answer_letter = str(row["answer"]).strip().upper()
            question = {
                "id": f"ceval-{subject}-{source_index}",
                "source": "C-Eval",
                "source_subject": subject,
                "split": "test",
                "domain": "coding",
                "stem": row["question"].strip(),
            }
            questions.append(
                _shuffle_question(
                    question,
                    [str(row[letter]).strip() for letter in letters],
                    position[answer_letter],
                )
            )
    return questions, source_rows, urls


def _sample_medqa(cache_dir):
    url = (
        f"{HF_MIRROR}/datasets/{MEDQA_DATASET}/resolve/main/"
        "phrases_no_exclude_test.jsonl?download=true"
    )
    target = cache_dir / "medqa-test.jsonl"
    _download(url, target)

    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid_rows = [
        (row_index, row)
        for row_index, row in enumerate(rows)
        if row["question"].strip()
        and len(row["options"]) == 4
        and row["answer_idx"] in row["options"]
    ]
    if len(valid_rows) < SAMPLES_PER_SOURCE:
        raise RuntimeError(
            f"MedQA has only {len(valid_rows)} valid test rows"
        )

    rng = random.Random(f"{SEED}:medqa")
    questions = []
    for source_index, row in rng.sample(valid_rows, SAMPLES_PER_SOURCE):
        option_keys = sorted(row["options"])
        correct_index = option_keys.index(row["answer_idx"])
        question = {
            "id": f"medqa-test-{source_index}",
            "source": "MedQA",
            "source_subject": "USMLE",
            "split": "test",
            "domain": "medical",
            "stem": row["question"].strip(),
        }
        questions.append(
            _shuffle_question(
                question,
                [str(row["options"][key]).strip() for key in option_keys],
                correct_index,
            )
        )
    return questions, {"USMLE": len(rows)}, [url]


def _parse_number(text):
    normalized = text.replace(",", "").strip()
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _format_number(number, use_commas):
    if number == number.to_integral():
        return f"{int(number):,}" if use_commas else str(int(number))
    text = format(number.normalize(), "f")
    return text.rstrip("0").rstrip(".")


def _gsm8k_distractors(correct_text, rng):
    correct_number = _parse_number(correct_text)
    use_commas = "," in correct_text
    if correct_number is None:
        raise ValueError(f"unsupported GSM8K answer: {correct_text}")

    deltas = [1, -1, 2, -2, 3, -3, 5, -5, 10, -10]
    if abs(correct_number) >= 100:
        deltas.extend([20, -20, 50, -50, 100, -100])
    rng.shuffle(deltas)

    distractors = []
    for delta in deltas:
        candidate = _format_number(correct_number + delta, use_commas)
        if candidate != correct_text and candidate not in distractors:
            distractors.append(candidate)
        if len(distractors) == 3:
            return distractors

    offset = 11
    while len(distractors) < 3:
        candidate = _format_number(correct_number + offset, use_commas)
        if candidate != correct_text and candidate not in distractors:
            distractors.append(candidate)
        offset += 1
    return distractors


def _sample_gsm8k(cache_dir):
    target = cache_dir / "gsm8k-main.zip"
    _download(GSM8K_ARCHIVE_URL, target)
    with zipfile.ZipFile(target) as archive:
        test_name = next(
            name
            for name in archive.namelist()
            if name.endswith("grade_school_math/data/test.jsonl")
        )
        rows = [
            json.loads(line)
            for line in archive.read(test_name).decode("utf-8").splitlines()
            if line.strip()
        ]

    parsed_rows = []
    for row_index, row in enumerate(rows):
        match = re.search(r"####\s*(.+?)\s*$", row["answer"])
        if match and row["question"].strip():
            parsed_rows.append((row_index, row, match.group(1).strip()))
    if len(parsed_rows) < SAMPLES_PER_SOURCE:
        raise RuntimeError(
            f"GSM8K has only {len(parsed_rows)} valid test rows"
        )

    rng = random.Random(f"{SEED}:gsm8k")
    questions = []
    for source_index, row, correct_text in rng.sample(
        parsed_rows, SAMPLES_PER_SOURCE
    ):
        question = {
            "id": f"gsm8k-test-{source_index}",
            "source": "GSM8K",
            "source_subject": "grade_school_math",
            "split": "test",
            "domain": "math",
            "stem": row["question"].strip(),
        }
        option_rng = random.Random(f"{SEED}:gsm8k:options:{source_index}")
        options = [correct_text] + _gsm8k_distractors(correct_text, option_rng)
        questions.append(_shuffle_question(question, options, 0))
    return questions, {"grade_school_math": len(rows)}, [GSM8K_ARCHIVE_URL]


def _validate_questions(questions):
    if len(questions) != TOTAL_QUESTIONS:
        raise RuntimeError(
            f"expected {TOTAL_QUESTIONS} questions, got {len(questions)}"
        )

    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate question ids found")
    stems = [question["stem"] for question in questions]
    if len(stems) != len(set(stems)):
        raise RuntimeError("duplicate question stems found")

    source_counts = Counter(question["source"] for question in questions)
    domain_counts = Counter(question["domain"] for question in questions)
    expected = {
        "MMLU": SAMPLES_PER_SOURCE,
        "C-Eval": SAMPLES_PER_SOURCE,
        "MedQA": SAMPLES_PER_SOURCE,
        "GSM8K": SAMPLES_PER_SOURCE,
    }
    if dict(source_counts) != expected:
        raise RuntimeError(f"unexpected source counts: {dict(source_counts)}")
    if domain_counts != Counter(
        {"general": 500, "coding": 500, "medical": 500, "math": 500}
    ):
        raise RuntimeError(f"unexpected domain counts: {dict(domain_counts)}")

    for question in questions:
        if not question["stem"] or len(question["options"]) != 4:
            raise RuntimeError(f"invalid question structure: {question['id']}")
        if any(not option for option in question["options"]):
            raise RuntimeError(f"empty option found: {question['id']}")
        if len(question["options"]) != len(set(question["options"])):
            raise RuntimeError(f"duplicate option found: {question['id']}")
        if not 0 <= question["answer"] < 4:
            raise RuntimeError(f"invalid answer index: {question['id']}")


def main():
    with tempfile.TemporaryDirectory(prefix="goa-public-data-") as temp_dir:
        cache_dir = Path(temp_dir)
        mmlu, mmlu_rows, mmlu_urls = _sample_mmlu(cache_dir)
        ceval, ceval_rows, ceval_urls = _sample_ceval(cache_dir)
        medqa, medqa_rows, medqa_urls = _sample_medqa(cache_dir)
        gsm8k, gsm8k_rows, gsm8k_urls = _sample_gsm8k(cache_dir)

    questions = mmlu + ceval + medqa + gsm8k
    random.Random(SEED).shuffle(questions)
    _validate_questions(questions)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(questions, ensure_ascii=False, indent=2)
    DATA_PATH.write_bytes(payload.encode("utf-8"))

    manifest = {
        "generated_on": date.today().isoformat(),
        "seed": SEED,
        "total_questions": len(questions),
        "counts_by_source": dict(Counter(q["source"] for q in questions)),
        "counts_by_domain": dict(Counter(q["domain"] for q in questions)),
        "unique_stems": len({q["stem"] for q in questions}),
        "dataset_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "sources": {
            "MMLU": {
                "dataset": MMLU_DATASET,
                "split": "test",
                "subjects": list(MMLU_SUBJECTS),
                "source_rows": mmlu_rows,
                "sampled": len(mmlu),
                "urls": mmlu_urls,
            },
            "C-Eval": {
                "dataset": CEVAL_DATASET,
                "split": "test",
                "subjects": list(CEVAL_SUBJECTS),
                "source_rows": ceval_rows,
                "sampled": len(ceval),
                "urls": ceval_urls,
            },
            "MedQA": {
                "dataset": MEDQA_DATASET,
                "split": "test",
                "subjects": ["USMLE"],
                "source_rows": medqa_rows,
                "sampled": len(medqa),
                "urls": medqa_urls,
            },
            "GSM8K": {
                "dataset": "openai/grade-school-math",
                "split": "test",
                "subjects": ["grade_school_math"],
                "source_rows": gsm8k_rows,
                "sampled": len(gsm8k),
                "urls": gsm8k_urls,
            },
        },
        "notes": [
            "Each source contributes exactly 500 questions from test splits.",
            "Option order is deterministically shuffled for every question.",
            "GSM8K is open-ended in the original dataset; its gold numeric answer "
            "is converted to a four-option question with deterministic numeric "
            "distractors.",
            "MMLU is mapped to general knowledge, C-Eval computer subjects to "
            "coding, MedQA to medical, and GSM8K to math.",
        ],
    }
    MANIFEST_PATH.write_bytes(
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    )

    print(
        f"generated {len(questions)} public questions "
        f"({SAMPLES_PER_SOURCE} per source) -> {DATA_PATH}"
    )
    print(f"manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
