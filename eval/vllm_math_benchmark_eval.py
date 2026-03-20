#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


INSTRUCTION_SUFFIX = "Let's think step by step and output the final answer within \\boxed{}."
ALLOWED_K = (1, 2, 4, 8, 16, 32)
BENCHMARK_ALIASES = {
    "AIME24": "AIME24",
    "AIME25": "AIME25",
    "AMC23": "AMC23",
    "AMC": "AMC23",
    "MATH500": "MATH500",
    "MATH-500": "MATH500",
    "MINERVA": "MINERVA",
    "OLYMPIAD": "OLYMPIAD",
    "OLYMPIADBENCH": "OLYMPIAD",
    "OLYMPIAD-BENCH": "OLYMPIAD",
}


@dataclass
class BenchmarkSpec:
    name: str
    data_source: str
    local_path: str | None = None
    hf_dataset: str | None = None
    split: str = "test"
    question_key: str = "problem"
    answer_key: str = "answer"
    instruction_suffix: str = INSTRUCTION_SUFFIX


@dataclass
class EvalExample:
    benchmark: str
    example_id: int
    prompt_messages: list[dict[str, str]]
    question: str
    ground_truth: str
    data_source: str
    extra_info: dict[str, Any]


def load_local_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATH_DAPO = load_local_module("_verl_eval_math_dapo", REPO_ROOT / "verl" / "utils" / "reward_score" / "math_dapo.py")
MATH_REWARD = load_local_module("_verl_eval_math_reward", REPO_ROOT / "verl" / "utils" / "reward_score" / "math_reward.py")
last_boxed_only_string = MATH_DAPO.last_boxed_only_string
remove_boxed = MATH_DAPO.remove_boxed


def build_default_registry() -> dict[str, BenchmarkSpec]:
    return {
        "AIME24": BenchmarkSpec(
            name="AIME24",
            data_source="aime24",
            local_path=str(REPO_ROOT / "data" / "aime" / "aime-2024.parquet"),
            hf_dataset="Maxwell-Jia/AIME_2024",
            split="train",
            question_key="Problem",
            answer_key="Answer",
        ),
        "AIME25": BenchmarkSpec(
            name="AIME25",
            data_source="aime25",
            local_path=str(REPO_ROOT / "data" / "aime" / "aime-2025.parquet"),
            hf_dataset="yentinglin/aime_2025",
            split="train",
            question_key="problem",
            answer_key="solution",
        ),
        "AMC23": BenchmarkSpec(
            name="AMC23",
            data_source="numina_amc_aime",
            local_path=str(REPO_ROOT / "data" / "amc23" / "test.parquet"),
            question_key="problem",
            answer_key="answer",
        ),
        "MATH500": BenchmarkSpec(
            name="MATH500",
            data_source="HuggingFaceH4/MATH-500",
            local_path=str(REPO_ROOT / "data" / "math500" / "test.jsonl"),
            hf_dataset="HuggingFaceH4/MATH-500",
            split="test",
            question_key="problem",
            answer_key="answer",
        ),
        "MINERVA": BenchmarkSpec(
            name="Minerva",
            data_source="math_dapo",
            local_path=str(REPO_ROOT / "data" / "minerva" / "test.jsonl"),
            question_key="problem",
            answer_key="solution",
        ),
        "OLYMPIAD": BenchmarkSpec(
            name="Olympiad",
            data_source="numina_olympiads",
            local_path=str(REPO_ROOT / "data" / "olympiad" / "test.parquet"),
            question_key="question",
            answer_key="answer",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a math LLM with vLLM and compute pass@k.")
    parser.add_argument("--model", required=True, help="Local model path or HF model name.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["AIME24", "AIME25", "AMC23", "MATH500", "Minerva", "Olympiad"],
        help="Benchmarks to evaluate. Supported aliases: AIME24 AIME25 AMC23 MATH500 Minerva Olympiad.",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(ALLOWED_K),
        help="pass@k values to compute. Allowed values: 1 2 4 8 16 32.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of generations per problem. Defaults to max(k). Must be >= max(k).",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=4,
        help="How many samples to request from vLLM at once. Lower this to reduce memory for pass@k.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--swap-space", type=float, default=4.0)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path/name override.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-benchmark limit for quick smoke tests.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face datasets cache dir.")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stop", nargs="*", default=None, help="Optional stop strings passed to vLLM.")
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        help="Override benchmark dataset path. Format: BENCHMARK=/path/to/file_or_dir",
    )
    parser.add_argument(
        "--hf-dataset",
        action="append",
        default=[],
        help="Override benchmark HF dataset id. Format: BENCHMARK=org/name",
    )
    parser.add_argument("--split", action="append", default=[], help="Override split. Format: BENCHMARK=test")
    parser.add_argument(
        "--question-key",
        action="append",
        default=[],
        help="Override question field for raw datasets. Format: BENCHMARK=field_or.nested.field",
    )
    parser.add_argument(
        "--answer-key",
        action="append",
        default=[],
        help="Override answer field for raw datasets. Format: BENCHMARK=field_or.nested.field",
    )
    parser.add_argument(
        "--data-source",
        action="append",
        default=[],
        help="Override verl reward data_source. Format: BENCHMARK=data_source_name",
    )
    parser.add_argument(
        "--instruction-suffix",
        default=INSTRUCTION_SUFFIX,
        help="Prompt suffix appended to raw datasets. Preprocessed verl parquet prompts are reused as-is.",
    )
    return parser.parse_args()


def normalize_benchmark_name(name: str) -> str:
    normalized = name.strip().replace("_", "").replace(" ", "").upper()
    if normalized not in BENCHMARK_ALIASES:
        supported = ", ".join(sorted({"AIME24", "AIME25", "AMC23", "MATH500", "MINERVA", "OLYMPIAD"}))
        raise ValueError(f"Unsupported benchmark '{name}'. Supported values: {supported}")
    return BENCHMARK_ALIASES[normalized]


def parse_mapping_entries(entries: list[str], flag_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"{flag_name} expects BENCHMARK=value, got '{entry}'")
        key, value = entry.split("=", 1)
        parsed[normalize_benchmark_name(key)] = value
    return parsed


def resolve_registry(args: argparse.Namespace) -> dict[str, BenchmarkSpec]:
    registry = build_default_registry()
    dataset_paths = parse_mapping_entries(args.dataset_path, "--dataset-path")
    hf_datasets = parse_mapping_entries(args.hf_dataset, "--hf-dataset")
    splits = parse_mapping_entries(args.split, "--split")
    question_keys = parse_mapping_entries(args.question_key, "--question-key")
    answer_keys = parse_mapping_entries(args.answer_key, "--answer-key")
    data_sources = parse_mapping_entries(args.data_source, "--data-source")

    for key, value in dataset_paths.items():
        registry[key].local_path = value
    for key, value in hf_datasets.items():
        registry[key].hf_dataset = value
    for key, value in splits.items():
        registry[key].split = value
    for key, value in question_keys.items():
        registry[key].question_key = value
    for key, value in answer_keys.items():
        registry[key].answer_key = value
    for key, value in data_sources.items():
        registry[key].data_source = value

    for spec in registry.values():
        spec.instruction_suffix = args.instruction_suffix
    return registry


def ensure_dependencies(names: list[str], hint: str = "") -> None:
    missing = []
    for name in names:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing Python package(s): {', '.join(missing)}. {hint}".strip())


def load_json_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON payload in {path}")


def load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_parquet_file(path: Path) -> list[dict[str, Any]]:
    loaders = []

    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        return table.to_pylist()
    except Exception as exc:
        loaders.append(f"pyarrow: {exc}")

    try:
        import pandas as pd  # type: ignore

        return pd.read_parquet(path).to_dict(orient="records")
    except Exception as exc:
        loaders.append(f"pandas: {exc}")

    try:
        import datasets  # type: ignore

        dataset = datasets.Dataset.from_parquet(str(path))
        return [dataset[i] for i in range(len(dataset))]
    except Exception as exc:
        loaders.append(f"datasets: {exc}")

    raise RuntimeError(f"Could not read parquet file {path}. Tried: {' | '.join(loaders)}")


def load_hf_dataset(spec: BenchmarkSpec, cache_dir: str | None) -> list[dict[str, Any]]:
    ensure_dependencies(["datasets"], "Install `datasets` to load Hugging Face datasets.")
    import datasets  # type: ignore

    dataset = datasets.load_dataset(spec.hf_dataset, split=spec.split, cache_dir=cache_dir)
    return [dataset[i] for i in range(len(dataset))]


def load_records_for_benchmark(spec: BenchmarkSpec, cache_dir: str | None) -> tuple[list[dict[str, Any]], str]:
    if spec.local_path:
        path = Path(spec.local_path).expanduser().resolve()
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                return load_parquet_file(path), str(path)
            if suffix == ".json":
                return load_json_file(path), str(path)
            if suffix == ".jsonl":
                return load_jsonl_file(path), str(path)
            raise ValueError(f"Unsupported dataset file format: {path}")
        if path.is_dir():
            ensure_dependencies(["datasets"], "Install `datasets` to load directory-based datasets.")
            import datasets  # type: ignore

            try:
                dataset = datasets.load_from_disk(str(path))
                if hasattr(dataset, "keys"):
                    dataset = dataset[spec.split]
                return [dataset[i] for i in range(len(dataset))], str(path)
            except Exception:
                dataset = datasets.load_dataset(str(path), split=spec.split, cache_dir=cache_dir)
                return [dataset[i] for i in range(len(dataset))], str(path)

        if spec.hf_dataset:
            return load_hf_dataset(spec, cache_dir), f"hf://{spec.hf_dataset}[{spec.split}]"
        raise FileNotFoundError(f"Dataset path does not exist for {spec.name}: {path}")

    if spec.hf_dataset:
        return load_hf_dataset(spec, cache_dir), f"hf://{spec.hf_dataset}[{spec.split}]"

    raise FileNotFoundError(
        f"No dataset configured for {spec.name}. Use --dataset-path {spec.name}=... or --hf-dataset {spec.name}=..."
    )


def get_nested_value(record: Mapping[str, Any], key: str) -> Any:
    value: Any = record
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Missing field '{key}'")
        value = value[part]
    return value


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def extract_ground_truth(value: Any) -> str:
    text = stringify_value(value)
    if not text:
        return text

    boxed = last_boxed_only_string(text)
    if boxed is not None:
        try:
            return remove_boxed(boxed).strip()
        except Exception:
            pass

    answer_matches = [match.strip() for match in re.findall(r"(?i)answer\s*:\s*([^\n]+)", text)]
    if answer_matches:
        return answer_matches[-1]

    if "####" in text:
        return text.split("####")[-1].strip()

    return text.strip()


def extract_question_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return stringify_value(message.get("content"))
    if messages:
        return stringify_value(messages[-1].get("content"))
    return ""


def build_prompt_messages(question: str, instruction_suffix: str) -> list[dict[str, str]]:
    prompt = question.strip()
    if instruction_suffix:
        prompt = f"{prompt} {instruction_suffix}".strip()
    return [{"role": "user", "content": prompt}]


def build_eval_examples(
    records: list[dict[str, Any]],
    spec: BenchmarkSpec,
    benchmark_key: str,
    limit: int | None,
) -> list[EvalExample]:
    if limit is not None:
        records = records[:limit]

    examples: list[EvalExample] = []
    for idx, record in enumerate(records):
        if isinstance(record.get("prompt"), list) and isinstance(record.get("reward_model"), Mapping):
            prompt_messages = [
                {
                    "role": stringify_value(message.get("role")),
                    "content": stringify_value(message.get("content")),
                }
                for message in record["prompt"]
            ]
            extra_info = copy.deepcopy(record.get("extra_info", {}))
            question = stringify_value(extra_info.get("question")) or extract_question_from_messages(prompt_messages)
            ground_truth = extract_ground_truth(record["reward_model"].get("ground_truth"))
            data_source = stringify_value(record.get("data_source")) or spec.data_source
        else:
            question = stringify_value(get_nested_value(record, spec.question_key))
            ground_truth = extract_ground_truth(get_nested_value(record, spec.answer_key))
            prompt_messages = build_prompt_messages(question, spec.instruction_suffix)
            data_source = spec.data_source
            extra_info = {
                key: value
                for key, value in record.items()
                if key.split(".")[0] not in {spec.question_key.split(".")[0], spec.answer_key.split(".")[0]}
            }

        examples.append(
            EvalExample(
                benchmark=benchmark_key,
                example_id=idx,
                prompt_messages=prompt_messages,
                question=question,
                ground_truth=ground_truth,
                data_source=data_source,
                extra_info=extra_info,
            )
        )
    return examples


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    parts = []
    for message in messages:
        role = message.get("role", "user").strip().capitalize() or "User"
        content = message.get("content", "").strip()
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def build_generator(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    ensure_dependencies(
        ["transformers", "vllm"],
        "Install `transformers` and `vllm` in the selected Conda environment before running evaluation.",
    )
    from transformers import AutoTokenizer  # type: ignore
    from vllm import LLM  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )

    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "swap_space": args.swap_space,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "stop": args.stop,
    }
    return llm, tokenizer, sampling_kwargs


def coerce_score_result(result: Any) -> tuple[float, bool, Any]:
    if isinstance(result, Mapping):
        score = float(result.get("score", result.get("acc", 0.0)))
        acc = bool(result.get("acc", score > 0))
        pred = result.get("pred")
        return score, acc, pred

    score = float(result)
    return score, score > 0, None


def compute_score(solution_str: str, ground_truth: str, data_source: str, benchmark_key: str) -> tuple[float, bool, Any]:
    math_reward_sources = {"HuggingFaceH4/MATH-500", "DigitalLearningGmbH/MATH-lighteval", "lighteval/MATH"}
    if benchmark_key == "MATH500" or data_source in math_reward_sources:
        return coerce_score_result(MATH_REWARD.compute_score(solution_str, ground_truth))
    return coerce_score_result(MATH_DAPO.compute_score(solution_str, ground_truth))


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float | None:
    if k > num_samples:
        return None
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - (math.comb(num_samples - num_correct, k) / math.comb(num_samples, k))


def evaluate_benchmark(
    benchmark_key: str,
    spec: BenchmarkSpec,
    source_ref: str,
    examples: list[EvalExample],
    llm: Any,
    tokenizer: Any,
    sampling_kwargs: dict[str, Any],
    ks: list[int],
    output_dir: Path,
    num_samples: int,
    temperature: float,
    sample_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import SamplingParams  # type: ignore

    prompts = [render_prompt(tokenizer, example.prompt_messages) for example in examples]
    benchmark_rows = [
        {
            "benchmark": example.benchmark,
            "example_id": example.example_id,
            "question": example.question,
            "ground_truth": example.ground_truth,
            "data_source": example.data_source,
            "prompt": rendered_prompt,
            "extra_info": example.extra_info,
            "samples": [],
        }
        for example, rendered_prompt in zip(examples, prompts, strict=True)
    ]
    pass_totals = {k: 0.0 for k in ks}
    mean_sample_accuracy = 0.0

    sample_index_offset = 0
    remaining = num_samples
    while remaining > 0:
        current_n = min(sample_batch_size, remaining)
        current_sampling_params = SamplingParams(
            n=current_n,
            seed=seed + sample_index_offset,
            **sampling_kwargs,
        )
        outputs = llm.generate(prompts, sampling_params=current_sampling_params, use_tqdm=True)
        for row, example, request_output in zip(benchmark_rows, examples, outputs, strict=True):
            for completion in request_output.outputs:
                text = completion.text
                score, acc, pred = compute_score(
                    solution_str=text,
                    ground_truth=example.ground_truth,
                    data_source=example.data_source,
                    benchmark_key=benchmark_key,
                )
                row["samples"].append(
                    {
                        "sample_index": len(row["samples"]),
                        "score": score,
                        "acc": acc,
                        "pred": pred,
                        "text": text,
                    }
                )
        sample_index_offset += current_n
        remaining -= current_n

    for row in benchmark_rows:
        sample_rows = row["samples"]
        num_correct = sum(int(sample["acc"]) for sample in sample_rows)

        pass_metrics = {}
        for k in ks:
            value = estimate_pass_at_k(len(sample_rows), num_correct, k)
            if value is not None:
                pass_metrics[f"pass@{k}"] = value
        for k in ks:
            key = f"pass@{k}"
            if key in pass_metrics:
                pass_totals[k] += pass_metrics[key]

        mean_sample_accuracy += num_correct / len(sample_rows)
        row["num_samples"] = len(sample_rows)
        row["num_correct"] = num_correct
        row.update(pass_metrics)

    jsonl_path = output_dir / f"{benchmark_key.lower()}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for row in benchmark_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    num_examples = len(benchmark_rows)
    summary = {
        "benchmark": benchmark_key,
        "display_name": spec.name,
        "source": source_ref,
        "num_examples": num_examples,
        "num_samples_per_example": num_samples,
        "sample_batch_size": sample_batch_size,
        "temperature": temperature,
        "mean_sample_accuracy": mean_sample_accuracy / num_examples if num_examples else 0.0,
        "results_file": str(jsonl_path),
    }
    for k in ks:
        summary[f"pass@{k}"] = pass_totals[k] / num_examples if num_examples else 0.0
    return summary


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # model_name = Path(args.model.rstrip("/")).name or "model"
    model_name = args.model
    model_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in model_name)
    k_part = "-".join(str(k) for k in args.k)
    return f"{model_name}_temp{args.temperature}_n{args.num_samples}_k{k_part}_{timestamp}"


def main() -> None:
    args = parse_args()

    args.k = sorted(set(args.k))
    invalid_ks = [k for k in args.k if k not in ALLOWED_K]
    if invalid_ks:
        raise ValueError(f"Unsupported k values: {invalid_ks}. Allowed values: {list(ALLOWED_K)}")

    if args.num_samples is None:
        args.num_samples = max(args.k)
    if args.num_samples < max(args.k):
        raise ValueError("--num-samples must be >= max(k)")
    if args.sample_batch_size <= 0:
        raise ValueError("--sample-batch-size must be > 0")
    if args.sample_batch_size > args.num_samples:
        args.sample_batch_size = args.num_samples

    registry = resolve_registry(args)
    selected_benchmarks = [normalize_benchmark_name(name) for name in args.benchmarks]

    run_dir = Path(args.output_dir).expanduser().resolve() / build_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    llm, tokenizer, sampling_kwargs = build_generator(args)

    benchmark_summaries = {}
    for benchmark_key in selected_benchmarks:
        spec = registry[benchmark_key]
        records, source_ref = load_records_for_benchmark(spec, cache_dir=args.cache_dir)
        examples = build_eval_examples(records, spec, benchmark_key, args.limit)
        if not examples:
            raise ValueError(f"No examples found for {benchmark_key} from {source_ref}")
        benchmark_summaries[benchmark_key] = evaluate_benchmark(
            benchmark_key=benchmark_key,
            spec=spec,
            source_ref=source_ref,
            examples=examples,
            llm=llm,
            tokenizer=tokenizer,
            sampling_kwargs=sampling_kwargs,
            ks=args.k,
            output_dir=run_dir,
            num_samples=args.num_samples,
            temperature=args.temperature,
            sample_batch_size=args.sample_batch_size,
            seed=args.seed,
        )

    overall = {
        "run_name": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "benchmarks": selected_benchmarks,
        "ks": args.k,
        "num_samples": args.num_samples,
        "sample_batch_size": args.sample_batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit": args.limit,
        "output_dir": str(run_dir),
        "benchmark_summaries": benchmark_summaries,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to: {run_dir}")


if __name__ == "__main__":
    main()


# GPU_MEMORY_UTILIZATION=0.75 \
# SAMPLE_BATCH_SIZE=4 \
# bash /mnt/data/safetyCode/P2/verl/eval/run_vllm_math_benchmark_eval.sh \
#   /mnt/data/safetyCode/P2/verl/checkpoints/qwen2_5_math_1_5b_grpo_math_paper_4gpu_arithmetic/global_step_1100/actor/huggingface