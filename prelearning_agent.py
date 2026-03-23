#!/usr/bin/env python3
"""PreLearning-Agent for MM-ML-1M multimodal embeddings.

Pipeline:
1) Load movie metadata from movies_details_clean.csv.
2) Extract poster visual embeddings from posters/ with local CLIP.
3) Extract movie description text embeddings with the same CLIP text encoder.
4) Save JSON keyed by item_id under Dataset/MM-ML-1M-main/Embeddings.
5) Verify completeness and write a report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


DEFAULT_DATASET_DIR = Path("/Users/chongzhang/Web_sim/Dataset/MM-ML-1M-main")
DEFAULT_MODEL_PATH = Path("/Users/chongzhang/openai-clip-vit-base-patch16")


@dataclass
class MovieRecord:
    item_id: str
    title: str
    description: str
    poster_path: Path | None


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_description(row: dict[str, Any]) -> str:
    # Prefer detailed overview, then review, then genres/title as fallback.
    for key in ("Overview", "Review", "Genres"):
        text = safe_text(row.get(key))
        if text:
            return text
    return safe_text(row.get("Title")) or "No description"


def resolve_poster_path(poster_dir: Path, item_id: str, poster_name: str) -> Path | None:
    if poster_name:
        candidate = poster_dir / poster_name
        if candidate.exists():
            return candidate

    # Fallback to "item_id_*.jpg" naming.
    matches = sorted(poster_dir.glob(f"{item_id}_*.jpg"))
    if matches:
        return matches[0]
    return None


def load_movie_records(csv_path: Path, poster_dir: Path) -> tuple[list[MovieRecord], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing csv file: {csv_path}")
    if not poster_dir.exists():
        raise FileNotFoundError(f"Missing posters dir: {poster_dir}")

    df = pd.read_csv(csv_path)
    if "Id" not in df.columns:
        raise ValueError(f"'Id' column not found in {csv_path}")

    records: list[MovieRecord] = []
    duplicates: list[str] = []
    seen_ids: set[str] = set()

    for row in df.to_dict(orient="records"):
        raw_id = row.get("Id")
        if pd.isna(raw_id):
            continue
        try:
            item_id = str(int(raw_id))
        except Exception:
            item_id = safe_text(raw_id)
        if not item_id:
            continue
        if item_id in seen_ids:
            duplicates.append(item_id)
            continue
        seen_ids.add(item_id)

        title = safe_text(row.get("Title")) or f"Movie {item_id}"
        description = build_description(row)
        poster_name = safe_text(row.get("Poster Path"))
        poster_path = resolve_poster_path(poster_dir, item_id, poster_name)

        records.append(
            MovieRecord(
                item_id=item_id,
                title=title,
                description=description,
                poster_path=poster_path,
            )
        )

    return records, duplicates


def batched(items: list[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class PreLearningAgent:
    def __init__(
        self,
        dataset_dir: Path,
        model_path: Path,
        output_dir: Path,
        text_batch_size: int,
        image_batch_size: int,
        strict_check: bool,
    ):
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.output_dir = output_dir
        self.text_batch_size = max(1, text_batch_size)
        self.image_batch_size = max(1, image_batch_size)
        self.strict_check = strict_check

        self.csv_path = self.dataset_dir / "movies_details_clean.csv"
        self.poster_dir = self.dataset_dir / "posters"
        self.output_json = self.output_dir / "item_embeddings.json"
        self.report_json = self.output_dir / "prelearning_report.json"

        self.device = choose_device()
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None
        self.embed_dim: int | None = None

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        self.processor = CLIPProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = CLIPModel.from_pretrained(str(self.model_path), local_files_only=True).to(self.device)
        self.model.eval()
        self.embed_dim = int(getattr(self.model.config, "projection_dim", 512))

    def extract_text_embeddings(self, records: list[MovieRecord]) -> dict[str, list[float] | None]:
        assert self.model is not None and self.processor is not None
        result: dict[str, list[float] | None] = {}
        total = len(records)
        print(f"[Text] start: items={total}, batch_size={self.text_batch_size}")

        for idx, batch in enumerate(batched(records, self.text_batch_size), start=1):
            texts = [x.description for x in batch]
            ids = [x.item_id for x in batch]
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
                feats = F.normalize(feats, p=2, dim=-1)
            feats = feats.detach().cpu().float().tolist()
            for item_id, vec in zip(ids, feats):
                result[item_id] = [float(v) for v in vec]

            if idx % 10 == 0 or idx == 1:
                done = min(idx * self.text_batch_size, total)
                print(f"[Text] progress: {done}/{total}")

        return result

    def extract_visual_embeddings(
        self, records: list[MovieRecord]
    ) -> tuple[dict[str, list[float] | None], dict[str, str]]:
        assert self.model is not None and self.processor is not None
        result: dict[str, list[float] | None] = {x.item_id: None for x in records}
        image_errors: dict[str, str] = {}

        valid_records = [x for x in records if x.poster_path is not None]
        total = len(valid_records)
        print(f"[Image] start: items_with_poster={total}, batch_size={self.image_batch_size}")

        for idx, batch in enumerate(batched(valid_records, self.image_batch_size), start=1):
            images: list[Image.Image] = []
            ids: list[str] = []

            for rec in batch:
                assert rec.poster_path is not None
                try:
                    with Image.open(rec.poster_path) as img:
                        images.append(img.convert("RGB"))
                    ids.append(rec.item_id)
                except Exception as err:
                    image_errors[rec.item_id] = str(err)
                    result[rec.item_id] = None

            if images:
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
                    feats = F.normalize(feats, p=2, dim=-1)
                feats = feats.detach().cpu().float().tolist()
                for item_id, vec in zip(ids, feats):
                    result[item_id] = [float(v) for v in vec]

            if idx % 10 == 0 or idx == 1:
                done = min(idx * self.image_batch_size, total)
                print(f"[Image] progress: {done}/{total}")

        return result, image_errors

    def build_output_items(
        self,
        records: list[MovieRecord],
        text_embeddings: dict[str, list[float] | None],
        visual_embeddings: dict[str, list[float] | None],
    ) -> dict[str, dict[str, Any]]:
        embed_dim = int(self.embed_dim or 0)
        items: dict[str, dict[str, Any]] = {}
        for rec in records:
            text_vec = text_embeddings.get(rec.item_id)
            visual_vec = visual_embeddings.get(rec.item_id)
            text_source = "clip_text"
            visual_source = "clip_image_poster"

            if text_vec is None:
                text_vec = [0.0] * embed_dim
                text_source = "zero_fallback_missing_text"
            if visual_vec is None:
                visual_vec = [0.0] * embed_dim
                visual_source = "zero_fallback_missing_poster_or_decode_error"

            items[rec.item_id] = {
                "visual_embedding": visual_vec,
                "text_embedding": text_vec,
                "visual_source": visual_source,
                "text_source": text_source,
            }
        return items

    def verify(
        self,
        records: list[MovieRecord],
        items: dict[str, dict[str, Any]],
        duplicates: list[str],
        image_errors: dict[str, str],
    ) -> dict[str, Any]:
        expected_ids = [x.item_id for x in records]
        expected_set = set(expected_ids)
        output_set = set(items.keys())

        missing_items = sorted(expected_set - output_set)
        extra_items = sorted(output_set - expected_set)

        missing_poster_path = sorted([x.item_id for x in records if x.poster_path is None])
        missing_visual = sorted([k for k, v in items.items() if v.get("visual_embedding") is None])
        missing_text = sorted([k for k, v in items.items() if v.get("text_embedding") is None])
        visual_fallback_ids = sorted(
            [k for k, v in items.items() if v.get("visual_source") != "clip_image_poster"]
        )
        text_fallback_ids = sorted(
            [k for k, v in items.items() if v.get("text_source") != "clip_text"]
        )

        embed_dim = int(self.embed_dim or 0)
        visual_wrong_dim = sorted(
            [
                k
                for k, v in items.items()
                if v.get("visual_embedding") is not None and len(v["visual_embedding"]) != embed_dim
            ]
        )
        text_wrong_dim = sorted(
            [
                k
                for k, v in items.items()
                if v.get("text_embedding") is not None and len(v["text_embedding"]) != embed_dim
            ]
        )

        report = {
            "dataset_dir": str(self.dataset_dir),
            "model_path": str(self.model_path),
            "device": str(self.device),
            "embedding_dim": embed_dim,
            "total_records": len(records),
            "output_items": len(items),
            "duplicate_item_ids_skipped": len(duplicates),
            "missing_items_in_output": len(missing_items),
            "extra_items_in_output": len(extra_items),
            "missing_poster_path": len(missing_poster_path),
            "missing_visual_embedding": len(missing_visual),
            "missing_text_embedding": len(missing_text),
            "visual_fallback_count": len(visual_fallback_ids),
            "text_fallback_count": len(text_fallback_ids),
            "visual_wrong_dim": len(visual_wrong_dim),
            "text_wrong_dim": len(text_wrong_dim),
            "image_decode_errors": len(image_errors),
            "details": {
                "duplicates_sample": duplicates[:20],
                "missing_items_sample": missing_items[:20],
                "extra_items_sample": extra_items[:20],
                "missing_poster_path_sample": missing_poster_path[:20],
                "missing_visual_sample": missing_visual[:20],
                "missing_text_sample": missing_text[:20],
                "visual_fallback_sample": visual_fallback_ids[:20],
                "text_fallback_sample": text_fallback_ids[:20],
                "visual_wrong_dim_sample": visual_wrong_dim[:20],
                "text_wrong_dim_sample": text_wrong_dim[:20],
                "image_decode_errors_sample": dict(list(image_errors.items())[:20]),
            },
        }

        blockers = (
            report["missing_items_in_output"]
            + report["extra_items_in_output"]
            + report["missing_visual_embedding"]
            + report["missing_text_embedding"]
            + report["visual_wrong_dim"]
            + report["text_wrong_dim"]
        )
        report["all_checks_passed"] = blockers == 0
        return report

    def save_outputs(self, items: dict[str, dict[str, Any]], report: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_json.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def run(self) -> None:
        self.load_model()
        records, duplicates = load_movie_records(self.csv_path, self.poster_dir)
        print(f"[Load] records={len(records)}, duplicates_skipped={len(duplicates)}")

        text_embeddings = self.extract_text_embeddings(records)
        visual_embeddings, image_errors = self.extract_visual_embeddings(records)

        items = self.build_output_items(records, text_embeddings, visual_embeddings)
        report = self.verify(records, items, duplicates, image_errors)
        self.save_outputs(items, report)

        print(f"[Save] embeddings_json={self.output_json}")
        print(f"[Save] report_json={self.report_json}")
        print(
            "[Check] "
            f"total={report['total_records']}, "
            f"missing_visual={report['missing_visual_embedding']}, "
            f"missing_text={report['missing_text_embedding']}, "
            f"all_checks_passed={report['all_checks_passed']}"
        )

        if self.strict_check and not report["all_checks_passed"]:
            raise RuntimeError(
                "PreLearning-Agent checks failed. "
                f"See report: {self.report_json}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreLearning-Agent for MM-ML-1M")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR / "Embeddings",
    )
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not raise error if completeness checks fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = PreLearningAgent(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
        strict_check=not args.allow_missing,
    )
    agent.run()


if __name__ == "__main__":
    main()
