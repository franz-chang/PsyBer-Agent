#!/usr/bin/env python3
"""Build Amazon Review 2018 multimodal subsets with resumable downloads.

Target subsets:
- All_Beauty
- Magazine_Subscriptions

Output structure (per subset) is aligned with MM-ML-1M style core files:
- ratings.dat
- users.dat
- movies.dat
- movies_details_clean.csv
- posters/

Plus multimodal manifests and logs:
- items_with_title_desc_image.jsonl
- items_with_title_desc_image.csv
- download_log.csv
- raw/* source files
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import re
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


@dataclass(frozen=True)
class CategorySpec:
    name: str
    meta_url: str
    ratings_url: str
    reviews_5_url: str


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "All_Beauty": CategorySpec(
        name="All_Beauty",
        meta_url="http://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz",
        ratings_url="http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/All_Beauty.csv",
        reviews_5_url="http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz",
    ),
    "Magazine_Subscriptions": CategorySpec(
        name="Magazine_Subscriptions",
        meta_url="http://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Magazine_Subscriptions.json.gz",
        ratings_url="http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Magazine_Subscriptions.csv",
        reviews_5_url="http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Magazine_Subscriptions_5.json.gz",
    ),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [normalize_text(x) for x in value]
        parts = [x for x in parts if x]
        return " ".join(parts).strip()
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_dat_field(value: str) -> str:
    text = normalize_text(value)
    text = text.replace("::", ":")
    return text


def safe_ext_from_url(url: str) -> str:
    base = url.split("?")[0].lower()
    if base.endswith(".png"):
        return ".png"
    if base.endswith(".webp"):
        return ".webp"
    if base.endswith(".gif"):
        return ".gif"
    if base.endswith(".jpeg"):
        return ".jpeg"
    if base.endswith(".jpg"):
        return ".jpg"
    return ".jpg"


def collect_image_urls(obj: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    for key in ("imageURLHighRes", "imageURL", "image"):
        value = obj.get(key)
        if isinstance(value, list):
            for v in value:
                if isinstance(v, str):
                    u = v.strip()
                    if u:
                        urls.append(u)
        elif isinstance(value, str):
            u = value.strip()
            if u:
                urls.append(u)

    # Backward compatibility with new-style metadata schema.
    images = obj.get("images")
    if isinstance(images, list):
        for img in images:
            if not isinstance(img, dict):
                continue
            for key in ("hi_res", "large", "thumb"):
                v = img.get(key)
                if isinstance(v, str):
                    u = v.strip()
                    if u:
                        urls.append(u)

    dedup: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for _ in reader:
            rows += 1
    return rows


def download_file_with_resume(url: str, dst: Path, timeout: int = 60, chunk_size: int = 1 << 20) -> None:
    """Resumable download to dst using a .part temp file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")

    for attempt in range(1, 6):
        existing = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200))

                # Server ignored Range, restart full file.
                if existing > 0 and status != 206:
                    tmp.unlink(missing_ok=True)
                    existing = 0

                mode = "ab" if existing > 0 else "wb"
                with tmp.open(mode) as wf:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        wf.write(chunk)

            if tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(dst)
                return

        except Exception as err:
            if attempt >= 5:
                raise RuntimeError(f"download failed: {url} -> {dst} ({err})") from err
            time.sleep(min(2 * attempt, 8))

    raise RuntimeError(f"download failed after retries: {url}")


def parse_ratings_csv(ratings_csv: Path) -> tuple[list[tuple[str, str, float, int]], set[str], dict[str, tuple[float, int]]]:
    interactions: list[tuple[str, str, float, int]] = []
    asins: set[str] = set()
    stat_sum: dict[str, float] = defaultdict(float)
    stat_cnt: dict[str, int] = defaultdict(int)

    with ratings_csv.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            asin = parts[0].strip()
            user = parts[1].strip()
            if not asin or not user:
                continue
            try:
                rating = float(parts[2])
                timestamp = int(float(parts[3]))
            except Exception:
                continue

            interactions.append((user, asin, rating, timestamp))
            asins.add(asin)
            stat_sum[asin] += rating
            stat_cnt[asin] += 1

    item_stats: dict[str, tuple[float, int]] = {}
    for asin in asins:
        cnt = stat_cnt[asin]
        avg = stat_sum[asin] / cnt if cnt else 0.0
        item_stats[asin] = (avg, cnt)

    return interactions, asins, item_stats


def parse_reviews_5(reviews_gz: Path, max_reviews_per_item: int = 3) -> dict[str, str]:
    out: dict[str, list[str]] = defaultdict(list)
    with gzip.open(reviews_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            asin = normalize_text(obj.get("asin"))
            if not asin:
                continue
            if len(out[asin]) >= max_reviews_per_item:
                continue
            text = normalize_text(obj.get("reviewText"))
            if not text:
                text = normalize_text(obj.get("summary"))
            if text:
                out[asin].append(text)

    return {k: "#####".join(v) for k, v in out.items() if v}


def parse_meta(meta_gz: Path, target_asins: set[str]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    with gzip.open(meta_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            asin = normalize_text(obj.get("asin"))
            if not asin or asin not in target_asins:
                continue

            title = normalize_text(obj.get("title"))
            description = normalize_text(obj.get("description"))
            if not description:
                description = normalize_text(obj.get("feature"))
            if not description:
                description = normalize_text(obj.get("features"))

            image_urls = collect_image_urls(obj)
            image_url = image_urls[0] if image_urls else ""
            if not (title and description and image_url):
                continue

            cats_raw = obj.get("category")
            categories: list[str] = []
            if isinstance(cats_raw, list):
                categories = [normalize_text(x) for x in cats_raw if normalize_text(x)]

            brand = normalize_text(obj.get("brand"))
            items[asin] = {
                "asin": asin,
                "title": title,
                "description": description,
                "image_url": image_url,
                "image_urls": image_urls,
                "categories": categories,
                "category_path": " > ".join(categories),
                "brand": brand,
            }

    return items


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_one(url: str, out_path: Path, timeout: int = 10) -> tuple[bool, str, int]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "exists", int(out_path.stat().st_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return False, "empty", 0
        out_path.write_bytes(data)
        return True, "downloaded", len(data)
    except Exception as err:
        return False, str(err), 0


def build_category_dataset(spec: CategorySpec, output_root: Path) -> dict[str, Any]:
    category_dir = output_root / spec.name
    raw_dir = category_dir / "raw"
    posters_dir = category_dir / "posters"
    category_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    posters_dir.mkdir(parents=True, exist_ok=True)

    raw_meta = raw_dir / f"meta_{spec.name}.json.gz"
    raw_ratings = raw_dir / f"{spec.name}.csv"
    raw_reviews_5 = raw_dir / f"{spec.name}_5.json.gz"

    log(f"[{spec.name}] download raw files")
    download_file_with_resume(spec.meta_url, raw_meta)
    download_file_with_resume(spec.ratings_url, raw_ratings)
    download_file_with_resume(spec.reviews_5_url, raw_reviews_5)

    log(f"[{spec.name}] parse ratings")
    interactions, rated_asins, rating_stats = parse_ratings_csv(raw_ratings)

    log(f"[{spec.name}] parse metadata")
    meta_items = parse_meta(raw_meta, rated_asins)

    log(f"[{spec.name}] parse 5-core reviews")
    review_map = parse_reviews_5(raw_reviews_5)

    filtered = [(u, a, r, t) for (u, a, r, t) in interactions if a in meta_items]
    if not filtered:
        raise RuntimeError(f"[{spec.name}] no usable interactions after metadata/image filtering")

    item_asins = sorted({a for _, a, _, _ in filtered})
    user_ids = sorted({u for u, _, _, _ in filtered})
    item_to_idx = {asin: i + 1 for i, asin in enumerate(item_asins)}
    user_to_idx = {uid: i + 1 for i, uid in enumerate(user_ids)}

    # Manifest rows for multimodal data + image download.
    manifest_rows: list[dict[str, Any]] = []
    for asin in item_asins:
        info = meta_items[asin]
        item_idx = item_to_idx[asin]
        image_url = str(info["image_url"])
        ext = safe_ext_from_url(image_url)
        url_hash = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:12]
        poster_file = f"{item_idx}_{url_hash}{ext}"

        row = {
            "item_idx": item_idx,
            "item_id": asin,
            "title": info["title"],
            "description": info["description"],
            "image_url": image_url,
            "image_urls": info["image_urls"],
            "category_path": info["category_path"],
            "brand": info["brand"],
            "poster_file": poster_file,
        }
        manifest_rows.append(row)

    manifest_rows.sort(key=lambda x: int(x["item_idx"]))

    manifest_jsonl = category_dir / "items_with_title_desc_image.jsonl"
    manifest_csv = category_dir / "items_with_title_desc_image.csv"
    write_jsonl(manifest_jsonl, manifest_rows)

    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "item_idx",
                "item_id",
                "title",
                "description",
                "image_url",
                "image_urls",
                "category_path",
                "brand",
                "poster_file",
            ]
        )
        for row in manifest_rows:
            writer.writerow(
                [
                    row["item_idx"],
                    row["item_id"],
                    row["title"],
                    row["description"],
                    row["image_url"],
                    " | ".join(row["image_urls"]),
                    row["category_path"],
                    row["brand"],
                    row["poster_file"],
                ]
            )

    item_poster = {r["item_id"]: r["poster_file"] for r in manifest_rows}

    # Write dat files.
    movies_dat = category_dir / "movies.dat"
    users_dat = category_dir / "users.dat"
    ratings_dat = category_dir / "ratings.dat"
    details_csv = category_dir / "movies_details_clean.csv"
    item_map_csv = category_dir / "item_id_map.csv"
    user_map_csv = category_dir / "user_id_map.csv"

    with item_map_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_idx", "asin"])
        for asin in item_asins:
            writer.writerow([item_to_idx[asin], asin])

    with user_map_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_idx", "reviewer_id"])
        for uid in user_ids:
            writer.writerow([user_to_idx[uid], uid])

    with movies_dat.open("w", encoding="utf-8") as f:
        for asin in item_asins:
            idx = item_to_idx[asin]
            info = meta_items[asin]
            title = sanitize_dat_field(info["title"]) or f"Item {idx}"
            genres = sanitize_dat_field("|".join(info["categories"])) or "Unknown"
            f.write(f"{idx}::{title}::{genres}\n")

    with users_dat.open("w", encoding="utf-8") as f:
        for uid in user_ids:
            f.write(f"{user_to_idx[uid]}::U::0::0::00000\n")

    filtered_sorted = sorted(filtered, key=lambda x: (user_to_idx[x[0]], x[3], item_to_idx[x[1]]))
    with ratings_dat.open("w", encoding="utf-8") as f:
        for user, asin, rating, ts in filtered_sorted:
            f.write(f"{user_to_idx[user]}::{item_to_idx[asin]}::{rating:.1f}::{ts}\n")

    with details_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Id",
                "Title",
                "Overview",
                "Genres",
                "Rating",
                "Vote Count",
                "Release Date",
                "Directors",
                "Actors",
                "Review",
                "Poster Path",
            ]
        )
        for asin in item_asins:
            idx = item_to_idx[asin]
            info = meta_items[asin]
            avg, cnt = rating_stats.get(asin, (0.0, 0))
            writer.writerow(
                [
                    idx,
                    info["title"],
                    info["description"],
                    ",".join(info["categories"]),
                    f"{avg:.4f}" if cnt else "",
                    cnt,
                    "",
                    info["brand"],
                    "",
                    review_map.get(asin, ""),
                    item_poster[asin],
                ]
            )

    stats = {
        "category": spec.name,
        "raw_meta": str(raw_meta),
        "raw_ratings": str(raw_ratings),
        "raw_reviews_5": str(raw_reviews_5),
        "ratings_total": len(interactions),
        "ratings_after_multimodal_filter": len(filtered),
        "users": len(user_ids),
        "items": len(item_asins),
        "manifest_jsonl": str(manifest_jsonl),
        "movies_dat": str(movies_dat),
        "users_dat": str(users_dat),
        "ratings_dat": str(ratings_dat),
        "details_csv": str(details_csv),
        "posters_dir": str(posters_dir),
    }

    (category_dir / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def download_images_for_category(
    category_dir: Path,
    workers: int,
    timeout: int,
    pending_factor: int,
    max_items: int,
    resume_fast: bool,
) -> dict[str, Any]:
    manifest = category_dir / "items_with_title_desc_image.jsonl"
    posters_dir = category_dir / "posters"
    log_csv = category_dir / "download_log.csv"

    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    already_logged = count_csv_rows(log_csv)
    attempted = already_logged if resume_fast else 0
    downloaded = 0
    existed = 0
    failed = 0
    bytes_downloaded = 0
    write_count = 0
    flush_every = 200
    start_ts = time.time()
    last_report = start_ts

    need_header = not log_csv.exists() or log_csv.stat().st_size == 0

    if resume_fast and already_logged > 0:
        log(f"[{category_dir.name}] FAST-RESUME: skip first {already_logged} rows")

    def handle_done(fut, writer):
        nonlocal downloaded, existed, failed, bytes_downloaded, write_count
        rec = fut.meta
        ok, status, size = fut.result()
        if ok:
            if status == "downloaded":
                downloaded += 1
                bytes_downloaded += int(size)
            else:
                existed += 1
        else:
            failed += 1

        writer.writerow(
            [
                rec["row_idx"],
                rec["item_idx"],
                rec["item_id"],
                rec["image_url"],
                rec["poster_file"],
                status,
                int(size),
            ]
        )
        write_count += 1

    with log_csv.open("a", encoding="utf-8", newline="") as cf, ThreadPoolExecutor(max_workers=workers) as ex:
        writer = csv.writer(cf)
        if need_header:
            writer.writerow(["row_idx", "item_idx", "item_id", "image_url", "poster_file", "status", "size_bytes"])

        pending = set()
        max_pending = max(workers * max(1, pending_factor), workers)

        with manifest.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if resume_fast and idx < already_logged:
                    continue

                if max_items > 0 and attempted >= max_items:
                    break

                attempted += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    writer.writerow([idx, "", "", "", "", "bad_json", 0])
                    failed += 1
                    continue

                image_url = str(rec.get("image_url") or "").strip()
                poster_file = str(rec.get("poster_file") or "").strip()
                if not image_url or not poster_file:
                    writer.writerow([idx, rec.get("item_idx", ""), rec.get("item_id", ""), image_url, poster_file, "missing_fields", 0])
                    failed += 1
                    continue

                out_path = posters_dir / poster_file
                fut = ex.submit(download_one, image_url, out_path, timeout)
                fut.meta = {
                    "row_idx": idx,
                    "item_idx": rec.get("item_idx", ""),
                    "item_id": rec.get("item_id", ""),
                    "image_url": image_url,
                    "poster_file": poster_file,
                }
                pending.add(fut)

                while len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut_done in done:
                        handle_done(fut_done, writer)

                now = time.time()
                if now - last_report >= 20:
                    elapsed = now - start_ts
                    rate = attempted / elapsed if elapsed > 0 else 0.0
                    log(
                        f"[{category_dir.name}] attempted={attempted} downloaded={downloaded} "
                        f"exists={existed} failed={failed} pending={len(pending)} rate={rate:.2f}/s"
                    )
                    last_report = now

                if write_count >= flush_every:
                    cf.flush()
                    write_count = 0

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut_done in done:
                handle_done(fut_done, writer)
                if write_count >= flush_every:
                    cf.flush()
                    write_count = 0

        cf.flush()

    elapsed = round(time.time() - start_ts, 2)
    stats = {
        "category": category_dir.name,
        "attempted": attempted,
        "newly_downloaded": downloaded,
        "already_exists": existed,
        "failed": failed,
        "bytes_downloaded": bytes_downloaded,
        "elapsed_seconds": elapsed,
        "log_file": str(log_csv),
        "fast_resume_used": bool(resume_fast),
        "resume_skipped_rows": already_logged if resume_fast else 0,
    }

    (category_dir / "image_download_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def parse_categories(raw: str) -> list[str]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in parts if x not in CATEGORY_SPECS]
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}. Choices: {list(CATEGORY_SPECS.keys())}")
    if not parts:
        raise ValueError("No categories selected")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Amazon Review 2018 multimodal datasets (resumable)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/SAMSUNGT7/Dataset_rebuild/Amazon_MM_2018"),
        help="Output root directory",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="All_Beauty,Magazine_Subscriptions",
        help="Comma-separated category names",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--pending-factor", type=int, default=8)
    parser.add_argument("--max-items-per-category", type=int, default=0)
    parser.add_argument("--skip-download-images", action="store_true")
    parser.add_argument("--only-download-images", action="store_true")
    parser.add_argument("--resume-fast", action="store_true", default=True)
    parser.add_argument("--no-resume-fast", dest="resume_fast", action="store_false")
    args = parser.parse_args()

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    categories = parse_categories(args.categories)

    build_all_stats: dict[str, Any] = {
        "output_root": str(output_root),
        "categories": categories,
        "build": {},
        "image_download": {},
        "generated_at_unix": int(time.time()),
    }

    if not args.only_download_images:
        for name in categories:
            spec = CATEGORY_SPECS[name]
            log(f"===== Build {name} =====")
            stats = build_category_dataset(spec, output_root)
            build_all_stats["build"][name] = stats
            log(f"[{name}] build done: items={stats['items']} ratings={stats['ratings_after_multimodal_filter']}")

    if not args.skip_download_images:
        for name in categories:
            category_dir = output_root / name
            log(f"===== Download images {name} =====")
            stats = download_images_for_category(
                category_dir=category_dir,
                workers=max(1, int(args.workers)),
                timeout=max(1, int(args.timeout)),
                pending_factor=max(1, int(args.pending_factor)),
                max_items=max(0, int(args.max_items_per_category)),
                resume_fast=bool(args.resume_fast),
            )
            build_all_stats["image_download"][name] = stats
            log(f"[{name}] image download done: downloaded={stats['newly_downloaded']} failed={stats['failed']}")

    (output_root / "dataset_build_stats.json").write_text(
        json.dumps(build_all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = output_root / "README.md"
    readme.write_text(
        "# Amazon MM 2018 (All_Beauty + Magazine_Subscriptions)\n\n"
        "Generated from Amazon Review 2018 resources.\n\n"
        "Per-category files:\n"
        "- ratings.dat / users.dat / movies.dat\n"
        "- movies_details_clean.csv\n"
        "- posters/\n"
        "- items_with_title_desc_image.jsonl/.csv\n"
        "- raw/* source files\n"
        "- download_log.csv\n",
        encoding="utf-8",
    )

    log(f"Done. Output root: {output_root}")


if __name__ == "__main__":
    main()
