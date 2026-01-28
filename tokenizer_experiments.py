# tokenizer_experiments.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Tuple
import multiprocessing as mp
import numpy as np

from tokenizerclass import Tokenizer  # 你实现的 Tokenizer 类


# -----------------------------
# Document iterators (streaming)
# -----------------------------

def iter_docs_delimited_bytes(
    path: str,
    delimiter: bytes,
    chunk_size: int = 8 * 1024 * 1024,
) -> Iterator[bytes]:
    """从二进制文件中按 delimiter(不包含)流式切分文档，yield 每个 doc 的原始 bytes。"""
    if not delimiter:
        raise ValueError("delimiter must be non-empty bytes")

    with open(path, "rb") as f:
        buf = b""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            while True:
                idx = buf.find(delimiter)
                if idx == -1:
                    break
                doc = buf[:idx]
                yield doc
                buf = buf[idx + len(delimiter):]
        if buf:
            yield buf


def iter_docs_one_per_line(path: str) -> Iterator[bytes]:
    """每行一个文档（去掉行尾换行），yield 原始 bytes。"""
    with open(path, "rb") as f:
        for line in f:
            line = line.rstrip(b"\r\n")
            yield line


def iter_docs_jsonl(path: str, field: str = "text") -> Iterator[bytes]:
    """jsonl，每行一个 JSON，取 obj[field] 作为文档文本，yield UTF-8 bytes。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            s = obj.get(field, "")
            if not isinstance(s, str):
                s = str(s)
            yield s.encode("utf-8")


def get_doc_iterator(
    path: str,
    mode: str,
    *,
    delimiter_str: Optional[str] = None,
    jsonl_field: str = "text",
) -> Iterator[bytes]:
    """
    mode:
      - eot:      用 delimiter_str 作为分隔符（比如 "<|endoftext|>"）
      - line:     每行一个 doc
      - jsonl:    json lines，取 jsonl_field 字段
      - dbl_nl:   双换行分隔（\n\n）
    """
    if mode == "eot":
        if not delimiter_str:
            raise ValueError("mode=eot requires delimiter_str")
        return iter_docs_delimited_bytes(path, delimiter_str.encode("utf-8"))
    if mode == "line":
        return iter_docs_one_per_line(path)
    if mode == "jsonl":
        return iter_docs_jsonl(path, field=jsonl_field)
    if mode == "dbl_nl":
        return iter_docs_delimited_bytes(path, b"\n\n")
    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# Reservoir sampling (one-pass)
# -----------------------------

def reservoir_sample_docs(
    doc_iter: Iterator[bytes],
    k: int,
    seed: int = 0,
    min_bytes: int = 1,
) -> List[bytes]:
    """对 doc_iter 做水塘抽样，单遍随机采样 k 篇文档。"""
    rng = np.random.default_rng(seed)
    sample: List[bytes] = []
    n = 0
    for doc in doc_iter:
        if doc is None or len(doc) < min_bytes:
            continue
        n += 1
        if len(sample) < k:
            sample.append(doc)
        else:
            j = int(rng.integers(0, n))
            if j < k:
                sample[j] = doc
    return sample


# -----------------------------
# Metrics
# -----------------------------

@dataclass
class CompressionStats:
    avg_bytes_per_token: float
    total_bytes: int
    total_tokens: int
    per_doc: List[Tuple[int, int, float]]  # (bytes, tokens, bytes/token)


def compression_ratio_bytes_per_token(tokenizer: Tokenizer, docs: List[bytes]) -> CompressionStats:
    per_doc: List[Tuple[int, int, float]] = []
    total_b = 0
    total_t = 0

    for db in docs:
        if not db:
            continue
        text = db.decode("utf-8", errors="replace")
        ids = tokenizer.encode(text)
        t = len(ids)
        b = len(db)
        if t == 0:
            continue
        per_doc.append((b, t, b / t))
        total_b += b
        total_t += t

    avg = (total_b / total_t) if total_t > 0 else 0.0
    return CompressionStats(avg, total_b, total_t, per_doc)


@dataclass
class ThroughputStats:
    bytes_processed: int
    tokens_emitted: int
    seconds: float
    bytes_per_sec: float
    tokens_per_sec: float


def measure_throughput(
    tokenizer: Tokenizer,
    doc_iter: Iterator[bytes],
    target_bytes: int,
) -> ThroughputStats:
    processed = 0
    tokens = 0

    t0 = time.perf_counter()
    for db in doc_iter:
        if not db:
            continue
        text = db.decode("utf-8", errors="replace")
        ids = tokenizer.encode(text)
        processed += len(db)
        tokens += len(ids)
        if processed >= target_bytes:
            break
    t1 = time.perf_counter()

    secs = max(1e-9, t1 - t0)
    return ThroughputStats(
        bytes_processed=processed,
        tokens_emitted=tokens,
        seconds=secs,
        bytes_per_sec=processed / secs,
        tokens_per_sec=tokens / secs,
    )


def estimate_time_for_pile(bytes_per_sec: float, pile_gb: float = 825.0, binary_gib: bool = True) -> str:
    """估算 tokenize Pile(825GB) 需要多久，返回可读字符串。"""
    if binary_gib:
        pile_bytes = int(pile_gb * (1024 ** 3))  # GiB
    else:
        pile_bytes = int(pile_gb * (1000 ** 3))  # GB (decimal)

    secs = pile_bytes / max(bytes_per_sec, 1e-9)
    hours = secs / 3600.0
    days = hours / 24.0
    return f"{hours:.2f} hours (~{days:.2f} days)"


# -----------------------------
# (d) Encode corpus -> uint16 .npy (two-pass)
# -----------------------------

def encode_corpus_to_uint16_npy_two_pass(
    tokenizer: Tokenizer,
    doc_iter_factory: Callable[[], Iterator[bytes]],
    out_npy_path: str,
    *,
    show_progress_every_docs: int = 2000,
) -> Tuple[int, int]:
    """
    两遍扫描：
      Pass1: 统计 token 总数
      Pass2: open_memmap 写入 uint16 npy
    返回 (total_tokens, total_bytes)
    """
    # -------- Pass 1: count tokens --------
    total_tokens = 0
    total_bytes = 0
    doc_count = 0

    t0 = time.perf_counter()
    for db in doc_iter_factory():
        if not db:
            continue
        text = db.decode("utf-8", errors="replace")
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_bytes += len(db)
        doc_count += 1
        if show_progress_every_docs and doc_count % show_progress_every_docs == 0:
            dt = time.perf_counter() - t0
            print(f"[Pass1] docs={doc_count:,} tokens={total_tokens:,} bytes={total_bytes:,} elapsed={dt:.1f}s")
    dt1 = time.perf_counter() - t0
    print(f"[Pass1 done] docs={doc_count:,} tokens={total_tokens:,} bytes={total_bytes:,} elapsed={dt1:.1f}s")

    if total_tokens <= 0:
        raise RuntimeError("No tokens produced. Check dataset format / tokenizer.")

    # dtype sanity: vocab size must fit in uint16
    if len(tokenizer.vocab) >= 65536:
        raise ValueError(f"vocab too large for uint16: vocab_size={len(tokenizer.vocab)}")

    # -------- Pass 2: write tokens --------
    os.makedirs(os.path.dirname(out_npy_path) or ".", exist_ok=True)
    arr = np.lib.format.open_memmap(out_npy_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))

    offset = 0
    doc_count2 = 0
    t2 = time.perf_counter()
    for db in doc_iter_factory():
        if not db:
            continue
        text = db.decode("utf-8", errors="replace")
        ids = tokenizer.encode(text)
        n = len(ids)
        if n == 0:
            continue
        if offset + n > total_tokens:
            raise RuntimeError("Pass2 overflow: token count mismatch (data changed between passes?)")
        arr[offset: offset + n] = np.asarray(ids, dtype=np.uint16)
        offset += n
        doc_count2 += 1
        if show_progress_every_docs and doc_count2 % show_progress_every_docs == 0:
            dt = time.perf_counter() - t2
            print(f"[Pass2] docs={doc_count2:,} written={offset:,}/{total_tokens:,} elapsed={dt:.1f}s")

    arr.flush()
    dt2 = time.perf_counter() - t2
    print(f"[Pass2 done] written={offset:,}/{total_tokens:,} elapsed={dt2:.1f}s")

    if offset != total_tokens:
        raise RuntimeError(f"Pass2 wrote {offset} tokens but expected {total_tokens}")

    return total_tokens, total_bytes

# =============================
# (d) Multiprocess encoder: shard -> merge to .npy
# =============================

def compute_chunk_ranges_line(path: str, approx_chunk_bytes: int) -> List[Tuple[int, int]]:
    """按换行对齐切 chunk（OWT 一行一文档最适合）。"""
    size = os.path.getsize(path)
    if size == 0:
        return [(0, 0)]

    ranges: List[Tuple[int, int]] = []
    with open(path, "rb") as f:
        start = 0
        while start < size:
            end = min(size, start + approx_chunk_bytes)
            if end < size:
                f.seek(end)
                f.readline()          # 读到下一行开头
                end = f.tell()
            if end <= start:
                end = min(size, start + approx_chunk_bytes)
            ranges.append((start, end))
            start = end
    return ranges


def compute_chunk_ranges_special_token(
    path: str,
    special_token: str,
    approx_chunk_bytes: int,
    search_window_bytes: int = 8 * 1024 * 1024,
) -> List[Tuple[int, int]]:
    """按 special token 起点对齐切 chunk，避免切断 <|endoftext|>。"""
    size = os.path.getsize(path)
    if size == 0:
        return [(0, 0)]

    sb = special_token.encode("utf-8")
    max_len = len(sb)

    ranges: List[Tuple[int, int]] = []
    with open(path, "rb") as f:
        start = 0
        while start < size:
            rough_end = min(size, start + approx_chunk_bytes)
            if rough_end >= size:
                ranges.append((start, size))
                break

            probe_start = max(0, rough_end - (max_len - 1))
            f.seek(probe_start)
            buf = f.read(search_window_bytes + (max_len - 1))
            local_from = rough_end - probe_start

            local_idx = buf.find(sb, local_from)
            if local_idx == -1:
                ranges.append((start, size))
                break

            end = probe_start + local_idx
            if end <= start:
                end = rough_end

            ranges.append((start, end))
            start = end

    return ranges


# ---- multiprocessing globals ----
_G_TOK: Optional[Tokenizer] = None

def _init_worker(vocab_path: str, merges_path: str, special_tokens: List[str]):
    """每个 worker 启动时加载 tokenizer（Linux/fork 下也稳）。"""
    global _G_TOK
    _G_TOK = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)


def _worker_encode_shard_line(args):
    """mode=line/jsonl：逐行读 + 写 shard.bin（uint16 raw）。"""
    shard_idx, input_path, start, end, shard_path, append_special = args
    assert _G_TOK is not None

    append_id = None
    if append_special:
        tmp = _G_TOK.encode(append_special)
        if len(tmp) != 1:
            raise RuntimeError(f"append_special must map to single id, got {tmp}")
        append_id = int(tmp[0])

    total_tokens = 0
    total_bytes = 0

    with open(input_path, "rb") as f_in, open(shard_path, "wb") as f_out:
        f_in.seek(start)
        while f_in.tell() < end:
            line = f_in.readline()
            if not line:
                break
            # 保持与你 iter_docs_one_per_line 一致：去掉行尾换行
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            total_bytes += len(line)

            text = line.decode("utf-8", errors="replace")
            ids = _G_TOK.encode(text)

            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(f_out)
            total_tokens += arr.size

            if append_id is not None:
                np.asarray([append_id], dtype=np.uint16).tofile(f_out)
                total_tokens += 1

    return shard_idx, total_tokens, total_bytes


def _worker_encode_shard_eot(args):
    """mode=eot：读 chunk -> split(eot) -> 写回 tokens + eot_id。"""
    shard_idx, input_path, start, end, shard_path, eot_str = args
    assert _G_TOK is not None

    eot_ids = _G_TOK.encode(eot_str)
    if len(eot_ids) != 1:
        raise RuntimeError(f"Special token {eot_str} did not encode to single id: {eot_ids}")
    eot_id = int(eot_ids[0])
    eot_bytes = eot_str.encode("utf-8")

    with open(input_path, "rb") as f_in:
        f_in.seek(start)
        raw = f_in.read(end - start)

    total_bytes = len(raw)
    parts = raw.split(eot_bytes)  # split 会吃掉分隔符

    total_tokens = 0
    with open(shard_path, "wb") as f_out:
        # parts[:-1] 后面都对应一个 eot 分隔符
        for p in parts[:-1]:
            if p:
                text = p.decode("utf-8", errors="replace")
                ids = _G_TOK.encode(text)
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(f_out)
                total_tokens += arr.size

            # 写回 eot token
            np.asarray([eot_id], dtype=np.uint16).tofile(f_out)
            total_tokens += 1

        # 最后一段没有尾随 eot
        last = parts[-1]
        if last:
            text = last.decode("utf-8", errors="replace")
            ids = _G_TOK.encode(text)
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(f_out)
            total_tokens += arr.size

    return shard_idx, total_tokens, total_bytes


def encode_corpus_to_uint16_npy_multiprocess(
    *,
    input_path: str,
    out_npy_path: str,
    mode: str,
    delim: Optional[str],
    jsonl_field: str,  # 这里为了接口对齐保留，line 用不到
    vocab_path: str,
    merges_path: str,
    special_tokens: List[str],
    num_workers: int,
    approx_chunk_bytes: int,
    keep_shards: bool = False,
    append_special_every_doc: str = "",  # line/jsonl 模式可选：每行后追加一个 special token（如 <|endoftext|>）
) -> Tuple[int, int]:
    """
    多进程分片编码（每个 worker 写 shard.bin），主进程合并成 .npy。
    返回 (total_tokens, total_bytes)
    """
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1

    # 1) 计算 chunk ranges（必须对齐边界，避免 tokenization 语义变化）
    if mode in ("line", "jsonl"):
        ranges = compute_chunk_ranges_line(input_path, approx_chunk_bytes)
        worker_fn = _worker_encode_shard_line
    elif mode == "eot":
        if not delim:
            raise ValueError("mode=eot requires delim")
        ranges = compute_chunk_ranges_special_token(input_path, delim, approx_chunk_bytes)
        worker_fn = _worker_encode_shard_eot
    else:
        raise ValueError(f"Multiprocess D only supports mode=line/jsonl/eot, got {mode}")

    # 2) shard 临时目录
    tmp_dir = os.path.join(os.path.dirname(out_npy_path) or ".", ".shards_" + os.path.basename(out_npy_path))
    os.makedirs(tmp_dir, exist_ok=True)

    # 3) 组织任务
    tasks = []
    for idx, (s, e) in enumerate(ranges):
        shard_path = os.path.join(tmp_dir, f"shard_{idx:05d}.bin")
        if worker_fn is _worker_encode_shard_line:
            tasks.append((idx, input_path, s, e, shard_path, append_special_every_doc))
        else:
            tasks.append((idx, input_path, s, e, shard_path, delim))

    # 4) 开 pool（Linux 默认 fork；Windows 会 spawn）
    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")

    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path, special_tokens),
    ) as pool:
        results = []
        done = 0
        t0 = time.perf_counter()

        for r in pool.imap_unordered(worker_fn, tasks, chunksize=1):
            results.append(r)
            done += 1

            # 每完成 1 个 shard 就打印一次（你也可以改成每 5/10 个打印）
            if done % 1 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"[D] shards done {done}/{len(tasks)} elapsed={elapsed:.1f}s", flush=True)

    results.sort(key=lambda x: x[0])  # 按 shard_idx 排序，保证顺序一致

    total_tokens = sum(r[1] for r in results)
    total_bytes = sum(r[2] for r in results)
    if total_tokens <= 0:
        raise RuntimeError("No tokens produced; check input format / tokenizer.")

    # 5) 合并 shards -> 最终 .npy
    os.makedirs(os.path.dirname(out_npy_path) or ".", exist_ok=True)
    out_arr = np.lib.format.open_memmap(out_npy_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))

    offset = 0
    for idx, tok_count, _ in results:
        shard_path = os.path.join(tmp_dir, f"shard_{idx:05d}.bin")
        mm = np.memmap(shard_path, dtype=np.uint16, mode="r", shape=(tok_count,))
        out_arr[offset: offset + tok_count] = mm
        offset += tok_count

    out_arr.flush()

    # 6) 清理 shards
    if not keep_shards:
        for idx, _, _ in results:
            os.remove(os.path.join(tmp_dir, f"shard_{idx:05d}.bin"))
        os.rmdir(tmp_dir)

    return total_tokens, total_bytes



# -----------------------------
# Main
# -----------------------------

def main() -> None:
    """
python -u tokenizer_experiments.py --ts_vocab bpe_tinystories_10k/vocab.pkl --ts_merges bpe_tinystories_10k/merges.pkl --owt_vocab bpe_owt_32k/vocab.pkl --owt_merges bpe_owt_32k/merges.pkl --ts_data TinyStories-train.txt --ts_mode eot --ts_delim "<|endoftext|>" --owt_data owt_train.txt --owt_mode line --sample_k 10 --seed 0
python -u tokenizer_experiments.py --ts_vocab bpe_tinystories_10k/vocab.pkl --ts_merges bpe_tinystories_10k/merges.pkl --owt_vocab bpe_owt_32k/vocab.pkl --owt_merges bpe_owt_32k/merges.pkl --ts_mode eot --ts_delim "<|endoftext|>" --owt_mode line --encode_ts_train TinyStories-train.txt --encode_ts_dev TinyStories-dev.txt --encode_owt_train owt_train.txt --encode_owt_dev owt_dev.txt --out_dir tokenized_ids
python -u tokenizer_experiments.py --ts_vocab bpe_tinystories_10k/vocab.pkl --ts_merges bpe_tinystories_10k/merges.pkl --owt_vocab bpe_owt_32k/vocab.pkl --owt_merges bpe_owt_32k/merges.pkl --ts_data TinyStories-train.txt --owt_data owt_train.txt --ts_mode eot --ts_delim "<|endoftext|>" --owt_mode line --encode_ts_train TinyStories-train.txt --encode_ts_dev TinyStories-dev.txt --encode_owt_train owt_train.txt --encode_owt_dev owt_dev.txt --out_dir tokenized_ids
    multiprocessing
python -u tokenizer_experiments.py --ts_vocab bpe_tinystories_10k/vocab.pkl --ts_merges bpe_tinystories_10k/merges.pkl --owt_vocab bpe_owt_32k/vocab.pkl --owt_merges bpe_owt_32k/merges.pkl --ts_mode eot --ts_delim "<|endoftext|>" --owt_mode line --encode_ts_train TinyStories-train.txt --encode_ts_dev TinyStories-dev.txt --encode_owt_train owt_train.txt --encode_owt_dev owt_dev.txt --out_dir tokenized_ids --d_workers 0 --d_chunk_mb 256


    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts_vocab", required=True)
    ap.add_argument("--ts_merges", required=True)
    ap.add_argument("--owt_vocab", required=True)
    ap.add_argument("--owt_merges", required=True)

    ap.add_argument("--ts_data", default=None, help="TinyStories dataset file for sampling/throughput")
    ap.add_argument("--owt_data", default=None, help="OpenWebText dataset file for sampling/throughput")
    ap.add_argument("--d_workers", type=int, default=0, help="(d) worker processes. 0=use all CPU cores")
    ap.add_argument("--d_chunk_mb", type=int, default=256, help="(d) approx chunk size per worker (MB)")
    ap.add_argument("--d_keep_shards", action="store_true", help="keep temporary shard .bin files for debugging")

    ap.add_argument("--ts_mode", default="eot", choices=["eot", "line", "jsonl", "dbl_nl"])
    ap.add_argument("--owt_mode", default="jsonl", choices=["eot", "line", "jsonl", "dbl_nl"])
    ap.add_argument("--ts_delim", default="<|endoftext|>")
    ap.add_argument("--owt_delim", default=None)
    ap.add_argument("--owt_jsonl_field", default="text")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_k", type=int, default=10)

    ap.add_argument("--throughput_bytes", type=int, default=200 * 1024 * 1024, help="bytes to measure throughput on")
    ap.add_argument("--pile_gb", type=float, default=825.0)

    # (d) encoding options
    ap.add_argument("--encode_ts_train", default=None)
    ap.add_argument("--encode_ts_dev", default=None)
    ap.add_argument("--encode_owt_train", default=None)
    ap.add_argument("--encode_owt_dev", default=None)
    ap.add_argument("--out_dir", default="tokenized_out", help="output dir for (d) .npy files")

    # special tokens when loading tokenizers
    ap.add_argument("--ts_special", default="<|endoftext|>", help="comma-separated special tokens for TinyStories tokenizer")
    ap.add_argument("--owt_special", default="", help="comma-separated special tokens for OWT tokenizer (optional)")

    args = ap.parse_args()

    ts_special = [s for s in (args.ts_special.split(",") if args.ts_special else []) if s]
    owt_special = [s for s in (args.owt_special.split(",") if args.owt_special else []) if s]

    ts_tok = Tokenizer.from_files(args.ts_vocab, args.ts_merges, special_tokens=ts_special)
    owt_tok = Tokenizer.from_files(args.owt_vocab, args.owt_merges, special_tokens=owt_special)

    # # -------- (a) sample 10 docs from each + compute compression --------
    # ts_docs = reservoir_sample_docs(
    #     get_doc_iterator(args.ts_data, args.ts_mode, delimiter_str=args.ts_delim, jsonl_field="text"),
    #     k=args.sample_k,
    #     seed=args.seed,
    # )
    # owt_docs = reservoir_sample_docs(
    #     get_doc_iterator(args.owt_data, args.owt_mode, delimiter_str=args.owt_delim, jsonl_field=args.owt_jsonl_field),
    #     k=args.sample_k,
    #     seed=args.seed + 1,
    # )
    #
    # ts_stats = compression_ratio_bytes_per_token(ts_tok, ts_docs)
    # owt_stats = compression_ratio_bytes_per_token(owt_tok, owt_docs)
    #
    # print("\n=== (a) Compression ratio (bytes/token) ===")
    # print(f"TinyStories tokenizer on TinyStories sample: {ts_stats.avg_bytes_per_token:.3f} bytes/token "
    #       f"(bytes={ts_stats.total_bytes:,} tokens={ts_stats.total_tokens:,})")
    # print(f"OWT tokenizer on OWT sample: {owt_stats.avg_bytes_per_token:.3f} bytes/token "
    #       f"(bytes={owt_stats.total_bytes:,} tokens={owt_stats.total_tokens:,})")
    #
    # # -------- (b) tokenize OWT sample with TinyStories tokenizer --------
    # owt_with_ts = compression_ratio_bytes_per_token(ts_tok, owt_docs)
    # print("\n=== (b) OWT sample tokenized with TinyStories tokenizer ===")
    # print(f"TinyStories tokenizer on OWT sample: {owt_with_ts.avg_bytes_per_token:.3f} bytes/token "
    #       f"(bytes={owt_with_ts.total_bytes:,} tokens={owt_with_ts.total_tokens:,})")
    # if owt_with_ts.avg_bytes_per_token < owt_stats.avg_bytes_per_token:
    #     print("Qualitative: tends to produce MORE tokens (lower bytes/token), likely because TinyStories merges/vocab "
    #           "are less suited to web text (URLs, markup, diverse symbols/encodings).")
    # else:
    #     print("Qualitative: compression is similar or better; check samples—could happen if the sample is simple English.")
    #
    # # -------- (c) throughput + estimate Pile time --------
    # print("\n=== (c) Throughput estimate ===")

    # # measure on each dataset's doc iterator (same bytes budget)
    # ts_thr = measure_throughput(
    #     ts_tok,
    #     get_doc_iterator(args.ts_data, args.ts_mode, delimiter_str=args.ts_delim, jsonl_field="text"),
    #     target_bytes=args.throughput_bytes,
    # )
    # owt_thr = measure_throughput(
    #     owt_tok,
    #     get_doc_iterator(args.owt_data, args.owt_mode, delimiter_str=args.owt_delim, jsonl_field=args.owt_jsonl_field),
    #     target_bytes=args.throughput_bytes,
    # )
    #
    # print(f"TinyStories tokenizer throughput: {ts_thr.bytes_per_sec:,.0f} bytes/s "
    #       f"({ts_thr.tokens_per_sec:,.0f} tok/s) over {ts_thr.bytes_processed:,} bytes")
    # print(f"OWT tokenizer throughput: {owt_thr.bytes_per_sec:,.0f} bytes/s "
    #       f"({owt_thr.tokens_per_sec:,.0f} tok/s) over {owt_thr.bytes_processed:,} bytes")
    #
    # pile_time = estimate_time_for_pile(owt_thr.bytes_per_sec, pile_gb=args.pile_gb, binary_gib=True)
    # print(f"Estimated time to tokenize Pile ({args.pile_gb}GB) at OWT throughput: {pile_time}")

    # -------- (d) encode train/dev to uint16 npy --------
    def maybe_encode(split_name: str, data_path: Optional[str], mode: str, delim: Optional[str], jsonl_field: str,
                     tok: Tokenizer, out_path: str) -> None:
        if not data_path:
            return
        print(f"\n=== (d) Encoding {split_name} -> {out_path} ===")

        def factory() -> Iterator[bytes]:
            return get_doc_iterator(data_path, mode, delimiter_str=delim, jsonl_field=jsonl_field)

        approx_chunk_bytes = args.d_chunk_mb * 1024 * 1024

        # 走多进程（d_workers=0 表示用满 CPU；d_workers=1 就单进程）
        if args.d_workers != 1:
            # 注意：多进程 worker 需要知道“从文件加载 tokenizer”，所以这里传 vocab/merges 路径
            # TinyStories 用 ts_vocab/ts_merges；OWT 用 owt_vocab/owt_merges
            if "TinyStories" in split_name:
                vocab_path = args.ts_vocab
                merges_path = args.ts_merges
                sp = ts_special
            else:
                vocab_path = args.owt_vocab
                merges_path = args.owt_merges
                sp = owt_special

            total_tokens, total_bytes = encode_corpus_to_uint16_npy_multiprocess(
                input_path=data_path,
                out_npy_path=out_path,
                mode=mode,
                delim=delim,
                jsonl_field=jsonl_field,
                vocab_path=vocab_path,
                merges_path=merges_path,
                special_tokens=sp,
                num_workers=args.d_workers,  # 0=自动用满
                approx_chunk_bytes=approx_chunk_bytes,
                keep_shards=args.d_keep_shards,
                append_special_every_doc="",  # 如果你想每行后加 <|endoftext|>，这里填 "<|endoftext|>"
            )
        else:
            total_tokens, total_bytes = encode_corpus_to_uint16_npy_two_pass(tok, factory, out_path)
        print(f"[Saved] {out_path}  tokens={total_tokens:,} bytes={total_bytes:,}")

    os.makedirs(args.out_dir, exist_ok=True)

    maybe_encode(
        "TinyStories train",
        args.encode_ts_train,
        args.ts_mode,
        args.ts_delim if args.ts_mode == "eot" else None,
        "text",
        ts_tok,
        os.path.join(args.out_dir, "tinystories_train_uint16.npy"),
    )
    maybe_encode(
        "TinyStories dev",
        args.encode_ts_dev,
        args.ts_mode,
        args.ts_delim if args.ts_mode == "eot" else None,
        "text",
        ts_tok,
        os.path.join(args.out_dir, "tinystories_dev_uint16.npy"),
    )
    maybe_encode(
        "OWT train",
        args.encode_owt_train,
        args.owt_mode,
        args.owt_delim if args.owt_mode == "eot" else None,
        args.owt_jsonl_field,
        owt_tok,
        os.path.join(args.out_dir, "owt_train_uint16.npy"),
    )
    maybe_encode(
        "OWT dev",
        args.encode_owt_dev,
        args.owt_mode,
        args.owt_delim if args.owt_mode == "eot" else None,
        args.owt_jsonl_field,
        owt_tok,
        os.path.join(args.out_dir, "owt_dev_uint16.npy"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

