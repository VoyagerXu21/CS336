from __future__ import annotations
from typing import DefaultDict, Dict, Iterable, Iterator, List, Tuple, Optional


import os
import time
import json
import pickle
import heapq
import multiprocessing as mp
from dataclasses import dataclass
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Iterator, List, Tuple

import regex as re  # 注意：不是内置 re，必须用 regex 包（支持 \p{L} 等）


# GPT-2 / tiktoken 的预分词正则（题目给的原样）
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ---------------------------
# 工具：特殊 token 处理
# ---------------------------

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """去重但保留顺序，避免 special_tokens 重复导致词表 ID 不确定。"""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_special_split_pattern(special_tokens: List[str]) -> re.Pattern | None:
    """
    构造一个用于 split 的正则：
    - 用 re.escape 防止特殊字符影响正则
    - 用捕获组保留分隔符也行，但本题我们直接丢掉 special token（不参与 merges）
    """
    if not special_tokens:
        return None
    pat = "|".join(re.escape(t) for t in special_tokens)
    return re.compile(pat)


def _split_on_special_tokens(text: str, split_pat: re.Pattern | None) -> List[str]:
    """按 special token 切分文本，返回普通文本片段列表。"""
    if split_pat is None:
        return [text]
    return split_pat.split(text)


# ---------------------------
# 工具：把 bytes 串变成初始 token 序列（每个 token 是单字节 bytes）
# ---------------------------

def _bytes_to_token_tuple(b: bytes) -> Tuple[bytes, ...]:
    return tuple(bytes([x]) for x in b)


# ---------------------------
# 并行预分词：切块边界（保证在 special token 起始处）
# ---------------------------

def _find_next_special(data: bytes, start_at: int, specials_b: List[bytes]) -> Optional[int]:
    """
    在 data 这个 bytes buffer 里，从 start_at 开始向后找任意 special token 的最早出现位置。
    - data: 我们读入的一段文件内容（bytes）
    - start_at: 从 data 的哪个索引开始找
    - specials_b: special tokens 的 bytes 形式列表，比如 [b"<|endoftext|>"]
    返回：
    - 找到：返回“最早出现的 special token 起点”在 data 里的索引（局部索引）
    - 找不到：返回 None
    """
    best = None
    for sb in specials_b:
        idx = data.find(sb, start_at)  # bytes.find：找子串 sb 在 data 里的首次出现位置
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


def compute_chunk_ranges(
        input_path: str,
        special_tokens: List[str],
        approx_chunk_bytes: int = 64 * 1024 * 1024,  # 目标：每块大概 64MB
        search_window_bytes: int = 8 * 1024 * 1024,  # 为了找“下一个 eot”，rough_end 后再额外往后看 8MB
) -> List[Tuple[int, int]]:
    """
    目的：把一个超大的 txt 文件按 (start,end) 字节区间切成多个 chunk，供多进程并行预分词。

    关键要求：chunk 的边界不能把 special token（如 <|endoftext|>）切断。
    - 如果切断，worker split 就失败，文档边界会被破坏，可能出现跨文档合并/统计。

    做法：
    - 先设一个粗略切点 rough_end = start + approx_chunk_bytes
    - 但 rough_end 可能落在 special token 中间，所以：
        1) 往前回退 (max_len-1) 个字节作为 probe_start，保证 buffer 里能看见完整 token
        2) 从 probe_start 开始读一个窗口 buf
        3) 在 buf 中，从 local_from(=rough_end-probe_start) 开始找“rough_end 之后”的下一个 special token 起点
        4) 把 chunk 的 end 对齐到这个起点
    """
    file_size = os.path.getsize(input_path)
    if file_size == 0:
        return [(0, 0)]

    # 把 special token 转成 bytes：因为我们要在文件的 bytes 流里查找它
    specials_b = [s.encode("utf-8") for s in special_tokens]

    # 如果没有 special token，就退化成简单的固定大小切块
    if not specials_b:
        ranges = []
        start = 0
        while start < file_size:
            end = min(file_size, start + approx_chunk_bytes)
            ranges.append((start, end))
            start = end
        return ranges

    # special token 的最长字节长度。TinyStories 的 <|endoftext|> 长度是 13
    max_len = max(len(x) for x in specials_b)

    ranges: List[Tuple[int, int]] = []

    with open(input_path, "rb") as f:
        start = 0  # 当前 chunk 的起点（文件字节偏移）
        while start < file_size:
            # 1) 粗略设一个“想切的位置”
            rough_end = min(file_size, start + approx_chunk_bytes)

            # 如果 rough_end 已经到文件尾，那最后一个 chunk 直接收尾
            if rough_end >= file_size:
                ranges.append((start, file_size))
                break

            # 2) 为了避免 special token 被切断：probe_start 往前回退 (max_len-1)
            #    为什么是 max_len-1？
            #    - 最坏情况：rough_end 落在 token 的最后一个字节前
            #    - 你往前退 max_len-1 就一定能把 token 的开头包含进 buf
            probe_start = max(0, rough_end - (max_len - 1))

            # 3) 从 probe_start 开始读 buf
            #    读多大？
            #    - 至少要覆盖 rough_end 之后的一段 search_window_bytes 才能找到“下一个 token 起点”
            #    - 同时加上 (max_len-1) 是为了覆盖可能跨边界的 token
            f.seek(probe_start)
            buf = f.read(search_window_bytes + (max_len - 1))

            # 4) 计算 rough_end 在 buf 里的位置（局部索引）
            #    - buf[0] 对应文件 probe_start
            #    - buf[local_from] 对应文件 rough_end
            local_from = rough_end - probe_start

            # 5) 在 buf 中从 local_from 开始找 special token 起点
            #    这一步的“含义”是：只找 rough_end 之后的下一个 token，
            #    从而跳过 probe_start 回退区间里可能存在的“rough_end 之前的 token”
            local_idx = _find_next_special(buf, local_from, specials_b)

            if local_idx is None:
                # 6) 如果在窗口里没找到 token（可能是窗口太小 or 后面很久才出现 token）
                #    为了保证正确性（不把 token 切断），保守做法：这个 chunk 直接延伸到文件末尾
                ranges.append((start, file_size))
                break

            # 7) 把 end 对齐到“下一个 special token 的起点”（文件全局索引）
            end = probe_start + local_idx

            # 8) 防御性：避免 end <= start 导致死循环（理论上不该发生）
            if end <= start:
                end = rough_end

            ranges.append((start, end))

            # 9) 更新 start，准备切下一个 chunk
            start = end

    return ranges


# ---------------------------
# Worker：对一个 chunk 做预分词并统计 word_counts
# ---------------------------

def _pretokenize_chunk_worker(args) -> Dict[Tuple[bytes, ...], int]:
    """
    Worker 进程：
    - 读取文件的 [start,end) 字节
    - 解码为文本（errors=replace）
    - 按 special token 切分（special token 不参与 merges）
    - 对每段普通文本做 regex 预分词
    - 把每个 pre-token 编码成 UTF-8 bytes，再拆成单字节 token 序列，累计计数
    """
    input_path, start, end, special_tokens = args
    split_pat = _build_special_split_pattern(special_tokens)

    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    text = raw.decode("utf-8", errors="replace")
    segments = _split_on_special_tokens(text, split_pat)

    local_counts: DefaultDict[Tuple[bytes, ...], int] = defaultdict(int)

    for seg in segments:
        if not seg:
            continue
        # regex.finditer：节省内存
        for m in re.finditer(PAT, seg):
            s = m.group(0)
            if not s:
                continue
            b = s.encode("utf-8")
            seq = _bytes_to_token_tuple(b)
            local_counts[seq] += 1

    return dict(local_counts)


def build_word_counts_parallel(
    input_path: str,
    special_tokens: List[str],
    num_workers: int | None = None,
    approx_chunk_bytes: int = 64 * 1024 * 1024,
) -> Dict[Tuple[bytes, ...], int]:
    """
    并行预分词 + 统计，输出 word_counts（pre-token 字节序列 -> 频次）
    """
    special_tokens = _dedupe_preserve_order(special_tokens)

    ranges = compute_chunk_ranges(
        input_path,
        special_tokens=special_tokens,
        approx_chunk_bytes=approx_chunk_bytes,
    )

    if num_workers is None:
        num_workers = max(1, os.cpu_count() or 1)

    ctx = mp.get_context("spawn")  # Windows 兼容
    tasks = [(input_path, s, e, special_tokens) for (s, e) in ranges]

    merged: DefaultDict[Tuple[bytes, ...], int] = defaultdict(int)

    # chunksize 取一个经验值，减少进程调度开销
    chunksize = max(1, len(tasks) // (num_workers * 4) or 1)

    with ctx.Pool(processes=num_workers) as pool:
        for local in pool.imap_unordered(_pretokenize_chunk_worker, tasks, chunksize=chunksize):
            for seq, c in local.items():
                merged[seq] += c

    return dict(merged)


# ---------------------------
# BPE merges：更快实现（pair_counts + pair_to_words + heap + lazy invalidation）
# ---------------------------

@dataclass(frozen=True)
class _RevPair:
    """
    让 heap 在频率相同的时候选择“字典序更大”的 pair。
    heapq 是最小堆，默认会选“更小”的 pair。
    我们把比较反过来：字典序更大 => 被认为“更小”，这样就会优先 pop 出来。
    """
    pair: Tuple[bytes, bytes]

    def __lt__(self, other: "_RevPair") -> bool:
        return self.pair > other.pair  # 反向比较


def _merge_sequence_list(seq: List[bytes], a: bytes, b: bytes) -> List[bytes]:
    """在一个 token 序列中把所有相邻 (a,b) 合并为 (a+b)。"""
    out: List[bytes] = []
    i = 0
    L = len(seq)
    while i < L:
        if i + 1 < L and seq[i] == a and seq[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_workers: int | None = None,
    approx_chunk_bytes: int = 64 * 1024 * 1024,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练 byte-level BPE tokenizer：
    - 并行预分词得到 word_counts
    - 用优化的 merges 过程训练到 vocab_size
    """
    special_tokens = _dedupe_preserve_order(special_tokens)

    # ---------- 1) 初始化 vocab：special tokens + 256 bytes ----------
    base_vocab_size = len(special_tokens) + 256
    if vocab_size < base_vocab_size:
        raise ValueError(
            f"vocab_size={vocab_size} 太小：至少需要 len(special_tokens)+256 = {base_vocab_size}"
        )

    vocab: Dict[int, bytes] = {}
    next_id = 0

    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    for byte_val in range(256):
        vocab[next_id] = bytes([byte_val])
        next_id += 1

    # ---------- 2) 并行预分词：构建 word_counts ----------
    word_counts = build_word_counts_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_workers=num_workers,
        approx_chunk_bytes=approx_chunk_bytes,
    )

    # 把 word_counts 转成“可更新”的 words/freq 结构（给 merge 用）
    words: List[List[bytes]] = []
    freqs: List[int] = []

    for seq, c in word_counts.items():
        words.append(list(seq))
        freqs.append(int(c))

    # ---------- 3) 初始化 pair_counts 与 pair_to_words ----------
    pair_counts: DefaultDict[Tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: DefaultDict[Tuple[bytes, bytes], set[int]] = defaultdict(set)

    for wid, seq in enumerate(words):
        if len(seq) < 2:
            continue
        local_pairs = set()
        f = freqs[wid]
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            pair_counts[p] += f
            local_pairs.add(p)
        for p in local_pairs:
            pair_to_words[p].add(wid)

    # ---------- 4) 用 heap 维护“当前最优 pair”（lazy 丢弃过期项） ----------
    heap: List[Tuple[int, _RevPair]] = []
    for p, cnt in pair_counts.items():
        heapq.heappush(heap, (-cnt, _RevPair(p)))

    merges: List[Tuple[bytes, bytes]] = []
    max_merges = vocab_size - len(vocab)

    for _ in range(max_merges):
        # 找到堆顶有效项
        while heap:
            neg_cnt, rp = heap[0]
            p = rp.pair
            curr = pair_counts.get(p, 0)
            if -neg_cnt != curr:
                # 过期项，丢掉
                heapq.heappop(heap)
                continue
            if curr <= 0:
                heap = []
                break
            break

        if not heap:
            break

        # 取当前最频繁 pair（平局时字典序更大）
        best_cnt_neg, best_rp = heapq.heappop(heap)
        a, b = best_rp.pair
        merges.append((a, b))

        # 新 token 入 vocab
        vocab[next_id] = a + b
        next_id += 1

        # 只更新“包含该 pair 的词”
        affected = list(pair_to_words.get((a, b), set()))
        if not affected:
            # 理论上不应发生（因为 pair_counts>0），但做防御
            pair_counts[(a, b)] = 0
            continue

        # 合并后，这个 pair 会消失（被替换）
        pair_to_words[(a, b)].clear()
        pair_counts[(a, b)] = 0

        changed_pairs: set[Tuple[bytes, bytes]] = set()

        for wid in affected:
            old_seq = words[wid]
            f = freqs[wid]
            if len(old_seq) < 2:
                continue

            # 1) 从全局统计里减去 old_seq 的所有相邻对贡献
            old_pairs_in_word = set()
            for i in range(len(old_seq) - 1):
                p = (old_seq[i], old_seq[i + 1])
                pair_counts[p] -= f
                old_pairs_in_word.add(p)
            for p in old_pairs_in_word:
                if wid in pair_to_words[p]:
                    pair_to_words[p].remove(wid)
            changed_pairs |= old_pairs_in_word

            # 2) 生成新序列（合并 (a,b)）
            new_seq = _merge_sequence_list(old_seq, a, b)
            words[wid] = new_seq

            # 3) 把 new_seq 的相邻对贡献加回去
            if len(new_seq) >= 2:
                new_pairs_in_word = set()
                for i in range(len(new_seq) - 1):
                    p = (new_seq[i], new_seq[i + 1])
                    pair_counts[p] += f
                    new_pairs_in_word.add(p)
                for p in new_pairs_in_word:
                    pair_to_words[p].add(wid)
                changed_pairs |= new_pairs_in_word

        # 把受影响的 pairs 的新计数 push 进 heap（lazy：旧的会在弹出时丢弃）
        for p in changed_pairs:
            cnt = pair_counts.get(p, 0)
            if cnt > 0:
                heapq.heappush(heap, (-cnt, _RevPair(p)))

    return vocab, merges


# ---------------------------
# 序列化保存（方便作业检查）
# ---------------------------

def save_vocab_merges(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) pickle：最省事、可逆、保存 bytes 最稳
    with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(out_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

    # 2) json（可读）：用 latin1 保证 bytes 可逆
    vocab_json = {str(i): b.decode("latin1") for i, b in vocab.items()}
    merges_json = [[a.decode("latin1"), b.decode("latin1")] for a, b in merges]

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "merges.json"), "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False)


# ---------------------------
# 运行示例（TinyStories / OWT）
# ---------------------------

def _find_longest_token(vocab: Dict[int, bytes]) -> Tuple[int, bytes]:
    return max(vocab.items(), key=lambda kv: len(kv[1]))


if __name__ == "__main__":
    # 你自己改路径：TinyStories txt 路径
    input_path = "owt_train.txt"
    out_dir = "bpe_owt_32k"

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=[],
        num_workers=None,            # None = 用满 CPU
        approx_chunk_bytes=64 * 1024 * 1024,
    )
    t1 = time.perf_counter()

    save_vocab_merges(vocab, merges, out_dir)

    longest_id, longest_bytes = _find_longest_token(vocab)
    print(f"Training done. elapsed={(t1-t0)/60:.2f} min")
    print(f"vocab size={len(vocab)}, merges={len(merges)}")
    print(f"longest token id={longest_id}, bytes_len={len(longest_bytes)}")
    print("longest token (utf8, replace):", longest_bytes.decode("utf-8", errors="replace")[:200])
