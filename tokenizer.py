from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import regex as re  # 注意：不是内置 re，而是第三方 regex

# GPT-2 / tiktoken 使用的预分词正则（按题目给的原样复制）
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    "去重但保留原顺序（避免special_tokens里重复导致ID不确定）"
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    按 special tokens 把文本切开，返回普通文本片段列表。
    保证后续预分词和 BPE merge 不会跨special token边界发生
    """
    if not special_tokens:
        return [text]


def _iter_pretokens(text: str):
    "用PAT对文本做预分词，逐个 yield pre-token 字符串 （用finditer省内存）"
    for m in re.finditer(PAT, text):
        yield m.group(0)


def _bytes_to_token_tuple(b: bytes) -> Tuple[bytes, ...]:
    """
    把一个 bytes 串变成 “初始 token 序列”（每个 token 是单字节 bytes）。
    例如 b"the" -> (b"t", b"h", b"e")
    """
    return tuple(bytes([x]) for x in b)


def _get_pair_counts(
        word_counts: Dict[Tuple[bytes, ...], int]
) -> Dict[Tuple[bytes, bytes], int]:
    """
    从“词(=token序列)->出现次数”的表中统计相邻 token 对的频率。
    注意：这里只统计 pre-token 内部的相邻对，不会跨 pre-token 边界。
    """
    pair_counts: DefaultDict[Tuple[bytes, bytes], int] = defaultdict(int)   #字节二元组+int次数

    for tok_seq, freq in word_counts.items():          #token序列和语料里出现次数
        if len(tok_seq) < 2:                           #无相邻对
            continue
        # tok_seq = (t0,t1,t2,...)
        for i in range(len(tok_seq) - 1):              #序列长度为n,相邻对n-1
            pair = (tok_seq[i], tok_seq[i + 1])        #取出前后两个token组成二元组
            pair_counts[pair] += freq                  #每个pair的次数等于语料里词汇的出现次数

    return dict(pair_counts)


def _select_best_pair(pair_counts: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes] | None:
    """
    选择要 merge 的 pair：
    - 先选频率最大
    - 频率平局时选“字典序更大”的 pair（题目要求 deterministic tie-break）
    """
    if not pair_counts:                                                           #字典为空
        return None

    # max 的 key 用 (frequency, pair)：
    # 频率越大越好；平局时 pair 越大越好（Python tuple/bytes 都支持字典序比较）
    best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))        #pair_counts.keys()得到所有 pair 的集合视图（可迭代）每个元素是 p = (bytes, bytes)，max 会遍历 iterable

    # 如果最大频率为 0（理论上不会发生，但做个保险）
    if pair_counts[best_pair] <= 0:
        return None
    return best_pair

def _merge_sequence(seq: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
    """
    在一个 token 序列里，把所有相邻出现的 (A,B) 合并成 (AB)。
    例如 seq=(b"t",b"h",b"e"), pair=(b"t",b"h") -> (b"th", b"e")
    """
    a, b = pair
    out: List[bytes] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(a + b)  # bytes 拼接得到新 token
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练一个 byte-level BPE tokenizer。

    返回：
    - vocab: dict[int, bytes]  (token_id -> token_bytes)
    - merges: list[tuple[bytes, bytes]]  (按生成顺序记录每次合并的 token 对)
    """
    # ---------- 0) 参数与 special tokens 预处理 ----------
    special_tokens = _dedupe_preserve_order(special_tokens)

    base_vocab_size = len(special_tokens) + 256
    if vocab_size < base_vocab_size:
        raise ValueError(
            f"vocab_size={vocab_size} 太小：至少需要 len(special_tokens)+256 = {base_vocab_size}"
        )

    # ---------- 1) 初始化 vocab：special tokens + 256 bytes ----------
    vocab: Dict[int, bytes] = {}
    next_id = 0

    # 1.1 special tokens：固定 ID（先放，符合题目示例：<|endoftext|> 在最前）
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    # 1.2 256 个单字节 token：b"\x00" ... b"\xff"
    for byte_val in range(256):
        vocab[next_id] = bytes([byte_val])
        next_id += 1

    # ---------- 2) 预分词 + 计数：构建 word_counts ----------
    # word_counts: dict[tuple[bytes], int]
    word_counts: DefaultDict[Tuple[bytes, ...], int] = defaultdict(int)

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        corpus = f.read()

    # 2.1 先按 special tokens 切开，保证不跨边界 merge
    for segment in _split_on_special_tokens(corpus, special_tokens):
        if not segment:
            continue

        # 2.2 对每个 segment 用 PAT 做预分词
        for pre_tok in _iter_pretokens(segment):
            if not pre_tok:
                continue
            b = pre_tok.encode("utf-8")
            seq = _bytes_to_token_tuple(b)  # 初始 token 序列（按字节拆）
            word_counts[seq] += 1

    # ---------- 3) 反复做 BPE merges ----------
    merges: List[Tuple[bytes, bytes]] = []

    # 允许的 merge 次数 = vocab_size - 初始 vocab 大小
    max_merges = vocab_size - len(vocab)

    for _ in range(max_merges):
        pair_counts = _get_pair_counts(word_counts)
        best_pair = _select_best_pair(pair_counts)
        if best_pair is None:
            break

        a, b = best_pair
        merges.append(best_pair)

        # 新 token = a+b，加到 vocab
        vocab[next_id] = a + b
        next_id += 1

        # 把所有 pre-token 序列里出现的 (a,b) 合并
        new_word_counts: DefaultDict[Tuple[bytes, ...], int] = defaultdict(int)
        for seq, freq in word_counts.items():
            merged_seq = _merge_sequence(seq, best_pair)
            new_word_counts[merged_seq] += freq
        word_counts = new_word_counts

    return vocab, merges
