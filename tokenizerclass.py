from __future__ import annotations  # 让类型注解可以用 "Tokenizer" 这类前向引用（Python 3.10+ 也可省）

import json   # 用于读取/写入 vocab.json, merges.json
import pickle # 用于读取/写入 vocab.pkl, merges.pkl
from typing import Dict, Iterable, Iterator, List, Tuple, Optional  # 类型注解

import regex as re  # 注意：第三方 regex 包（不是内置 re），支持 \p{L} / \p{N} 等 Unicode 类别


# GPT-2 风格预分词正则（与你训练代码一致）
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# 解释：
# - 这个 PAT 会把文本切成“pre-token”（例如：'Hello', ' world', '!'）
# - 特点：很多 token 会把“前导空格”一起吃进去（例如 ' accomplishment'）


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """去重但保持原顺序，避免重复 special token 导致 id 不稳定。"""
    seen = set()        # 用 set 做 O(1) 查询
    out: List[str] = [] # 输出列表
    for x in items:     # 逐个扫描
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_special_split_regex(special_tokens: List[str]) -> Optional[re.Pattern]:
    """
    构造一个用于 split 的 regex，使得 split 之后 special token 自己也会出现在结果里。
    关键点：用括号 () 捕获组。
      re.split(r'(<A>|<B>)', text) -> 会把分隔符 <A>/<B> 也保留在返回列表中
    """
    if not special_tokens:        # 如果没有 special tokens
        return None               # 返回 None，后面就不做 special 切分

    # re.escape：把 special token 里的特殊字符转义，避免破坏正则语义
    # 用 | 拼接多个 token：(<|endoftext|>|<|pad|>|...)
    pat = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
    return re.compile(pat)        # 编译成 regex Pattern，后面反复使用更快


def _bytes_to_initial_tokens(b: bytes) -> List[bytes]:
    """
    byte-level BPE 的“初始 token”是每个单字节：
    b"hi" -> [b"h", b"i"]
    """
    return [bytes([x]) for x in b]


def _get_pairs(seq: List[bytes]) -> List[Tuple[bytes, bytes]]:
    """
    把 token 序列的相邻对列出来：
    [t0,t1,t2] -> [(t0,t1),(t1,t2)]
    """
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def _merge_all_occurrences(seq: List[bytes], a: bytes, b: bytes) -> List[bytes]:
    """
    在序列 seq 中，把所有相邻 (a,b) 合并成 (a+b)。
    注意：一次线性扫描就能把所有出现位置都合并掉。
    """
    out: List[bytes] = []    # 新序列
    i = 0                    # 指针
    L = len(seq)             # 序列长度

    while i < L:             # 线性扫描
        if i + 1 < L and seq[i] == a and seq[i + 1] == b:
            out.append(a + b) # 合并成新 token（bytes 拼接）
            i += 2            # 跳过两个旧 token
        else:
            out.append(seq[i]) # 不合并，原样保留
            i += 1

    return out


class Tokenizer:
    """
    Byte-level BPE Tokenizer（与你训练代码的 vocab/merges 格式兼容）

    vocab: dict[int, bytes]            # id -> token_bytes
    merges: list[tuple[bytes, bytes]]  # merge 规则（训练顺序就是 rank 顺序）
    special_tokens: list[str]          # 用户额外指定的 special tokens（文本形式）
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        # 1) 保存原始 vocab/merges（防止外部传进来的 dict/list 被改动）
        self.vocab: Dict[int, bytes] = dict(vocab)
        self.merges: List[Tuple[bytes, bytes]] = list(merges)

        # 2) 构造反向词表：bytes -> id（encode 时需要从 token_bytes 查 id）
        self.byte_to_id: Dict[bytes, int] = {}
        for i, b in self.vocab.items():
            # 如果 vocab 里存在重复的 bytes，这里会覆盖；正常训练不会出现
            self.byte_to_id[b] = i

        # 3) 处理 special tokens：去重并保持顺序
        self.special_tokens: List[str] = _dedupe_preserve_order(list(special_tokens or []))

        # 4) special token 不在 vocab 时，追加到 vocab 尾部（确保 encode 能识别它）
        for st in self.special_tokens:
            b = st.encode("utf-8")             # special token 的 bytes 表示
            if b not in self.byte_to_id:       # 如果 vocab 还没有这个 bytes
                new_id = (max(self.vocab.keys()) + 1) if self.vocab else 0
                self.vocab[new_id] = b         # 追加到 vocab
                self.byte_to_id[b] = new_id    # 同步反向表

        # 5) 构造 special 切分正则（用来把文本分成 普通片段 / special token）
        self._special_split_re = _build_special_split_regex(self.special_tokens)

        # 6) 预分词正则（PAT），后续 encode 会对普通片段做 finditer
        self._pretok_re = re.compile(PAT)

        # 7) merge ranks：pair -> rank（rank 越小越先合并）
        self._merge_ranks: Dict[Tuple[bytes, bytes], int] = {}
        for r, pair in enumerate(self.merges):
            self._merge_ranks[pair] = r

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        从你训练脚本输出的文件里恢复 Tokenizer。
        支持：
          - vocab.pkl / merges.pkl（pickle 直接存 bytes，最稳）
          - vocab.json / merges.json（latin1 可逆编码）
        """

        def load_vocab(path: str) -> Dict[int, bytes]:
            # 1) pickle：直接 load 后就是 dict[int, bytes]
            if path.endswith(".pkl") or path.endswith(".pickle"):
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                return {int(k): v for k, v in obj.items()}

            # 2) json：训练时存成 { "id": latin1_str }，这里要还原回 bytes
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            out: Dict[int, bytes] = {}
            for k, v in obj.items():
                out[int(k)] = v.encode("latin1")  # latin1: 0..255 一一对应，保证可逆
            return out

        def load_merges(path: str) -> List[Tuple[bytes, bytes]]:
            # 1) pickle：直接 list[(bytes,bytes)]
            if path.endswith(".pkl") or path.endswith(".pickle"):
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                return [(a, b) for (a, b) in obj]

            # 2) json：训练时存成 [[latin1_str_a, latin1_str_b], ...]
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return [(a.encode("latin1"), b.encode("latin1")) for (a, b) in obj]

        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _bpe(self, token_bytes: bytes) -> List[bytes]:
        """
        对“一个 pre-token（bytes）”做 BPE 合并。
        流程：
          - 初始：单字节序列 [b0,b1,...]
          - 循环：
              1) 列出所有相邻 pair
              2) 找到 rank 最小（最先合并）的 pair
              3) 把该 pair 在序列中所有出现位置合并
              4) 重复直到没有可合并 pair
        """
        seq = _bytes_to_initial_tokens(token_bytes)  # bytes -> [单字节bytes,...]
        if len(seq) < 2:
            return seq

        while True:
            pairs = _get_pairs(seq)      # 当前所有相邻对
            best_rank = None             # 当前最优 rank（越小越好）
            best_pair = None             # 当前最优 pair

            # 遍历所有 pairs，找 rank 最小的那个
            for p in pairs:
                r = self._merge_ranks.get(p)  # 如果 p 不在 merges 里，r=None
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = p

            # 若没有任何 pair 在 merges 中，说明无法继续合并
            if best_pair is None:
                break

            a, b = best_pair
            seq = _merge_all_occurrences(seq, a, b)  # 合并该 pair 的所有出现位置

            if len(seq) < 2:  # 序列长度 < 2 时，不可能再有 pair
                break

        return seq

    def encode(self, text: str) -> List[int]:
        """
        编码文本为 token ids。
        关键点：special tokens 作为“特殊情况”在 BPE merges 之前处理：
          1) 先 split：把 special token 单独切出来并保留
          2) special token -> 直接映射成一个 id（不参与 BPE）
          3) 普通文本段 -> PAT 预分词 -> 每个 pre-token 做 _bpe -> 查 id
        """
        if not text:
            return []

        out_ids: List[int] = []

        # 1) special split：如果没有 special tokens 就不切
        if self._special_split_re is None:
            parts = [text]
        else:
            parts = self._special_split_re.split(text)  # split 会保留捕获组（special token）

        # 2) 逐段处理（每段要么是 special token，要么是普通文本）
        for part in parts:
            if part == "":
                continue

            # 2.1) 如果 part 本身就是 special token：直接编码成一个 id
            if part in set(self.special_tokens):
                b = part.encode("utf-8")
                out_ids.append(self.byte_to_id[b])
                continue

            # 2.2) 普通文本：PAT 预分词
            for m in self._pretok_re.finditer(part):
                s = m.group(0)       # 一个 pre-token 字符串（可能含前导空格）
                if not s:
                    continue

                b = s.encode("utf-8") # pre-token -> bytes
                pieces = self._bpe(b) # bytes 做 BPE 合并 -> 若干 token_bytes

                # 2.3) 每个 token_bytes 查到对应的 id
                for pb in pieces:
                    try:
                        out_ids.append(self.byte_to_id[pb])
                    except KeyError as e:
                        # 如果某个 token_bytes 不在 vocab，说明 vocab/merges 不匹配或文件读错
                        raise KeyError(
                            f"Token bytes not found in vocab: {pb!r}. "
                            "Did you load matching vocab/merges?"
                        ) from e

        return out_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        流式编码：给一个可迭代的字符串序列（例如文件句柄，每次读一行），
        返回一个 generator，边读边 yield token ids，避免一次性把大文件读到内存。
        """
        for text in iterable:     # 每次拿到一段文本（如一行）
            ids = self.encode(text)
            for i in ids:
                yield i

    def decode(self, ids: List[int]) -> str:
        """
        解码：ids -> bytes -> 拼接 -> utf-8 解码。
        注意：utf-8 decode(errors="replace") 会把非法字节替换成 �（replacement char）。
        """
        if not ids:
            return ""

        bs: List[bytes] = []
        for i in ids:
            if i not in self.vocab:
                raise KeyError(f"Unknown token id: {i}")
            bs.append(self.vocab[i])

        data = b"".join(bs)  # bytes 串拼接
        return data.decode("utf-8", errors="replace")

