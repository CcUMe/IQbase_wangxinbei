"""Bounded BPE fitting and rank-ordered merge application with patch backfill."""
import heapq
import numpy as np


def learn_merges(sequences, atomic_vocab, number, max_span=5):
    width = max(map(len, sequences))
    tokens = np.full((len(sequences), width), -1, np.int32)
    for i, seq in enumerate(sequences):
        tokens[i, :len(seq)] = seq
    capacity = atomic_vocab + number
    lengths = np.ones(capacity, np.int32)
    rules = []
    for step in range(number):
        left, right = tokens[:, :-1], tokens[:, 1:]
        valid = (left >= 0) & (right >= 0)
        valid &= lengths[left.clip(min=0)] + lengths[right.clip(min=0)] <= max_span
        encoded = left[valid].astype(np.int64) * capacity + right[valid]
        if not len(encoded):
            break
        counts = np.bincount(encoded, minlength=capacity * capacity)
        pair = int(counts.argmax())
        a, b = divmod(pair, capacity)
        new = atomic_vocab + step
        lengths[new] = lengths[a] + lengths[b]
        rules.append([a, b, new])
        # Greedy, non-overlapping replacement including self-pair runs.
        for col in range(width - 1):
            hit = (tokens[:, col] == a) & (tokens[:, col + 1] == b)
            tokens[hit, col] = new
            tokens[hit, col + 1] = -1
        kept = tokens >= 0
        rows, cols = np.nonzero(kept)
        destination = np.cumsum(kept, axis=1) - 1
        compact = np.full_like(tokens, -1)
        compact[rows, destination[rows, cols]] = tokens[rows, cols]
        tokens = compact
    return rules, lengths[:atomic_vocab + len(rules)].tolist()


class StructuralTokenizer:
    def __init__(self, merges):
        self.rules = {(int(a), int(b)): (rank, int(new)) for rank, (a, b, new) in enumerate(merges)}

    def encode(self, sequence):
        tokens = [int(x) for x in sequence]
        n = len(tokens)
        following = list(range(1, n)) + [-1]
        previous = [-1] + list(range(n - 1))
        ends = list(range(1, n + 1))
        active = [True] * n
        queue = []
        def offer(left):
            if left < 0 or not active[left]:
                return
            right = following[left]
            if right < 0:
                return
            rule = self.rules.get((tokens[left], tokens[right]))
            if rule:
                rank, new = rule
                heapq.heappush(queue, (rank, left, right, tokens[left], tokens[right], new))
        for i in range(n - 1):
            offer(i)
        while queue:
            _, left, right, a, b, new = heapq.heappop(queue)
            if not active[left] or not active[right] or following[left] != right or tokens[left] != a or tokens[right] != b:
                continue
            tokens[left] = new
            ends[left] = ends[right]
            active[right] = False
            after = following[right]
            following[left] = after
            if after >= 0:
                previous[after] = left
            offer(previous[left])
            offer(left)
        result = np.empty(n, np.uint16)
        for i in range(n):
            if active[i]:
                result[i:ends[i]] = tokens[i]
        return result
