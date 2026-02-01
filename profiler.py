import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self._stack = []

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.times[name] += elapsed
            self.counts[name] += 1

    def reset(self):
        self.times.clear()
        self.counts.clear()

    def summary(self, limit=30):
        items = sorted(self.times.items(), key=lambda x: x[1], reverse=True)
        lines = []
        total = sum(self.times.values())
        lines.append(f"total={total:.4f}s")
        for name, t in items[:limit]:
            cnt = self.counts.get(name, 0)
            avg = t / cnt if cnt else 0.0
            lines.append(f"{name}: {t:.4f}s | count={cnt} | avg={avg:.6f}s")
        return "\n".join(lines)

    def dump(self, path: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write('--------------------')
            f.write(self.summary())
            f.write("\n")


profiler = Profiler()
