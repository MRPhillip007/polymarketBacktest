"""Range-request reader for remote Parquet: fetch the footer once, then only the column chunks needed."""
import io, struct, requests, threading, bisect
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
import concurrent.futures as cf

_tl = threading.local()
def sess():
    s = getattr(_tl, 's', None)
    if s is None:
        s = requests.Session(); _tl.s = s
    return s

def backoff(i):
    """pause before retry i: 1, 2, 4, 8, 16, 30, 30 ... s with jitter (the archive answers 503 when busy)"""
    import time, random
    time.sleep(min(30, 2 ** i) * (0.75 + 0.5 * random.random()))

def get_range(url, a, b, tries=10):
    """bytes [a, b) of url"""
    for i in range(tries):
        try:
            r = sess().get(url, headers={'Range': f'bytes={a}-{b-1}'}, timeout=120)
            if r.status_code == 206 and len(r.content) == b - a:
                return r.content
            err = f'status {r.status_code} len {len(r.content)}'
        except Exception as e:
            err = repr(e)
        backoff(i)
    raise IOError(f'range {a}-{b} of {url}: {err}')

class Sparse(io.RawIOBase):
    """Read-only file over a dict of fetched byte ranges; misses are fetched on demand (and counted)."""
    def __init__(self, url, size, chunks):
        self.url, self.size, self.pos = url, size, 0
        self.starts = sorted(chunks); self.chunks = chunks; self.misses = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def _find(self, a, n):
        i = bisect.bisect_right(self.starts, a) - 1
        if i >= 0:
            s = self.starts[i]; c = self.chunks[s]
            if a + n <= s + len(c): return c[a - s:a - s + n]
        return None
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        d = self._find(self.pos, n)
        if d is None:
            self.misses += 1
            d = get_range(self.url, self.pos, self.pos + n)
            self.chunks[self.pos] = d; bisect.insort(self.starts, self.pos)
        self.pos += n
        return d
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def open_remote(url, tail=2_000_000, tries=10):
    for i in range(tries):
        try:
            r = sess().get(url, headers={'Range': f'bytes=-{tail}'}, timeout=120)
            if r.status_code == 206: break
        except Exception:
            pass
        backoff(i)
    assert r.status_code == 206, r.status_code
    size = int(r.headers['Content-Range'].split('/')[-1])
    t = r.content
    flen = struct.unpack('<I', t[-8:-4])[0]
    start = size - len(t)
    if flen + 8 > len(t):
        more = get_range(url, size - flen - 8, start)
        t = more + t; start = size - flen - 8
    sp = Sparse(url, size, {start: t})
    return sp, pq.ParquetFile(sp)

def rg_stats(pfile, col):
    md = pfile.metadata; j = pfile.schema_arrow.get_field_index(col)
    out = []
    for i in range(md.num_row_groups):
        s = md.row_group(i).column(j).statistics
        out.append((s.min, s.max) if s is not None and s.has_min_max else (None, None))
    return out

def prefetch(sp, pfile, rgs, cols, gap=1_000_000, workers=6):
    md = pfile.metadata; names = [md.schema.column(i).path for i in range(md.num_columns)]
    rngs = []
    for g in rgs:
        rg = md.row_group(g)
        for i in range(rg.num_columns):
            c = rg.column(i)
            if c.path_in_schema.split('.')[0] not in cols: continue
            a = c.dictionary_page_offset if c.has_dictionary_page and c.dictionary_page_offset else c.data_page_offset
            rngs.append((a, a + c.total_compressed_size))
    rngs.sort(); merged = []
    for a, b in rngs:
        if merged and a - merged[-1][1] <= gap: merged[-1][1] = max(merged[-1][1], b)
        else: merged.append([a, b])
    # split big ranges into ~8MB pieces for parallelism
    parts = []
    for a, b in merged:
        while a < b:
            e = min(b, a + 8_000_000); parts.append((a, e)); a = e
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda ab: (ab[0], get_range(sp.url, *ab)), parts))
    for a, d in res:
        sp.chunks[a] = d; bisect.insort(sp.starts, a)
    return sum(b - a for a, b in parts)
