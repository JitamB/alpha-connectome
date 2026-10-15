"""Resumable download of starterpack files straight from the remote ZIP (no archive kept on disk).

  python tools/fetch_data.py DEST_DIR datasets/train.parquet [datasets/valid.parquet ...]

Reads the ZIP central directory from the archive's last MiB, then streams each requested entry's
deflated bytes through curl into zlib; on a network failure curl restarts at the next compressed
byte, so decompression state is never lost. Every file is CRC-checked. Tries the mirrors in order.
"""
import io
import re
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

URLS = [
    'https://storage.yandexcloud.net/files-alternative.wundernn.io/wnn_connectome_starterpack.zip',
    'https://files.wundernn.io/wnn_connectome_starterpack.zip',
]
PREFIX = 'wnn_connectome_starterpack/'
CHUNK = 1 << 20
TAIL = 1 << 20


def curl(url, start, end, stream=False):
    # --max-filesize: the main server sometimes ignores the range and sends the whole 33 GB archive (seen twice;
    # once it filled 11.7 GB of RAM and the VM's OOM killer stopped the fetch). curl then aborts and we retry.
    cmd = ['curl', '-s', '--fail', '--speed-limit', '200000', '--speed-time', '60', '-r', f'{start}-{end}',
           '--max-filesize', str(end - start + 1), url]
    if stream:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for attempt in range(1, 6):
        r = subprocess.run(cmd, stdout=subprocess.PIPE)
        if r.returncode == 0 and len(r.stdout) == end - start + 1:
            return r.stdout
        print(f'range {start}-{end}: curl exit {r.returncode}, {len(r.stdout)} bytes; retry {attempt}', flush=True)
        time.sleep(5 * attempt)
    raise SystemExit(f'range {start}-{end} failed')


def archive_size(url):
    """HEAD Content-Length; some servers refuse HEAD, so fall back to a short ranged GET's Content-Range."""
    head = subprocess.run(['curl', '-sI', '--fail', '--max-time', '30', url], stdout=subprocess.PIPE).stdout.decode()
    if m := re.search(r'content-length: (\d+)', head, re.I):
        return int(m.group(1))
    head = subprocess.run(['curl', '-s', '--fail', '--max-time', '20', '-D', '-', '-o', '/dev/null', '-r', '0-1', url],
                          stdout=subprocess.PIPE).stdout.decode()
    return int(re.search(r'content-range: bytes 0-1/(\d+)', head, re.I).group(1))


class TailFile(io.RawIOBase):
    """Seekable view of a remote file of `size` bytes where only the last len(tail) bytes are present."""

    def __init__(self, size, tail):
        self.size, self.tail, self.pos = size, tail, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        base = self.size - len(self.tail)
        if self.pos < base:
            raise IOError('read outside downloaded tail')
        end = self.size if n < 0 else min(self.size, self.pos + n)
        data = self.tail[self.pos - base:end - base]
        self.pos = end
        return data


def pick_url():
    for url in URLS:
        try:
            return url, archive_size(url)
        except Exception as e:  # noqa: BLE001 - try the next mirror
            print(f'{url}: unavailable ({e})', flush=True)
    raise SystemExit('no reachable archive URL')


def fetch(url, info, dest: Path):
    header = curl(url, info.header_offset, info.header_offset + 29)
    n, m = struct.unpack('<HH', header[26:30])
    pos = info.header_offset + 30 + n + m
    end = pos + info.compress_size - 1
    dec, crc, written, retries, t0 = zlib.decompressobj(-15), 0, 0, 0, time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, 'wb') as out:
        while pos <= end:
            proc = curl(url, pos, end, stream=True)
            while data := proc.stdout.read(CHUNK):
                pos += len(data)
                chunk = dec.decompress(data)
                out.write(chunk)
                crc = zlib.crc32(chunk, crc)
                if (written + len(chunk)) // (1 << 30) != written // (1 << 30):
                    print(f'{dest.name}: {(written + len(chunk)) / info.file_size:6.1%}  '
                          f'{(written + len(chunk)) / (time.time() - t0) / 1e6:.0f} MB/s', flush=True)
                written += len(chunk)
            proc.wait()
            if pos <= end:
                retries += 1
                print(f'{dest.name}: curl exit {proc.returncode} at byte {pos}, retry {retries}', flush=True)
                time.sleep(min(60, 5 * retries))
        tail = dec.flush()
        out.write(tail)
        crc, written = zlib.crc32(tail, crc), written + len(tail)
    if crc != info.CRC or written != info.file_size:
        raise SystemExit(f'{dest}: CRC/size mismatch')
    print(f'{dest}: OK {written} bytes in {time.time() - t0:.0f}s, retries={retries}', flush=True)


if __name__ == '__main__':
    dest_dir, names = Path(sys.argv[1]), sys.argv[2:]
    url, size = pick_url()
    print(f'using {url} ({size} bytes)', flush=True)
    entries = {i.filename: i for i in zipfile.ZipFile(TailFile(size, curl(url, size - TAIL, size - 1))).infolist()}
    for name in names:
        fetch(url, entries[PREFIX + name], dest_dir / name)
