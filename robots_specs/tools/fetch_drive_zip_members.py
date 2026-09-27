#!/usr/bin/env python3
"""Extract a few files from a zip nested inside a large Google Drive zip, without
downloading the outer archive.

    fetch_drive_zip_members.py --drive-id ID --outer-bytes N \
        --member yahboomcar_ws.zip --member-sha256 HEX --cache FILE \
        --prefix yahboomcar_ws/src/.../meshes/ --dest DIR  REL [REL ...]

The outer zip's central directory is read with HTTP Range requests, then only the
member's own bytes are range-downloaded, verified against --member-sha256 and kept at
--cache (so a second run downloads nothing). Each REL is extracted from the member as
<prefix><REL> to <dest>/<REL>. Standard library only.
"""
import argparse
import hashlib
import html
import os
import re
import struct
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
import zlib

UA = "Mozilla/5.0 (fetch_drive_zip_members.py)"
CHUNK = 1 << 20


def log(msg):
    print(f">> {msg}", flush=True)


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def open_range(url, start, end):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=120)


def resolve_url(drive_id, size):
    """The direct-download URL for a Drive file too large for Drive's virus scan.

    confirm=t skips the "can't scan this file" page; should Drive still answer with that
    page, its form carries the real parameters (including a uuid), so it is resubmitted."""
    url = ("https://drive.usercontent.google.com/download?"
           + urllib.parse.urlencode({"id": drive_id, "export": "download", "confirm": "t"}))
    for _ in range(2):
        with open_range(url, 0, 0) as r:
            ctype = r.headers.get("Content-Type", "")
            body = r.read(1 << 16)
            if r.status == 206 and "text/html" not in ctype:
                total = r.headers.get("Content-Range", "").rpartition("/")[2]
                if total.isdigit() and int(total) != size:
                    die(f"Drive file {drive_id} is {total} bytes, expected {size}; upstream changed")
                return url
        page = body.decode("utf-8", "replace")
        form = re.search(r'<form[^>]*action="([^"]+)"(.*?)</form>', page, re.S)
        if not form:
            die(f"Drive did not serve {drive_id} (no download form; quota exceeded or file moved?)")
        params = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', form.group(2)))
        url = html.unescape(form.group(1)) + "?" + urllib.parse.urlencode(params)
    die(f"could not resolve a direct download for Drive file {drive_id}")


def fetch(url, start, length):
    with open_range(url, start, start + length - 1) as r:
        if r.status != 206:
            die(f"server ignored the Range request (HTTP {r.status})")
        data = r.read()
    if len(data) != length:
        die(f"short read: wanted {length} bytes at {start}, got {len(data)}")
    return data


def central_directory(url, size):
    tail_len = min(size, 1 << 16)
    tail = fetch(url, size - tail_len, tail_len)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        die("no end-of-central-directory record; not a zip?")
    cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    loc = tail.rfind(b"PK\x06\x07", 0, eocd)
    if loc >= 0:  # ZIP64
        (z64_off,) = struct.unpack("<Q", tail[loc + 8:loc + 16])
        rec = fetch(url, z64_off, 56)
        if rec[:4] != b"PK\x06\x06":
            die("bad ZIP64 end-of-central-directory record")
        cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    return fetch(url, cd_off, cd_size)


def find_member(cd, name):
    i = 0
    while i + 46 <= len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        (method, _t, _d, _crc, csize, usize, nlen, xlen, clen, _dn, _ia, _ea,
         off) = struct.unpack("<HHHIIIHHHHHII", cd[i + 10:i + 46])
        fname = cd[i + 46:i + 46 + nlen].decode("utf-8", "replace")
        extra = cd[i + 46 + nlen:i + 46 + nlen + xlen]
        if fname == name:
            j = 0
            while j + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[j:j + 4])
                if hid == 0x0001:  # ZIP64 sizes/offset, present only for fields at 0xFFFFFFFF
                    vals = list(struct.unpack(f"<{hlen // 8}Q", extra[j + 4:j + 4 + hlen - hlen % 8]))
                    if usize == 0xFFFFFFFF:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = vals.pop(0)
                    if off == 0xFFFFFFFF:
                        off = vals.pop(0)
                j += 4 + hlen
            return method, csize, usize, off
        i += 46 + nlen + xlen + clen
    die(f"{name} is not in the archive")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download_member(a):
    url = resolve_url(a.drive_id, a.outer_bytes)
    log(f"reading the central directory of Drive file {a.drive_id}")
    method, csize, usize, off = find_member(central_directory(url, a.outer_bytes), a.member)
    if method not in (0, 8):
        die(f"{a.member}: unsupported compression method {method}")
    nlen, xlen = struct.unpack("<HH", fetch(url, off + 26, 4))
    start = off + 30 + nlen + xlen
    log(f"downloading {a.member} only: {csize / 1e6:.0f} MB of {a.outer_bytes / 1e9:.1f} GB")
    os.makedirs(os.path.dirname(a.cache), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(a.cache), suffix=".part")
    try:
        h = hashlib.sha256()
        inflate = zlib.decompressobj(-15) if method == 8 else None
        done, shown = 0, 0
        with os.fdopen(fd, "wb") as out, open_range(url, start, start + csize - 1) as r:
            if r.status != 206:
                die(f"server ignored the Range request (HTTP {r.status})")
            while True:
                block = r.read(CHUNK)
                if not block:
                    break
                done += len(block)
                data = inflate.decompress(block) if inflate else block
                h.update(data)
                out.write(data)
                if done - shown >= 32 * CHUNK or done == csize:
                    shown = done
                    print(f"   {done / 1e6:.0f}/{csize / 1e6:.0f} MB", flush=True)
            if inflate:
                rest = inflate.flush()
                h.update(rest)
                out.write(rest)
        if done != csize:
            die(f"{a.member}: got {done} of {csize} bytes")
        if h.hexdigest() != a.member_sha256:
            die(f"{a.member}: sha256 {h.hexdigest()} != expected {a.member_sha256}")
        if os.path.getsize(tmp) != usize:
            die(f"{a.member}: size {os.path.getsize(tmp)} != {usize}")
        os.replace(tmp, a.cache)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-id", required=True)
    p.add_argument("--outer-bytes", type=int, required=True)
    p.add_argument("--member", required=True)
    p.add_argument("--member-sha256", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--dest", required=True)
    p.add_argument("files", nargs="+")
    a = p.parse_args()

    if os.path.isfile(a.cache) and sha256_file(a.cache) == a.member_sha256:
        log(f"{a.member} cached at {a.cache}")
    else:
        download_member(a)

    with zipfile.ZipFile(a.cache) as z:
        for rel in a.files:
            try:
                data = z.read(a.prefix + rel)
            except KeyError:
                die(f"{a.prefix + rel} is not in {a.member}")
            target = os.path.join(a.dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(data)


if __name__ == "__main__":
    main()
