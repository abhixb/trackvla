"""Pure-Python LZO1X decompression for TMNF Gbx bodies.

``pygbx`` imports the ``lzo`` C extension, which has no Python 3.13 wheel.
Replay bodies are tens of kilobytes, so a straightforward port of the Linux
kernel's ``lzo1x_decompress_safe`` is fast enough. ``install()`` registers it
as the ``lzo`` module before ``pygbx`` is imported.
"""

from __future__ import annotations

import sys
import types

_M2_MAX_OFFSET = 0x0800


def _zero_run(src: bytes, ip: int, base: int) -> tuple[int, int]:
    """Decode an LZO length extension: runs of zero bytes add 255 each."""
    length = 0
    while src[ip] == 0:
        length += 255
        ip += 1
    return length + base + src[ip], ip + 1


def decompress(src: bytes, _header: bool = False, dst_len: int | None = None) -> bytes:
    """Decompress a raw LZO1X stream (signature matches ``lzo.decompress``)."""
    src = bytes(src)
    out = bytearray()
    ip = 0
    state = 0
    if src[0] > 17:
        literal = src[0] - 17
        ip = 1
        out += src[ip : ip + literal]
        ip += literal
        state = literal if literal < 4 else 4
    while True:
        t = src[ip]
        ip += 1
        if t < 16:
            if state == 0:
                literal = t
                if literal == 0:
                    literal, ip = _zero_run(src, ip, 15)
                literal += 3
                out += src[ip : ip + literal]
                ip += literal
                state = 4
                continue
            trailing = t & 3
            if state != 4:
                position = len(out) - 1 - (t >> 2) - (src[ip] << 2)
                length = 2
            else:
                position = len(out) - (1 + _M2_MAX_OFFSET) - (t >> 2) - (src[ip] << 2)
                length = 3
            ip += 1
        elif t >= 64:
            trailing = t & 3
            position = len(out) - 1 - ((t >> 2) & 7) - (src[ip] << 3)
            ip += 1
            length = (t >> 5) + 1
        elif t >= 32:
            length = t & 31
            if length == 0:
                length, ip = _zero_run(src, ip, 31)
            length += 2
            word = src[ip] | (src[ip + 1] << 8)
            ip += 2
            position = len(out) - 1 - (word >> 2)
            trailing = word & 3
        else:
            position = len(out) - ((t & 8) << 11)
            length = t & 7
            if length == 0:
                length, ip = _zero_run(src, ip, 7)
            length += 2
            word = src[ip] | (src[ip + 1] << 8)
            ip += 2
            position -= word >> 2
            trailing = word & 3
            if position == len(out):
                break
            position -= 0x4000
        if position < 0:
            raise ValueError("corrupt LZO stream: match before start of output")
        # Matches may overlap their own output, so copy byte by byte.
        for offset in range(length):
            out.append(out[position + offset])
        state = trailing
        out += src[ip : ip + trailing]
        ip += trailing
    if dst_len is not None and len(out) != dst_len:
        raise ValueError(f"LZO output is {len(out)} bytes; expected {dst_len}")
    return bytes(out)


def install() -> None:
    """Expose this module as ``lzo`` unless the C extension is available."""
    if "lzo" in sys.modules:
        return
    try:
        import lzo  # noqa: F401
    except ImportError:
        shim = types.ModuleType("lzo")
        shim.decompress = decompress  # type: ignore[attr-defined]
        sys.modules["lzo"] = shim
