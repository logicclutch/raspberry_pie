"""Ed25519 signatures (RFC 8032), plain Python: no native package to install on the Pi.

Used for the licence file: the vendor signs it with a private key that never leaves the vendor's
machine, and the software checks it with the public key built into it. Checked against the RFC 8032
test vectors and OpenSSL in tests/test_licence.py.

Not constant-time. That is fine here: verification only touches public data, and signing runs on the
vendor's own computer, never on a client's Pi.
"""

# ruff: noqa: N806 - A..H follow the names in RFC 8032
from __future__ import annotations

import hashlib

_P = 2**255 - 19
_Q = 2**252 + 27742317777372353535851937790883648493  # order of the base point
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)

_Point = tuple[int, int, int, int]  # extended coordinates (X, Y, Z, T), x = X/Z, y = Y/Z, xy = T/Z


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _add(a: _Point, b: _Point) -> _Point:
    A = (a[1] - a[0]) * (b[1] - b[0]) % _P
    B = (a[1] + a[0]) * (b[1] + b[0]) % _P
    C = 2 * a[3] * b[3] * _D % _P
    D = 2 * a[2] * b[2] % _P
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _mul(s: int, pt: _Point) -> _Point:
    out: _Point = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            out = _add(out, pt)
        pt = _add(pt, pt)
        s >>= 1
    return out


def _equal(a: _Point, b: _Point) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * _inv(5) % _P
_GX = _recover_x(_GY, 0)
assert _GX is not None
_G: _Point = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(pt: _Point) -> bytes:
    zi = _inv(pt[2])
    x, y = pt[0] * zi % _P, pt[1] * zi % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decompress(s: bytes) -> _Point | None:
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _h(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little") % _Q


def _expand(seed: bytes) -> tuple[int, bytes]:
    if len(seed) != 32:
        raise ValueError("an Ed25519 private key is 32 bytes")
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(seed: bytes) -> bytes:
    """The 32-byte public key for a 32-byte private key (seed)."""
    a, _ = _expand(seed)
    return _compress(_mul(a, _G))


def sign(seed: bytes, msg: bytes) -> bytes:
    """64-byte signature of `msg` with the private key `seed`."""
    a, prefix = _expand(seed)
    pub = _compress(_mul(a, _G))
    r = _h(prefix + msg)
    big_r = _compress(_mul(r, _G))
    s = (r + _h(big_r + pub + msg) * a) % _Q
    return big_r + s.to_bytes(32, "little")


def verify(pub: bytes, msg: bytes, signature: bytes) -> bool:
    """True only if `signature` is a valid signature of `msg` by the owner of `pub`."""
    if len(pub) != 32 or len(signature) != 64:
        return False
    big_a = _decompress(pub)
    big_r = _decompress(signature[:32])
    if big_a is None or big_r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    k = _h(signature[:32] + pub + msg)
    return _equal(_mul(s, _G), _add(big_r, _mul(k, big_a)))
