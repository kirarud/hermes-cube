"""Hermes Face v0.1: deterministic, morphable 3D companion silhouette.

Uses the same particle IDs as all existing shapes; no random frame jitter,
no external model/assets and no new rendering engine. Front is +Z.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def regions(n: int) -> tuple[int, int, int, int, int]:
    """Cumulative boundaries: head, body, eyes, mouth, antennae."""
    head = round(n * 0.66)
    body = round(n * 0.19)
    eyes = round(n * 0.075)
    mouth = round(n * 0.035)
    return head, head + body, head + body + eyes, head + body + eyes + mouth, n


def generate_face(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Produce a recognizable face/bust while preserving particle count/order."""
    n = len(points)
    result = np.zeros((n, 3), dtype=np.float64)
    if not n:
        return result
    a, b, c, d, _ = regions(n)
    if a:
        i = np.arange(a, dtype=np.float64) + 0.5
        y = 1.0 - 2.0 * i / a
        theta = i * (np.pi * (3.0 - np.sqrt(5.0)))
        r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
        depth = 0.76 + 0.24 * np.mod(i * 0.61803398875, 1.0)
        result[:a, 0] = 0.78 * r * np.cos(theta) * depth
        result[:a, 1] = 0.40 + 0.76 * y * depth
        result[:a, 2] = 0.55 * r * np.sin(theta) * depth
    if b > a:
        i = np.arange(b-a, dtype=np.float64) + 0.5
        y = 1.0 - 2.0 * i / (b-a)
        theta = i * (np.pi * (3.0 - np.sqrt(5.0)))
        r = np.sqrt(np.maximum(0.0, 1.0 - y*y))
        depth = 0.75 + 0.25*np.mod(i * 0.754877666, 1.0)
        result[a:b, 0] = 0.82 * r * np.cos(theta) * depth
        result[a:b, 1] = -0.78 + 0.48 * y * depth
        result[a:b, 2] = 0.43 * r * np.sin(theta) * depth
    if c > b:
        i = np.arange(c-b, dtype=np.float64)
        left = i < (c-b)//2
        count_left = max(1, (c-b)//2)
        local = np.where(left, i, i-count_left)
        half = np.where(left, count_left, max(1,(c-b)-count_left))
        t = 2*np.pi*(local+0.5)/half
        result[b:c, 0] = np.where(left, -0.30, 0.30) + 0.105*np.cos(t)
        result[b:c, 1] = 0.48 + 0.065*np.sin(t)
        result[b:c, 2] = 0.62 + 0.012*np.sin(t*2)
    if d > c:
        t = np.linspace(0.1, np.pi-0.1, d-c)
        result[c:d, 0] = 0.18*np.cos(t)
        result[c:d, 1] = 0.12 - 0.10*np.sin(t)
        result[c:d, 2] = 0.62
    if n > d:
        i = np.arange(n-d, dtype=np.float64)
        side = np.where(i < (n-d)//2, -1.0, 1.0)
        m = max(1,(n-d)//2)
        local = np.where(side<0, i, i-m)
        t = local / m
        result[d:n,0] = side*(0.40+0.22*t)
        result[d:n,1] = 1.02+0.39*t
        result[d:n,2] = 0.04*np.sin(local*2)
    return result
