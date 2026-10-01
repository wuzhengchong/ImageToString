"""Structural image-to-ASCII converter (character *shape* matching).

Instead of mapping brightness to a fixed ramp, every candidate character is
pre-rendered into a small glyph bitmap (the "table").  The source image is cut
into ``out_width x out_height`` cells, and each cell is matched against the
glyphs by pixel-pattern similarity, so a horizontal line tends to become '-',
a vertical line '|', a soft round area 'o', blank areas a space, and so on.

This module is a **single, self-contained file**: the glyph table below is
pre-computed offline (every candidate character is rendered once with Pillow
in an 8x16 canvas, binarized around its own mean, and reduced to an 8x8 bitmap
by OR-ing each vertical pixel pair).  At run time no font is loaded and no
glyph is ever rasterized -- nothing in the table is computed on the fly.

Implementation notes
--------------------
* Each image cell is sampled as an 8x8 binarized pattern (64 bits), packed
  into one ``uint64``, and matched against every glyph with numpy's vectorized
  ``bitwise_count`` (hardware popcount).
* Binarization uses exact integer arithmetic: ``v < mean`` is written as
  ``v < ceil(sum/n)`` for the power-of-two ``n`` used here, so no float
  rounding creeps in.
* The per-cell matching -- the only compute-bound stage -- is split into small
  blocks and run on a thread pool, which scales across cores.

Dependencies: numpy >= 2.0 (for ``bitwise_count``) and Pillow.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Pre-computed glyph table (baked offline; see the module docstring on how it
# was produced).  Each row across MASKS / INKS / MEANS belongs to the character
# at the same index of POOL.
# ---------------------------------------------------------------------------

POOL: str = ' .-|#@oX/\\*+=()[]{}:;cvszn^\'`",~!?JLYkdhbxuaeqpg%$&8MWB0QNDRU'

# Quality -> number of characters taken from the pool (more = finer match).
SIZES: dict[str, int] = {"min": 8, "low": 16, "medium": 32, "high": 48, "max": 61}

# 8x8 glyph bitmaps, one uint64 per character (bit r*8+c marks row r, col c).
MASKS: tuple = (
    0x0000000000000000, 0x00001C0000000000, 0x0000003E00000000, 0x1818181818181818, 0x0000367F34FF2C00, 0x3C237DF5FFC27C00,
    0x00007E63673C0000, 0x0000E73E1C7E6300, 0x0006060C18307000, 0x00607038180C0600, 0x000000007E3E7E00, 0x000018FF18180000,
    0x0000007F7F000000, 0x30380C0C0C1C3800, 0x041C383030180C00, 0x3C0C0C0C0C0C3C00, 0x3C30303030303C00, 0x3818180E0C183800,
    0x0E18187838181E00, 0x00001C001C1C0000, 0x001E180018180000, 0x00007E060E7C0000, 0x00001C3E66630000, 0x00007E7C0E3C0000,
    0x00007E1C307E0000, 0x000066666E3E0000, 0x00000000663E1C00, 0x0000000000181800, 0x0000000000001E00, 0x00000000003E3E00,
    0x001E1C0000000000, 0x0000007FCE000000, 0x0000181818181800, 0x00000C0C3C703C00, 0x00003E3030303E00, 0x00007E0606060600,
    0x000018183C67C300, 0x0000761E3E660600, 0x00007E63667C6000, 0x000066666E3E0600, 0x00007E666E3E0600, 0x0000773C3E660000,
    0x00007E6666660000, 0x00007E7E663C0000, 0x00007E7F663C0000, 0x60607E63667C0000, 0x06067E666E3E0000, 0x3E637E3E667C0000,
    0x0000F3FE183FEF00, 0x000C7F783E1E7C00, 0x0000FF7B5E3E3E00, 0x00007E763E663C00, 0x0000C3DB5F776700, 0x0000767F5BC3C300,
    0x00007E667E663E00, 0x00007E6F7F763C00, 0x00F87EE3C3673C00, 0x000072727A6E6600, 0x00007F6343733F00, 0x000066363E763E00,
    0x00007E6363636300,
)

# Ink counts (number of set bits in each mask).
INKS: tuple = (
    0, 3, 5, 16, 25, 35, 19, 24, 13, 14, 17, 14,
    14, 17, 15, 18, 18, 17, 20, 9, 10, 16, 16, 18,
    17, 18, 12, 4, 4, 10, 7, 12, 10, 15, 16, 14,
    17, 20, 21, 20, 22, 19, 18, 20, 21, 23, 24, 29,
    28, 27, 29, 24, 27, 25, 25, 28, 29, 22, 25, 23,
    22,
)

# Mean gray value of each character's original 8x16 raster (on a white canvas).
MEANS: tuple = (
    255.0000, 248.0469, 246.8672, 223.5000, 199.8438, 170.0781, 215.5781, 205.5547,
    230.0156, 230.0078, 229.1172, 227.8906, 231.5781, 223.5391, 223.5703, 216.4375,
    216.1094, 217.8750, 218.1484, 242.1875, 235.2266, 226.3047, 224.3906, 221.4453,
    223.7188, 216.6016, 238.8203, 246.7109, 249.2109, 239.2109, 241.7500, 234.7500,
    232.1016, 226.5391, 221.1172, 225.0000, 216.8672, 209.9062, 205.3438, 209.4141,
    205.7578, 218.8672, 216.5547, 212.4688, 212.5469, 205.1172, 204.8906, 189.4141,
    196.5391, 196.0312, 188.8203, 199.5234, 199.0000, 199.2578, 196.0234, 196.5547,
    194.4688, 198.5781, 200.6641, 201.9141, 208.5234,
)

# ---------------------------------------------------------------------------
# Runtime configuration and derived constants.
# ---------------------------------------------------------------------------

# Every output cell is matched as an 8x8 binarized pattern.
_GW, _GH = 8, 8
_N = _GW * _GH
_FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
# Bit offsets (0, 8, 16, ... 56) that pack the eight row-bytes into one uint64.
_SHIFTS = np.arange(_GH, dtype=np.uint64) * np.uint64(_GW)

# A cell whose gray variance is below this counts as featureless (flat).
_FLAT_VAR = 25.0
# If the overall brightness (mean gray) exceeds this, dim the image first so
# the output does not come out mostly blank.
_BRIGHT_LIMIT: float = 200.0

_VALID_QUALITY = frozenset(SIZES)
_CHARS = np.frombuffer(POOL.encode("ascii"), np.uint8)
_MASKS = np.array(MASKS, np.uint64)
_INKS = np.array(INKS, np.int32)
_MEANS = np.array(MEANS, np.float32)

# Matching is the bottleneck and parallelizes well: one cell-block per worker.
_THREADS = min(8, os.cpu_count() or 1)

# quality -> (masks, ink-penalty LUT, inverted LUT, sorted means, argsort)
_QPREP: dict[str, tuple] = {}
# Cell ink counts live in [0, 64]: the penalty is a plain table lookup, so the
# hot loop never has to build a wide (cells x glyphs) integer temporary.
_INK_RANGE = np.arange(_N + 1, dtype=np.int16)


def _prepare(quality: str) -> tuple:
    """Return the glyph arrays for *quality* (pool prefix, sorted by mean)."""
    prep = _QPREP.get(quality)
    if prep is None:
        n = SIZES[quality]
        means = _MEANS[:n]
        order = np.argsort(means, kind="stable")
        glyph_ink = _INKS[:n].astype(np.int16)
        ink_diff = np.abs(_INK_RANGE[:, None] - glyph_ink[None]).astype(np.uint8)
        inv_diff = np.abs((_N - _INK_RANGE)[:, None] - glyph_ink[None]).astype(
            np.uint8
        )
        prep = (_MASKS[:n], ink_diff, inv_diff, means[order], order)
        _QPREP[quality] = prep
    return prep


def ImageToString(
    image_path: str,
    out_length: int,
    quality: str = "medium",
    invert: int = 0,
) -> list[list[str]]:
    """Convert an image into a 2-D ASCII character array by shape matching.

    Args:
        image_path: Path to the source image (PNG or JPG/JPEG).  PNGs with an
            alpha channel are composited onto a white background first.
        out_length: The output "length" -- the number of character columns (a
            positive integer).  The number of rows follows the photo's aspect
            ratio automatically, so a 16:9 photo passed 32 becomes 32 columns
            x 18 rows, and landscape/portrait orientation is preserved.
        quality:    One of {"min", "low", "medium", "high", "max"}, default
            "medium".  Higher quality matches against more (finer) characters.
        invert:     Optional inversion switch: 0 (default) leaves the image as
            is; 1 inverts it -- brighter areas get denser characters and pure
            black becomes spaces.

    Returns:
        A ``list[list[str]]`` of shape ``rows x out_length``.  Each element is
        a single character (a space for blank areas).  ``result[y][x]`` is the
        character at row *y*, column *x*.

    Raises:
        ValueError:     If *quality* is not in the allowed set, if
                        *out_length* is not a positive integer, or if *invert*
                        is not 0 or 1.
        FileNotFoundError: If *image_path* does not exist.
        PIL.UnidentifiedImageError: If the file is not a readable image.
        OSError / other PIL errors: Propagated unchanged from Pillow.
    """
    # ---- input validation ------------------------------------------------
    if quality not in _VALID_QUALITY:
        raise ValueError(
            f"quality must be one of {sorted(_VALID_QUALITY)}, got {quality!r}"
        )
    if out_length <= 0:
        raise ValueError("out_length must be a positive integer")
    if invert not in (0, 1):
        raise ValueError("invert must be 0 (off) or 1 (on)")

    masks, ink_diff, inv_diff, means_sorted, mean_order = _prepare(quality)

    # ---- load & preprocess ----------------------------------------------
    img = Image.open(image_path)
    iw, ih = img.size

    # Columns = out_length; rows follow the photo's aspect ratio.  The viewer
    # renders square character cells, so no aspect compensation is needed here.
    out_width = out_length
    out_height = max(1, round(out_length * ih / iw))

    # Flatten alpha channel onto a white background.
    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg

    # JPEG: ask the decoder for no more pixels than we could possibly use.
    img.draft("L", (out_width * _GW, out_height * _GH))
    # Going to L *before* the big resize is several times cheaper than resizing
    # in RGB and converting afterwards.
    gray = img.convert("L")

    target_w, target_h = out_width * _GW, out_height * _GH
    # Upscaling with BOX only ever replicates source pixels, so NEAREST gives a
    # near-identical result for a fraction of the cost; downscaling needs BOX.
    resample = (
        Image.Resampling.NEAREST
        if iw <= target_w and ih <= target_h
        else Image.Resampling.BOX
    )
    raw = np.asarray(gray.resize((target_w, target_h), resample), np.uint8)

    # Overall brightness check: if the mean gray is too high (> _BRIGHT_LIMIT)
    # the picture is mostly white and the output would come out blank, so dim
    # it proportionally towards _BRIGHT_LIMIT (brighter -> dimmed more).  Both
    # brightness and inversion are applied through one 256-entry lookup table;
    # a strided sample is enough to estimate the mean, no full extra resize.
    mean_gray = float(raw[::4, ::4].mean())
    lookup = None
    if mean_gray > _BRIGHT_LIMIT:
        gain = _BRIGHT_LIMIT / mean_gray
        lookup = [min(255, int(v * gain)) for v in range(256)]
    if invert:  # swap black and white; the matching logic needs no change
        flipped = [255 - v for v in range(256)]
        lookup = flipped if lookup is None else [flipped[v] for v in lookup]
    if lookup is not None:
        raw = np.asarray(lookup, np.uint8)[raw]

    # Resize once, then view the buffer as (cells, 64) so the whole match runs
    # on flat arrays: one row of 8 bytes == one cell's 8x8 binarized pattern.
    cells = np.ascontiguousarray(
        raw.reshape(out_height, _GH, out_width, _GW).transpose(0, 2, 1, 3)
    ).reshape(-1, _N)

    # Per-cell sum and sum-of-squares -> variance, which only decides whether a
    # cell is featureless (matched by brightness instead of by shape).
    total = cells.sum(axis=1, dtype=np.uint16)
    total_sq = np.einsum("ij,ij->i", cells, cells, dtype=np.uint32)
    var = (_N * total_sq.astype(np.float64) - total.astype(np.float64) ** 2) / (
        float(_N) * _N
    )
    flat = var < _FLAT_VAR
    gray_avg = total.astype(np.float32) / _N

    # Exact integer form of "v < mean" for n = 64: v < ceil(sum / 64), which
    # keeps the binarization a plain uint8 comparison (no wide temporary).
    ceil_mean = ((total.astype(np.uint16) + (_N - 1)) >> 6).astype(np.uint8)

    n_cells = cells.shape[0]
    out = np.empty(n_cells, np.int32)
    # Several small blocks per worker keeps the match temporaries cache-friendly.
    band = max(1, -(-n_cells // (_THREADS * 4)))

    def _band(i0: int) -> None:
        """Match the cells ``[i0, i0 + band)`` against the glyph table."""
        i1 = min(i0 + band, n_cells)
        bits = cells[i0:i1] < ceil_mean[i0:i1, None]
        # packbits along the cell's 64 bits yields its 8 row-bytes directly.
        packed = np.packbits(bits, axis=1, bitorder="little")
        pattern = np.bitwise_or.reduce(
            packed.astype(np.uint64) << _SHIFTS[None, :], axis=1
        )

        # Scores fit in a byte: every popcount here is <= 64, and the ink
        # penalty is pre-tabulated as |cell_ink - glyph_ink| (also <= 64), so
        # the whole (cells x glyphs) working set stays 4x smaller than int32.
        ink = np.bitwise_count(pattern)
        # Symmetric pixel difference PLUS an ink-density penalty, so a sparse
        # glyph cannot "hide" inside a dark cell: a thin line prefers '-', a
        # solid block prefers a dense glyph.  Both polarities are tried.
        cost = np.bitwise_count(pattern[:, None] ^ masks[None])
        cost += ink_diff[ink]
        alt = np.bitwise_count((~pattern & _FULL)[:, None] ^ masks[None])
        alt += inv_diff[ink]
        np.minimum(cost, alt, out=cost)
        best = cost.argmin(1)

        # Featureless cells fall back to the glyph with the closest brightness
        # (mirrors the classic ramp; bright -> space, dark -> dense char).
        avg = gray_avg[i0:i1]
        pos = np.clip(np.searchsorted(means_sorted, avg), 1, len(means_sorted) - 1)
        pick = np.where(
            np.abs(avg - means_sorted[pos - 1]) <= np.abs(means_sorted[pos] - avg),
            pos - 1,
            pos,
        )
        out[i0:i1] = np.where(flat[i0:i1], mean_order[pick], best)

    if _THREADS <= 1:
        for i0 in range(0, n_cells, band):
            _band(i0)
    else:
        with ThreadPoolExecutor(max_workers=_THREADS) as pool:
            list(pool.map(_band, range(0, n_cells, band)))

    return [
        list(row.tobytes().decode("ascii"))
        for row in _CHARS[out.reshape(out_height, out_width)]
    ]


# ---------------------------------------------------------------------------
# Direct-run entry point (not executed on import).
# Prompts for path / length / quality / optional inversion, then prints the art
# and opens the viewer (requires Show.py).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    from Show import ShowS

    image_path = input("Image path: ").strip().strip('"')
    out_length = int(
        input("Length (character columns; rows follow the aspect ratio): ")
    )
    quality = input("Quality (min/low/medium/high/max, Enter = medium): ").strip()
    invert = input("Invert (0 = off / 1 = on, Enter = 0): ").strip()

    t0 = time.perf_counter()
    result = ImageToString(
        image_path, out_length, quality or "medium", int(invert or "0")
    )
    elapsed = time.perf_counter() - t0

    art = "\n".join("".join(row) for row in result)
    print(art)
    print(f"\nElapsed {elapsed * 1000:.0f} ms  ({len(result)} rows x {len(result[0])} cols)")
    ShowS(art)