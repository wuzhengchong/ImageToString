This is a program that can convert an image into a string. The generated string, when zoomed out or viewed from a distance, closely resembles the original image. It is suitable for outputting images on a TFT-OLED screen using a microcontroller, displaying images as text, or just for personal entertainment. The main body of this program is ImageChange.py. To run it directly, you can input parameters and call Show.py to display. Alternatively, you can use import ImageChange | a = ImageChange.ImageToString([path],[length],[quality],[reverse]) to display. Show.py is a string displayer that provides greater zooming capabilities. The length is recommended to be between 500-1500, which is clear and displays without lagging; the quality is ["min","low","medium","high","max"]. If running on a computer, it is recommended to use max. For a string of length 1000, it takes only about 2 seconds on an old e3 processor. The effect of reversing colors requires testing and observation to see which one better meets your expectations.

以上是我自己写的，用百度翻译。下面是ai写的（代码也基本是deepseek-v4.1-pro写的）



# ImageChange — structural image-to-ASCII-art converter

Turn a photo into a picture "drawn" with characters: horizontal lines tend to
become `-`, vertical lines `|`, round areas `o`, dark regions dense glyphs like
`@` / `#` — instead of naively mapping brightness onto a fixed ramp of
characters.

It is designed to be used **as a library**: `import ImageChange`, call one
function, done. The whole project is two files:

| File | Purpose |
| --- | --- |
| `ImageChange.py` | The converter. A **single, self-contained file** — the baked glyph table lives inside it. This is the only module your code needs to import. |
| `Show.py` | Optional tkinter viewer (standard library) that displays the result in square, gap-free character cells with zoom and scrolling. Not needed for the conversion itself. |

## Features

- **Shape matching** — every candidate character is pre-rendered into a glyph
  bitmap; each image cell is matched against all glyphs by "pixel difference +
  ink-density penalty", so structure (line direction, block size) maps to
  visually similar characters.
- **Fully pre-tabulated** — the glyph table is generated offline and written as
  literals; at run time **no font is loaded and no glyph is rasterized**.
- **Multi-threaded** — the per-cell matching (the only compute-bound stage) is
  split into blocks and run on a thread pool of `min(8, cpu_count)` workers.
- **Vectorized to the hilt** — each cell is sampled as an 8x8 binary pattern
  packed into one `uint64`, matched with numpy's `bitwise_count` (hardware
  popcount); scores are stored as `uint8` and the ink penalty is a table
  lookup, cutting the memory traffic of a naive implementation 4x.
- **Aspect-preserving** — row count is derived from the photo's aspect ratio,
  so landscape and portrait orientations come out right.
- **Brightness guard** — overly bright photos (mean gray > 200) are dimmed
  proportionally first, so the output never comes out mostly blank.

## Dependencies

- Python 3.9+
- `numpy >= 2.0` (for `np.bitwise_count`)
- `Pillow`

```bash
pip install "numpy>=2.0" pillow
```

## Library usage

### Convert an image

```python
from ImageChange import ImageToString

# path:   source image (PNG / JPG / JPEG)
# 100:    output length in character columns (rows follow the aspect ratio)
# "high": quality tier; invert=0 leaves the image as is
art = ImageToString("photo.jpg", 100, quality="high", invert=0)

# art is a list[list[str]] of shape rows x 100; art[y][x] is one character
text = "\n".join("".join(row) for row in art)
print(text)
```

### Show it in a viewer (optional)

```python
from Show import ShowS

ShowS(text)                  # square cells, zoom with Ctrl+wheel
# or with a custom window title:
ShowS(text, title="my photo")
```

### API

```python
ImageToString(image_path: str, out_length: int,
              quality: str = "medium", invert: int = 0) -> list[list[str]]
```

| Parameter | Meaning |
| --- | --- |
| `image_path` | Path to the source image. PNGs with an alpha channel are composited onto a white background first. |
| `out_length` | Output length — the number of character columns (positive integer). Rows = `round(cols × height / width)`. |
| `quality` | One of `"min"` / `"low"` / `"medium"` / `"high"` / `"max"`; uses the first 8 / 16 / 32 / 48 / 61 characters of the pool. Higher = finer matching. Default `"medium"`. |
| `invert` | `0` leaves the image as is (default); `1` inverts it — brighter areas get denser characters, pure black becomes spaces. |

Return value: a `list[list[str]]` of shape `rows × out_length`. Each element is
a single character (a space for blank areas).

Exceptions: invalid `quality` / `out_length` / `invert` raise `ValueError`;
a missing file raises `FileNotFoundError`; an unreadable image raises
`PIL.UnidentifiedImageError`. Other PIL/OS errors propagate unchanged.

## Command-line demo

```bash
python ImageChange.py
```

It prompts for the image path, output length, quality, and optional inversion,
prints the art, then opens the viewer (requires `Show.py` in the same folder).

You can also run `python Show.py` alone: paste any ASCII text and click "Show"
— it does not depend on ImageChange.

## Performance

Measured on an 8-core machine, JPEG source, 400-column output,
`quality="high"`:

- 3000×3000 → 400×400 output: ~**160 ms**
- 4032×3024 (12 MP photo) → 400×300 output: ~**190 ms**

Notes:

- Each cell is sampled as an 8×8 pattern; compared with a full-precision 8×16
  version, ~93% of cells get the identical character and the visual difference
  is negligible.
- With large **PNG** files, total time is dominated by PIL decoding (a 20+ MB
  PNG takes hundreds of milliseconds to decode alone) — the matching itself
  only costs about a hundred milliseconds. Prefer JPEG when speed matters.

## How it works (brief)

1. **Offline tabulation** — each pool character is rendered once into an 8×16
   bitmap, binarized around its own mean, reduced to an 8×8 mask by OR-ing
   vertical pixel pairs, and stored — together with its ink count and mean
   gray — as literals inside `ImageChange.py`.
2. **Gridding** — the source image is converted to gray and resized once to
   `columns×8 by rows×8`, so every output cell is exactly one 8×8 pixel block;
   each cell's sum and sum-of-squares are precomputed (needed to detect
   "flat" cells).
3. **Binarization** — each cell becomes a 64-bit pattern using its own mean,
   via the exact integer equivalent `v < ceil(sum/64)` (no float rounding).
4. **Matching** — the pattern is XOR-ed against every glyph and popcounted for
   the pixel difference, plus an `|ink − glyph ink|` density penalty; both
   polarities are tried and the cheaper one wins. Low-variance cells bypass
   shape matching and pick the glyph with the closest brightness.
5. **Parallelism** — matching runs in blocks on a `ThreadPoolExecutor` across
   up to 8 cores.
