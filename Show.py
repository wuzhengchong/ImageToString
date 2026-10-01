"""ASCII-art viewer window built on tkinter (standard library, no extra deps).

Each character occupies **2 text columns** and 1 row.  With Consolas at size 9,
2x the character width == 14 px == the line height, so every character lands in
a 14x14 **square** cell with no row/column gaps between characters (all Text
widget spacing values are 0).

Rendering is done entirely by tkinter's native text engine, so scrolling and
zooming stay smooth.

Two ways to use it:
1. Run this file directly: paste the generated ASCII text into the input box,
   then click "Show".
2. Import it from another module: ``from Show import ShowS; ShowS(text)``
   (ImageChange's ``__main__`` calls it automatically after converting).

Window features:
- Square cells: line spacing = font height, column spacing = 2x font width,
  with no row/column gaps;
- No automatic wrapping: content wider than the window is reached with the
  horizontal/vertical scrollbars; shrinking the window never hides content;
- Maximized on startup; Ctrl+wheel or the toolbar +/- adjust the font size
  (zoom); wheel scrolls vertically; Shift+wheel scrolls horizontally.
"""

import tkinter as tk
from tkinter import font as tkfont

_DUP: int = 2                 # text columns each character occupies (to make a square cell)
# At the default size 9, 2x char width = 14 px = line height 14 px: an exact
# square.  Consolas at 5 / 7 / 9 / 16 also have 2x width == line height, i.e.
# exact square cells.
_DEFAULT_SIZE: int = 9
_MIN_SIZE: int = 1
_MAX_SIZE: int = 200


def _widen(text: str) -> str:
    """Duplicate every character ``_DUP`` times so 1:2 cells become ~1:1 (square)."""
    return "\n".join("".join(ch * _DUP for ch in line) for line in text.split("\n"))


def _build_view(root: tk.Tk, text: str, title: str) -> None:
    """Build the square-cell, gap-free, zoomable ASCII view inside *root*."""
    root.title(title)
    root.geometry("900x600")
    try:
        root.state("zoomed")  # maximize at startup to fit as many characters as possible
    except tk.TclError:
        pass

    char_font = tkfont.Font(family="Consolas", size=_DEFAULT_SIZE)

    # ---- toolbar: font size (zoom) controls ----------------------------------
    top = tk.Frame(root)
    top.pack(side="top", fill="x")
    size_var = tk.StringVar(value=f"Font: {_DEFAULT_SIZE}")

    def set_size(new_size: int) -> None:
        new_size = max(_MIN_SIZE, min(_MAX_SIZE, int(new_size)))
        char_font.configure(size=new_size)
        cw, ch = char_font.measure("M") * _DUP, char_font.metrics("linespace")
        size_var.set(f"Font: {new_size}   Cell: {cw}x{ch}px")

    tk.Button(top, text="-", width=3,
              command=lambda: set_size(char_font.cget("size") - 1)).pack(side="left")
    tk.Button(top, text="+", width=3,
              command=lambda: set_size(char_font.cget("size") + 1)).pack(side="left")
    tk.Button(top, text="Reset",
              command=lambda: set_size(_DEFAULT_SIZE)).pack(side="left", padx=4)
    tk.Button(top, text="Min",
              command=lambda: set_size(_MIN_SIZE)).pack(side="left")
    tk.Label(top, textvariable=size_var).pack(side="left", padx=8)
    tk.Label(top, text="Ctrl+wheel zoom | wheel scroll | Shift+wheel horizontal scroll"
             ).pack(side="right", padx=6)

    # ---- body: no-wrap Text + both scrollbars (all spacing 0, no gaps) -------
    body = tk.Frame(root)
    body.pack(fill="both", expand=True)
    txt = tk.Text(body, wrap="none", font=char_font,
                  bg="#111111", fg="#dddddd", insertbackground="#dddddd",
                  padx=0, pady=0, bd=0, highlightthickness=0,
                  spacing1=0, spacing2=0, spacing3=0, state="disabled")
    vsb = tk.Scrollbar(body, orient="vertical", command=txt.yview)
    hsb = tk.Scrollbar(body, orient="horizontal", command=txt.xview)
    txt.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
    txt.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)
    txt.configure(state="normal")
    txt.insert("1.0", _widen(text))
    txt.configure(state="disabled")
    set_size(_DEFAULT_SIZE)

    # ---- wheel bindings ------------------------------------------------------
    def on_ctrl_wheel(event: tk.Event) -> str:
        set_size(char_font.cget("size") + (1 if event.delta > 0 else -1))
        return "break"

    def on_wheel(event: tk.Event) -> str:
        txt.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def on_shift_wheel(event: tk.Event) -> str:
        txt.xview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    root.bind("<Control-MouseWheel>", on_ctrl_wheel)
    root.bind("<Shift-MouseWheel>", on_shift_wheel)
    root.bind("<MouseWheel>", on_wheel)


def ShowS(text: str, title: str = "ASCII Viewer") -> None:
    """Open a window showing *text* in square cells (no row/column gaps).

    The function only returns after the window is closed.

    Args:
        text:  The ASCII text to display (lines separated by newlines).
        title: Window title, default "ASCII Viewer".
    """
    root = tk.Tk()
    _build_view(root, text, title)
    root.mainloop()


# ---------------------------------------------------------------------------
# Direct run: paste the text first, then display it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    entry_root = tk.Tk()
    entry_root.title('Paste ASCII text, then click "Show"')
    entry_root.geometry("720x480")
    tk.Label(entry_root, text="Paste the generated ASCII art below:").pack(
        side="top", fill="x", padx=6, pady=(6, 0))

    entry_box = tk.Text(entry_root, wrap="none")
    entry_box.pack(side="top", fill="both", expand=True, padx=6, pady=6)

    def _show() -> None:
        pasted = entry_box.get("1.0", "end").rstrip("\n")
        entry_root.destroy()
        if pasted.strip():
            ShowS(pasted)

    tk.Button(entry_root, text="Show", command=_show).pack(
        side="top", fill="x", padx=6, pady=(0, 6))
    entry_root.mainloop()