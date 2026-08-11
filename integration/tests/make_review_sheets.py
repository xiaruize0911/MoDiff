"""Contact sheets for human review of the generated samples.

generate_fid_samples.py uses the SAME seed sequence in every mode, so `000000.png` in one folder and
`000000.png` in another are the same noise through different datapaths. That makes a per-image
side-by-side the diagnostic view -- quantization damage shows up as a difference against a picture
you can also see, rather than as a picture you have to judge on its own.

Two sheets:
  * comparison_<i>.png -- one row per sample, one column per mode, at full 256px. This is the one to
    look at. Column order is fp16 first, then increasing aggression, so damage should read left to
    right.
  * grid_<mode>.png -- all N samples of one mode, downscaled, for spotting mode-wide failure
    (colour casts, texture collapse) that a handful of side-by-sides could miss.
  * linear_<i>.png -- with --dir2, the same-seed comparison ACROSS two generation batches. Two runs
    at the same --seed0 share their noise, so this reads a configuration difference (e.g.
    MODIFF_LINEAR 0 vs 1) the same way the columns above read a precision difference. Only the
    MoDiff arms appear: the baselines are identical between such batches by construction.

Run: python integration/tests/make_review_sheets.py [--dir /workspace/review_2026-08-10]
"""
import argparse
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

#: Folder -> column label, in the order the sheets present them: reference, then W8A8, W8A4, W4A4,
#: each as PTQ-baseline then +MoDiff. MoDiff's claim is that the +MoDiff column at a given precision
#: beats the baseline column, and that W8A4+MoDiff holds up against fp16.
ORDER = [("fp16", "fp16"),
         ("int8_baseline", "W8A8 PTQ"),
         ("int8_modiff", "W8A8 +MoDiff"),
         ("int8_modiff_l0", "W8A8 MoDiff\nconv only"),
         ("int8_modiff_l1", "W8A8 MoDiff\nconv+proj"),
         ("w8a4_baseline", "W8A4 PTQ"),
         ("w8a4_modiff", "W8A4 +MoDiff"),
         ("w8a4_modiff_l0", "W8A4 MoDiff\nconv only"),
         ("w8a4_modiff_l1", "W8A4 MoDiff\nconv+proj"),
         ("int4_baseline", "W4A4 PTQ"),
         ("int4_modiff", "W4A4 +MoDiff"),
         ("int4_modiff_l0", "W4A4 MoDiff\nconv only"),
         ("int4_modiff_l1", "W4A4 MoDiff\nconv+proj")]


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def present(root):
    """The subset of ORDER that actually has images, so a partial run still produces sheets."""
    out = []
    for folder, label in ORDER:
        d = os.path.join(root, folder)
        if os.path.isdir(d) and any(f.endswith(".png") for f in os.listdir(d)):
            out.append((folder, label, sorted(f for f in os.listdir(d) if f.endswith(".png"))))
    return out


def comparison(root, cols, indices, out, tile=256):
    hdr, pad = 54, 4   # two label lines
    W = len(cols) * (tile + pad) + pad
    H = hdr + len(indices) * (tile + pad) + pad
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    f = font(19)
    for c, (folder, label, files) in enumerate(cols):
        x = pad + c * (tile + pad)
        # Centre the label over its column rather than left-aligning: at 7+ columns a left-aligned
        # label sits closer to the neighbouring column than to its own. Two-line labels exist because
        # the l0/l1 modes need both the precision and which layers carry MoDiff.
        for li, line in enumerate(label.split("\n")):
            try:
                w = dr.textlength(line, font=f)
            except Exception:
                w = len(line) * 9
            dr.text((x + (tile - w) / 2, 6 + li * 22), line, fill=(235, 235, 235), font=f)
    for r, idx in enumerate(indices):
        y = hdr + pad + r * (tile + pad)
        for c, (folder, label, files) in enumerate(cols):
            if idx >= len(files):
                continue
            im = Image.open(os.path.join(root, folder, files[idx])).convert("RGB")
            if im.size != (tile, tile):
                im = im.resize((tile, tile), Image.LANCZOS)
            sheet.paste(im, (pad + c * (tile + pad), y))
    sheet.save(out, "PNG")
    return out


def grid(root, folder, label, files, out, tile=128, limit=100):
    files = files[:limit]
    n = len(files)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    hdr, pad = 30, 2
    sheet = Image.new("RGB", (cols * (tile + pad) + pad, hdr + rows * (tile + pad) + pad),
                      (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad + 2, 7), f"{label}   ({n} samples)", fill=(235, 235, 235), font=font(18))
    for i, fn in enumerate(files):
        im = Image.open(os.path.join(root, folder, fn)).convert("RGB").resize(
            (tile, tile), Image.LANCZOS)
        sheet.paste(im, (pad + (i % cols) * (tile + pad),
                         hdr + pad + (i // cols) * (tile + pad)))
    sheet.save(out, "PNG")
    return out


def cross(root1, root2, l1, l2, indices, out, tile=256):
    """fp16 once, then each MoDiff arm from both batches side by side."""
    cols = [(root1, "fp16", "fp16")]
    for folder, label in ORDER:
        if not folder.endswith("_modiff"):
            continue
        if os.path.isdir(os.path.join(root1, folder)) and os.path.isdir(os.path.join(root2, folder)):
            cols.append((root1, folder, f"{label}\n{l1}"))
            cols.append((root2, folder, f"{label}\n{l2}"))
    if len(cols) < 2:
        return None
    hdr, pad = 52, 4
    sheet = Image.new("RGB", (len(cols) * (tile + pad) + pad,
                              hdr + len(indices) * (tile + pad) + pad), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    f = font(17)
    for c, (root, folder, label) in enumerate(cols):
        x = pad + c * (tile + pad)
        for li, line in enumerate(label.split("\n")):
            try:
                w = dr.textlength(line, font=f)
            except Exception:
                w = len(line) * 8
            dr.text((x + (tile - w) / 2, 6 + li * 21), line, fill=(235, 235, 235), font=f)
    for r, idx in enumerate(indices):
        y = hdr + pad + r * (tile + pad)
        for c, (root, folder, label) in enumerate(cols):
            files = sorted(x for x in os.listdir(os.path.join(root, folder)) if x.endswith(".png"))
            if idx >= len(files):
                continue
            im = Image.open(os.path.join(root, folder, files[idx])).convert("RGB")
            if im.size != (tile, tile):
                im = im.resize((tile, tile), Image.LANCZOS)
            sheet.paste(im, (pad + c * (tile + pad), y))
    sheet.save(out, "PNG")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/review_2026-08-10")
    ap.add_argument("--dir2", default="", help="second batch for the cross-batch comparison")
    ap.add_argument("--label1", default="LINEAR=0")
    ap.add_argument("--label2", default="LINEAR=1")
    ap.add_argument("--rows", type=int, default=6, help="samples per comparison sheet")
    ap.add_argument("--sheets", type=int, default=3, help="how many comparison sheets")
    args = ap.parse_args()

    cols = present(args.dir)
    if not cols:
        print(f"no images under {args.dir}")
        return 1
    print(f"{len(cols)} modes: " + ", ".join(l for _, l, _ in cols))
    sheets = os.path.join(args.dir, "sheets")
    os.makedirs(sheets, exist_ok=True)

    made = []
    for s in range(args.sheets):
        idx = list(range(s * args.rows, (s + 1) * args.rows))
        made.append(comparison(args.dir, cols, idx,
                               os.path.join(sheets, f"comparison_{s + 1}.png")))
    for folder, label, files in cols:
        made.append(grid(args.dir, folder, label, files,
                         os.path.join(sheets, f"grid_{folder}.png")))
    if args.dir2:
        for s2 in range(args.sheets):
            idx = list(range(s2 * args.rows, (s2 + 1) * args.rows))
            m = cross(args.dir, args.dir2, args.label1, args.label2, idx,
                      os.path.join(sheets, f"linear_{s2 + 1}.png"))
            if m:
                made.append(m)
    for m in made:
        print(f"  {m}  {os.path.getsize(m) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
