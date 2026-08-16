"""Export N real LSUN church_outdoor images from the LMDB as the FID reference set.

LSUN LMDB stores each image as encoded JPEG/WebP bytes under an opaque key, so the export has to
decode and resize. Resizing convention matters for FID and is stated rather than left implicit:

  * center crop to the short side, then resize to 256x256 with bicubic + antialias.
    The LDM LSUN-Churches model generates 256x256, so the reference must be 256x256 too. Feeding
    FID a reference at native resolution and samples at 256 measures the resize, not the model.
  * saved as PNG, i.e. losslessly. Saving the reference as JPEG would bake an encoder's artifacts
    into the reference statistics; pytorch_fid reads whatever is in the folder.

Both choices are applied identically to the generated samples (see generate_fid_samples.py), which
is the property that actually matters: FID compares two Inception feature distributions, and any
preprocessing difference between the two sides shows up as distance.
"""
import argparse
import io
import os
import sys

#: lmdb was missing on 2026-08-16 and this script is the prerequisite for an ~8 h B4 run, so the
#: check goes BEFORE the import that would otherwise be the error message.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..")))
from integration.utils.preflight import preflight, LMDB
preflight(*LMDB, what="export_lsun_reference.py")
import lmdb
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--lmdb", default="/workspace/lsun_dl/church_outdoor_train_lmdb")
ap.add_argument("--out", default="/workspace/fid/real")
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--size", type=int, default=256)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)


def center_square(im):
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def main():
    env = lmdb.open(a.lmdb, map_size=1 << 40, readonly=True, lock=False,
                    readahead=False, meminit=False)
    written = 0
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for _, val in cur:
            if written >= a.n:
                break
            try:
                im = Image.open(io.BytesIO(val)).convert("RGB")
            except Exception:
                continue                                  # a handful of entries fail to decode
            im = center_square(im).resize((a.size, a.size), Image.BICUBIC)
            im.save(os.path.join(a.out, f"{written:06d}.png"), "PNG")
            written += 1
            if written % 1000 == 0:
                print(f"  {written}/{a.n}", flush=True)
    print(f"wrote {written} reference images to {a.out}")
    return 0 if written == a.n else 1


if __name__ == "__main__":
    sys.exit(main())
