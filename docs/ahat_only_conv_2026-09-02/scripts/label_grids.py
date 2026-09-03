"""Rebuild the sample comparison grids with the arm/mode burned in as a left-hand label strip,
so each image is self-describing instead of needing the surrounding text.

Labels are ENGLISH on purpose: no CJK font is installed (fc-list finds none), and DejaVuSans
renders Chinese as tofu boxes."""
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
D="docs/ahat_only_conv_2026-09-02/samples"
FP=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/DejaVuSans.ttf")
FPB=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
F=ImageFont.truetype(FP,26); FB=ImageFont.truetype(FPB,30)

GRIDS={
 "labeled_bits_boundary.png": ("W4A4  —  where does a_hat bit-width actually break?   (B=32, all SIM except row 2)", [
   ("int4_ahat0",        "a_hat fp16",       "reference,  eta_cum ~ 0.001"),
   ("int4_ahat32",       "a_hat i8 B=32",    "REAL kernel,  eta_cum 0.053"),
   ("int4_ahat32_sim7",  "a_hat 7-bit B=32", "eta_cum 0.110"),
   ("int4_ahat32_sim6",  "a_hat 6-bit B=32", "eta_cum 0.254"),
   ("int4_ahat32_sim5",  "a_hat 5-bit B=32", "eta_cum 0.657"),
   ("int4_ahat32_sim4",  "a_hat 4-bit B=32", "eta_cum 1.982"),
   ("int4_ahat32_sim3",  "a_hat 3-bit B=32", "eta_cum ~ 6")]),
 "labeled_ahat_bits.png": ("W4A4  —  a_hat bit-width sweep   (batch 128, 50 DDIM, seed 1234)", [
   ("int4_ahat0",        "a_hat fp16", "reference"),
   ("int4_ahat32",       "a_hat i8 B=32", "REAL kernel"),
   ("int4_ahat32_sim8",  "a_hat 8-bit B=32", "SIM  -  matches the row above => sim is trustworthy"),
   ("int4_ahat32_sim4",  "a_hat 4-bit B=32", "SIM  -  collapsed"),
   ("int4_ahat32_sim3",  "a_hat 3-bit B=32", "SIM  -  pure noise")]),
 "labeled_blocks.png": ("a_hat block-size sweep, both precisions   (batch 128, 50 DDIM, seed 1234)", [
   ("int8_ahat0",  "W8A8  a_hat fp16", "1403 MB cache"),
   ("int8_ahat16", "W8A8  a_hat i8 B=16", "877 MB"),
   ("int8_ahat32", "W8A8  a_hat i8 B=32", "789 MB   <-- recommended"),
   ("int8_ahat64", "W8A8  a_hat i8 B=64", "745 MB"),
   ("int4_ahat0",  "W4A4  a_hat fp16", "1403 MB"),
   ("int4_ahat16", "W4A4  a_hat i8 B=16", "877 MB"),
   ("int4_ahat32", "W4A4  a_hat i8 B=32", "789 MB   <-- recommended"),
   ("int4_ahat64", "W4A4  a_hat i8 B=64", "745 MB")]),
 "labeled_i4_blocks.png": ("W4A4  —  4-bit a_hat at three block sizes   (all SIMULATED)", [
   ("int4_ahat32",      "a_hat i8 B=32", "REAL, 1.125 B/elem  -  ok"),
   ("int4_ahat32_sim4", "a_hat 4-bit B=32", "0.625 B/elem  -  collapsed"),
   ("int4_ahat16_sim4", "a_hat 4-bit B=16", "0.750 B/elem  -  still unusable"),
   ("int4_ahat8_sim4",  "a_hat 4-bit B=8",  "1.000 B/elem  -  still unusable")]),
}
LW, TH = 620, 62
for out,(title,rows) in GRIDS.items():
    tiles=[]
    for f,_,_ in rows:
        im=Image.open(f"{D}/{f}.png").convert("RGB")
        tiles.append(im.crop((0,0,im.width,im.height//2)))   # top row of that arm's 4x2 grid
    w=min(t.width for t in tiles); h=min(t.height for t in tiles)
    canvas=Image.new("RGB",(LW+w, TH+h*len(rows)),"white")
    d=ImageDraw.Draw(canvas)
    d.text((16,18), title, font=FB, fill=(11,11,11))
    for i,(t,(_,lab,note)) in enumerate(zip(tiles,rows)):
        y=TH+i*h
        canvas.paste(t.crop((0,0,w,h)),(LW,y))
        d.text((16,y+h//2-34), lab, font=FB, fill=(11,11,11))
        d.text((16,y+h//2+6), note, font=F, fill=(82,81,78))
        d.line([(0,y),(LW+w,y)], fill=(200,200,196), width=2)
    canvas.save(f"{D}/{out}")
    print(f"{D}/{out}  {canvas.size}")
