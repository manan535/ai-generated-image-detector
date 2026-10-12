"""Extract GLIDE from tiny-genimage.zip into D:/IAD_data/genimage_glide/FAKE + REAL."""
import os, zipfile, shutil

ZIP_PATH = "D:/IAD_data/tiny-genimage.zip"
OUT_DIR = "D:/IAD_data/genimage_glide"
FAKE_DIR = os.path.join(OUT_DIR, "FAKE")
REAL_DIR = os.path.join(OUT_DIR, "REAL")

os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)

zf = zipfile.ZipFile(ZIP_PATH, "r")
ai_files = sorted([n for n in zf.namelist() if "imagenet_glide/train/ai/" in n and not n.endswith("/")])
nat_files = sorted([n for n in zf.namelist() if "imagenet_glide/train/nature/" in n and not n.endswith("/")])

print(f"Extracting {len(ai_files)} FAKE and {len(nat_files)} REAL GLIDE images...")

for i, name in enumerate(ai_files):
    ext = os.path.splitext(name)[1]
    dest = os.path.join(FAKE_DIR, f"glide_fake_{i:05d}{ext}")
    with zf.open(name) as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)

for i, name in enumerate(nat_files):
    ext = os.path.splitext(name)[1]
    dest = os.path.join(REAL_DIR, f"glide_real_{i:05d}{ext}")
    with zf.open(name) as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)

zf.close()

fake_count = len(os.listdir(FAKE_DIR))
real_count = len(os.listdir(REAL_DIR))
print(f"Done. FAKE: {fake_count}, REAL: {real_count}")
