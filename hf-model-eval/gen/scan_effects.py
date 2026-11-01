"""Render a PDF page to PNG and apply a scanner / fax / phone-photo degradation.

Shared by every document generator. Usage:
    python gen/scan_effects.py in.pdf out.png <profile> [seed]
Profiles (assign by doc number: PROFILES[(n-1) % 4]):
    clean       flatbed scan, 200 dpi, light noise
    skew_noise  slightly rotated, blurred, JPEG artifacts (multi-page feeder scan)
    fax         bilevel, downsampled, speckled (fax / cheap copier)
    photo       phone photo: perspective, uneven light, blur
"""
import io
import random
import sys

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageEnhance, ImageFilter

PROFILES = ["clean", "skew_noise", "fax", "photo"]
DPI = 200


def render_pdf(pdf_path, dpi=DPI, page=0):
    pdf = pdfium.PdfDocument(pdf_path)
    img = pdf[page].render(scale=dpi / 72).to_pil().convert("RGB")
    pdf.close()
    return img


def _noise(img, sigma, seed):
    arr = np.array(img).astype(np.float32)
    arr += np.random.default_rng(seed).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _perspective_coeffs(src, dst):
    m = []
    for (x, y), (u, v) in zip(src, dst):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(m, dtype=np.float64)
    b = np.array([c for pt in dst for c in pt], dtype=np.float64)
    return np.linalg.solve(a, b).tolist()


def degrade(img, profile, seed=0):
    rnd = random.Random(seed)
    w, h = img.size
    if profile == "clean":
        img = img.filter(ImageFilter.GaussianBlur(0.4))
        return _noise(img, 3, seed)
    if profile == "skew_noise":
        img = img.rotate(rnd.uniform(-2.0, 2.0), resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        img = img.filter(ImageFilter.GaussianBlur(0.7))
        img = _noise(img, 9, seed)
        return _jpeg(img, 55)
    if profile == "fax":
        small = img.convert("L").resize((int(w * 0.62), int(h * 0.62)), Image.BILINEAR)
        small = ImageEnhance.Contrast(small).enhance(0.85)
        arr = np.array(small).astype(np.float32)
        arr += np.random.default_rng(seed).normal(0, 16, arr.shape)
        bw = (np.clip(arr, 0, 255) > 150).astype(np.uint8) * 255
        return Image.fromarray(bw).resize((w, h), Image.NEAREST).convert("RGB")
    if profile == "photo":
        dx = [rnd.uniform(0.005, 0.035) * w for _ in range(4)]
        dy = [rnd.uniform(0.005, 0.035) * h for _ in range(4)]
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [(dx[0], dy[0]), (w - dx[1], dy[1]), (w - dx[2], h - dy[2]), (dx[3], h - dy[3])]
        coeffs = _perspective_coeffs(dst, src)  # output -> input mapping
        img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(118, 108, 98))
        grad_x = np.linspace(0.78, 1.04, w).reshape(1, w, 1)
        grad_y = np.linspace(1.0, 0.86, h).reshape(h, 1, 1)
        arr = np.clip(np.array(img).astype(np.float32) * grad_x * grad_y, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.9))
        img = _noise(img, 6, seed)
        return _jpeg(img, 70)
    raise ValueError(f"unknown profile {profile}")


def pdf_to_scan(pdf_path, png_path, profile, seed=0, dpi=DPI):
    img = degrade(render_pdf(pdf_path, dpi=dpi), profile, seed)
    img.save(png_path, "PNG", optimize=True)
    return png_path


if __name__ == "__main__":
    pdf, png, prof = sys.argv[1:4]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(pdf_to_scan(pdf, png, prof, seed))
