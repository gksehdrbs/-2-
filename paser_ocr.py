# -*- coding: utf-8 -*-
# paser_ocr.py
# HWP/HWPX/DOCX → JSON(lines)
# - 본문 텍스트 + 이미지 내 텍스트(OCR) 포함
# - HWP: BinData 전체 스캔(중간 오프셋도), zlib 재시도, 다중 이미지 추출
# - EMF/WMF 벡터 → (wand/magick/inkscape) 래스터화 후 OCR
# - 이미지 저장 없음(메모리 OCR), 필요시 임시파일은 즉시 삭제
# - 캡션/원본정보/페이지라벨 필터링

from __future__ import annotations

import sys
import struct
import zlib
import json
import re
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

# ----- 콘솔 UTF-8 -----
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ----- 선택 의존성 -----
_HAS_OLE = False
try:
    import olefile
    _HAS_OLE = True
except Exception:
    pass

_HAS_DOCX = False
try:
    from docx import Document
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    _HAS_DOCX = True
except Exception:
    pass

_HAS_OCR_STACK = False
try:
    import numpy as _np  # noqa
    import cv2 as _cv2   # noqa
    import easyocr
    _HAS_OCR_STACK = True
except Exception:
    pass

# 벡터 래스터화 도구
_WAND_AVAILABLE = False
try:
    from wand.image import Image as _WandImage  # requires ImageMagick
    _WAND_AVAILABLE = True
except Exception:
    pass

# ----- 시그니처/상수 -----
OLE_SIG = b"\xD0\xCF\x11\xE0"
ZIP_SIG = b"PK\x03\x04"
HWPTAG_PARA_TEXT = 66

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
VECTOR_EXTS = {".emf", ".wmf"}


# ================= 공통 유틸 =================
def detect_container(p: Path) -> str:
    if not p.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
    head = p.read_bytes()[:8]
    ext = p.suffix.lower()
    if head.startswith(OLE_SIG) or ext == ".hwp":
        return "HWP5"
    if head.startswith(ZIP_SIG) or ext == ".hwpx":
        return "HWPX"
    if ext == ".docx":
        return "DOCX"
    return "UNKNOWN"


def split_lines_like_paragraphs(text: str) -> list[str]:
    out = []
    for s in re.split(r"[\n]", text):
        s = s.strip()
        if not s:
            continue
        if not re.search(r"[.!?]$", s):
            s += "."
        out.append(s)
    return out


def hard_wrap_lines(lines: list[str], width: int) -> list[str]:
    if width <= 0:
        return lines
    wrapped: list[str] = []
    for ln in lines:
        s = ln
        while len(s) > width:
            cut = s.rfind(" ", 0, width)
            if cut < int(width * 0.6):
                for p in (". ", ") ", "] ", "· ", "• ", ", "):
                    pos = s.rfind(p, 0, width)
                    if pos > 0:
                        cut = pos + len(p) - 1
                        break
            if cut <= 0:
                cut = width
            wrapped.append(s[:cut].rstrip())
            s = s[cut:].lstrip()
        if s:
            wrapped.append(s)
    return wrapped


# ================= 라인 필터(캡션/라벨 제거) =================
import re as _re

_CAPTION_RE = _re.compile(
    r"""(?xs)
    ^\s*그림입니다\.\s*
    (?:원본\s*그림의\s*이름:\s*.+?\s*)?
    (?:원본\s*그림의\s*크기:\s*가로\s*\d+\s*pixel,\s*세로\s*\d+\s*pixel\.?\s*)?
    $
    """
)
_CAPTION_PARTS = [
    _re.compile(r"^\s*그림입니다\.\s*$"),
    _re.compile(r"^\s*원본\s*그림의\s*이름:\s*.+$"),
    _re.compile(r"^\s*원본\s*그림의\s*크기:\s*가로\s*\d+\s*pixel,\s*세로\s*\d+\s*pixel\.?\s*$"),
]
_PAGE_LABEL_RE = _re.compile(r"^\s*(?:\d{1,3}|[IVXLCDMⅰ-ⅻⅠ-Ⅻ])\s*$", _re.IGNORECASE)


def _is_noise_line(s: str) -> bool:
    if not s:
        return True
    t = s.strip().replace("\\r", "\n").replace("\\n", "\n")
    if _CAPTION_RE.match(t):
        return True
    for rx in _CAPTION_PARTS:
        if rx.match(t):
            return True
    if _PAGE_LABEL_RE.match(t):
        return True
    return False


def _post_filter_lines(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        ln2 = (ln or "").strip()
        if not ln2:
            continue
        if _is_noise_line(ln2):
            continue
        out.append(ln2)
    return out


# ================= OCR(메모리) =================
def init_easyocr(langs: list[str]):
    if not _HAS_OCR_STACK:
        raise RuntimeError("--ocr 사용에는 easyocr, opencv-python, numpy 설치가 필요합니다.")
    return easyocr.Reader(langs, gpu=False)


def _ocr_variant_pipeline(reader, img, table_mode: bool):
    """여러 전처리 변형으로 순차 시도해 최초 성공 결과 반환."""
    import cv2
    import numpy as np

    def _post(results):
        if not results:
            return []
        return format_ocr_as_table(results) if table_mode else results

    tries = []
    tries.append(img)

    h, w = img.shape[:2]
    scale = 2.0 if max(h, w) < 1200 else 1.5
    nh, nw = int(h * scale), int(w * scale)
    if max(nh, nw) > 2000:
        r = 2000 / max(nh, nw)
        nh, nw = int(nh * r), int(nw * r)
    img_up = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    tries.append(img_up)

    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    tries.append(g1)

    g2 = cv2.GaussianBlur(g1, (3, 3), 0)
    _, th = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tries.append(th)

    ath = cv2.adaptiveThreshold(g1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    kernel = np.ones((2, 2), np.uint8)
    open1 = cv2.morphologyEx(ath, cv2.MORPH_OPEN, kernel, iterations=1)
    tries.append(open1)
    tries.append(255 - open1)

    for variant in tries:
        try:
            results = reader.readtext(variant, detail=0)
        except Exception:
            continue
        results = [t.strip() for t in results if isinstance(t, str) and len(t.strip()) >= 2]
        if results:
            return _post(results)
    return []


def ocr_bytes_to_lines(reader, img_bytes: bytes, table_mode: bool, idx: int, debug: bool = False) -> list[str]:
    import numpy as np
    import cv2
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return [f"[OCR {idx}] decode error"] if debug else []
    res = _ocr_variant_pipeline(reader, img, table_mode)
    if res:
        return ([f"[OCR {idx}] ok"] + res) if debug else res
    else:
        return [f"[OCR {idx}] no text"] if debug else []


def format_ocr_as_table(texts: list[str], header_count: int | None = None) -> list[str]:
    if not texts:
        return []
    if header_count is None:
        header_count = 0
        for t in texts:
            if any(k in t for k in ("년", "구분", "%", "월", "합계", "구성비", "금액", "증감")):
                header_count += 1
            else:
                break
        if header_count < 2:
            header_count = 4
    headers = texts[:header_count]
    rows = []
    for i in range(header_count, len(texts), header_count):
        row = texts[i:i + header_count]
        if len(row) == header_count:
            rows.append(row)
    lines = [" | ".join(headers)]
    lines.extend(" | ".join(r) for r in rows)
    return lines


# ================= 벡터 → 래스터 (EMF/WMF) =================
def _which(cmd: str):
    return shutil.which(cmd)


def rasterize_vector_to_png_bytes(blob: bytes, ext: str) -> bytes | None:
    """
    EMF/WMF를 PNG로 변환.
    우선순위: 1) wand (ImageMagick bindings) 2) 'magick' 또는 'convert' CLI 3) 'inkscape' CLI
    실패하면 None
    """
    ext = ext.lower().lstrip(".")
    if ext not in ("emf", "wmf"):
        return None

    # 1) wand
    if _WAND_AVAILABLE:
        try:
            with _WandImage(blob=blob, format=ext) as im:
                im.format = 'png'
                return im.make_blob('png')
        except Exception:
            pass

    # 임시파일 전략
    tmpdir = Path(tempfile.mkdtemp(prefix="vec2png_"))
    in_path = tmpdir / f"in.{ext}"
    out_path = tmpdir / "out.png"
    try:
        in_path.write_bytes(blob)

        # 2) magick/convert (ImageMagick)
        magick = _which("magick") or _which("magick.exe")
        convert = _which("convert") or _which("convert.exe")

        if magick:
            try:
                subprocess.run([magick, str(in_path), str(out_path)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if out_path.exists():
                    data = out_path.read_bytes()
                    return data
            except Exception:
                pass

        if convert:
            try:
                subprocess.run([convert, str(in_path), str(out_path)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if out_path.exists():
                    data = out_path.read_bytes()
                    return data
            except Exception:
                pass

        # 3) inkscape
        inkscape = _which("inkscape") or _which("inkscape.exe")
        if inkscape:
            try:
                subprocess.run([inkscape, str(in_path), "--export-type=png",
                                f"--export-filename={out_path}"],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if out_path.exists():
                    data = out_path.read_bytes()
                    return data
            except Exception:
                pass
    finally:
        # 깨끗이 정리
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    return None


# ================= HWP 텍스트/이미지 =================
def _hwp_read_fileheader(ole: "olefile.OleFileIO") -> dict:
    with ole.openstream("FileHeader") as fp:
        raw = fp.read()
    if len(raw) < 48:
        raise ValueError("FileHeader 길이 오류")
    ver, = struct.unpack("<I", raw[32:36])
    attr1, = struct.unpack("<I", raw[36:40])  # bit0=compressed, bit1=encrypted
    return {"compressed": bool(attr1 & 1), "encrypted": bool(attr1 & 2), "ver": ver}


def _hwp_iter_records(buf: bytes):
    off, n = 0, len(buf)
    while off + 4 <= n:
        (hdr,) = struct.unpack("<I", buf[off:off + 4])
        off += 4
        tag = hdr & 0x3FF
        size = (hdr >> 20) & 0xFFF
        if size == 0xFFF:
            if off + 4 > n:
                break
            (size,) = struct.unpack("<I", buf[off:off + 4])
            off += 4
        if off + size > n:
            payload = buf[off:n]
            off = n
        else:
            payload = buf[off:off + size]
            off += size
        yield tag, payload


def _zlib_if_needed(raw: bytes, expect: bool) -> bytes:
    try:
        return zlib.decompress(raw) if expect else raw
    except Exception:
        return raw


def extract_hwp_text_lines(path: Path) -> list[str]:
    if not _HAS_OLE:
        raise RuntimeError("HWP 파싱에는 olefile가 필요합니다. (pip install olefile)")
    out: list[str] = []
    with olefile.OleFileIO(str(path)) as ole:
        hdr = _hwp_read_fileheader(ole)
        if hdr["encrypted"]:
            raise RuntimeError("암호화된 HWP는 지원하지 않습니다.")
        entries = [e for e in ole.listdir(streams=True, storages=True)
                   if len(e) == 2 and e[0] == "BodyText" and e[1].startswith("Section")]
        entries.sort(key=lambda e: int("".join(ch for ch in e[1] if ch.isdigit()) or "0"))
        for e in entries:
            with ole.openstream("/".join(e)) as fp:
                raw = fp.read()
            data = _zlib_if_needed(raw, hdr["compressed"])
            for tag, payload in _hwp_iter_records(data):
                if tag == HWPTAG_PARA_TEXT:
                    s = payload.decode("utf-16le", errors="ignore").replace("\r", "\n")
                    for ln in s.split("\n"):
                        ln = "".join(ch if (ch >= " " or ch in "\t") else " " for ch in ln).strip()
                        if ln:
                            out.append(ln)
    return out


# --- 스트림 전체에서 래스터 이미지 매직 스캔 ---
_MAGIC_PATTERNS = [
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xFF\xD8\xFF", ".jpg"),
    (b"BM", ".bmp"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
    (b"RIFF", ".webp"),  # 확인은 +8에 'WEBP'
]


def _scan_embedded_images(buf: bytes) -> list[bytes]:
    imgs = []
    n = len(buf)
    i = 0
    while i < n:
        hit = None
        for sig, ext in _MAGIC_PATTERNS:
            j = buf.find(sig, i)
            if j != -1 and (hit is None or j < hit[0]):
                hit = (j, sig, ext)
        if not hit:
            break
        j, sig, ext = hit
        if ext == ".webp":
            if j + 12 > n or buf[j + 8:j + 12] != b"WEBP":
                i = j + 1
                continue
        next_j = n
        for sig2, _ in _MAGIC_PATTERNS:
            k = buf.find(sig2, j + 1)
            if k != -1:
                next_j = min(next_j, k)
        chunk = buf[j:next_j]
        if len(chunk) >= 200:
            imgs.append(chunk)
        i = next_j
    return imgs


def extract_hwp_images_bytes(path: Path, want_vectors: bool = True) -> tuple[list[bytes], list[tuple[str, bytes]]]:
    """반환: (raster_images, vector_images) ; vector_images는 [(ext, blob), ...]"""
    if not _HAS_OLE:
        return [], []
    rasters: list[bytes] = []
    vectors: list[tuple[str, bytes]] = []
    with olefile.OleFileIO(str(path)) as ole:
        for entry in ole.listdir(streams=True, storages=True):
            if not entry or entry[0] != "BinData":
                continue
            name = "/".join(entry)
            try:
                with ole.openstream(name) as fp:
                    blob = fp.read()
            except Exception:
                continue

            # 1) 원본에서 래스터 스캔
            chunks = _scan_embedded_images(blob)

            # 2) zlib 재시도
            try:
                decomp = zlib.decompress(blob)
                chunks += _scan_embedded_images(decomp)
            except Exception:
                pass

            # 3) 오프셋 쉬프트 재시도
            if not chunks:
                for off in (4, 8, 16, 32, 64, 128):
                    if len(blob) > off:
                        more = _scan_embedded_images(blob[off:])
                        if more:
                            chunks += more
                            break

            if chunks:
                for c in chunks:
                    rasters.append(c)
            elif want_vectors:
                # 간단 매직으로 EMF/WMF 추정
                if blob[:4] == b"\x01\x00\x00\x00" and b" EMF" in blob[:128]:
                    vectors.append((".emf", blob))
                elif blob[:4] == b"\xd7\xcd\xc6\x9a":
                    vectors.append((".wmf", blob))
    return rasters, vectors


# ================= HWPX 텍스트/이미지 =================
def _lname(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


_PARA_END = {"p", "para", "paragraph", "line", "li", "item", "title", "subtitle", "caption"}
_BR = {"br", "linebreak"}
_TABLE_BREAK = {"tbl", "table", "tr", "row", "tc", "cell", "th", "thead", "tbody", "tfoot"}


def extract_hwpx_text_lines(path: Path) -> list[str]:
    out: list[str] = []
    with zipfile.ZipFile(str(path)) as z:
        sections = sorted([n for n in z.namelist() if n.startswith("Contents/section") and n.endswith(".xml")])
        if not sections:
            sections = sorted([n for n in z.namelist() if n.lower().endswith(".xml")])
        for name in sections:
            try:
                root = ET.fromstring(z.read(name))
            except Exception:
                continue

            buf: list[str] = []

            def flush():
                nonlocal buf
                line = "".join(buf)
                line = re.sub(r"[ \t\u00A0]{2,}", " ", line).strip()
                if line:
                    out.append(line)
                buf = []

            stack = [root]
            while stack:
                node = stack.pop()
                if node.text:
                    buf.append(node.text)
                lname = _lname(node.tag).lower()
                if lname in _BR:
                    flush()
                if lname in _TABLE_BREAK:
                    flush()
                for child in reversed(list(node)):
                    stack.append(child)
                if node.tail:
                    buf.append(node.tail)
                if lname in _PARA_END:
                    flush()
            flush()
    return out


def extract_hwpx_images_bytes(path: Path) -> tuple[list[bytes], list[tuple[str, bytes]]]:
    rasters: list[bytes] = []
    vectors: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(str(path)) as z:
        xml_names = sorted([n for n in z.namelist() if n.startswith("Contents/section") and n.lower().endswith(".xml")])
        if not xml_names:
            xml_names = sorted([n for n in z.namelist() if n.lower().endswith(".xml")])
        refs: set[str] = set()
        for name in xml_names:
            try:
                root = ET.fromstring(z.read(name))
            except Exception:
                continue
            for node in root.iter():
                for attr in ("src", "href"):
                    v = node.attrib.get(attr)
                    if not v:
                        continue
                    vv = v.strip().lstrip("./").replace("\\", "/")
                    if vv in z.namelist():
                        refs.add(vv)

        # 우선 참조된 경로
        for ref in sorted(refs):
            ext = Path(ref).suffix.lower()
            try:
                blob = z.read(ref)
            except Exception:
                continue
            if ext in IMG_EXTS:
                rasters.append(blob)
            elif ext in VECTOR_EXTS:
                vectors.append((ext, blob))

        # 보조로 ZIP 전체 스캔(누락 방지)
        for name in z.namelist():
            if name in refs:
                continue
            ext = Path(name).suffix.lower()
            if ext in IMG_EXTS:
                try:
                    rasters.append(z.read(name))
                except Exception:
                    pass
            elif ext in VECTOR_EXTS:
                try:
                    vectors.append((ext, z.read(name)))
                except Exception:
                    pass
    return rasters, vectors


# ================= DOCX 텍스트/이미지 =================
def ensure_docx_available():
    if not _HAS_DOCX:
        raise RuntimeError("DOCX 파싱에는 python-docx가 필요합니다. (pip install python-docx)")


def extract_docx_text_lines(path: Path) -> list[str]:
    ensure_docx_available()
    doc = Document(str(path))
    out: list[str] = []
    for block in doc.element.body:
        if isinstance(block, CT_P):
            paragraph = Paragraph(block, doc)
            text = paragraph.text.strip()
            if text:
                out.extend(split_lines_like_paragraphs(text))
        elif isinstance(block, CT_Tbl):
            table = Table(block, doc)
            for row in table.rows:
                rowData = [cell.text.strip() for cell in row.cells]
                if any(rowData):
                    out.append(" | ".join(rowData))
    return out


def extract_docx_images_bytes(path: Path) -> tuple[list[bytes], list[tuple[str, bytes]]]:
    ensure_docx_available()
    doc = Document(str(path))
    rasters: list[bytes] = []
    vectors: list[tuple[str, bytes]] = []
    for rel in doc.part._rels.values():
        try:
            if rel.reltype.endswith("/image"):  # 대부분 래스터이나 간혹 EMF/WMF도 올 수 있음
                target = rel.target_part
                blob = target.blob
                # 파일 확장자 추정
                name = getattr(target, 'partname', None)
                ext = Path(str(name)).suffix.lower() if name else ""
                if ext in VECTOR_EXTS:
                    vectors.append((ext, blob))
                else:
                    rasters.append(blob)
        except Exception:
            continue
    return rasters, vectors


# ================= 통합 실행 =================
def parse_and_ocr_to_lines(
    p: Path,
    do_ocr: bool,
    langs: list[str],
    table_mode: bool,
    include_text: bool,
    ocr_debug: bool = False
):
    kind = detect_container(p)
    lines: list[str] = []

    # 1) 텍스트
    if include_text:
        if kind == "HWP5":
            lines.extend(extract_hwp_text_lines(p))
        elif kind == "HWPX":
            lines.extend(extract_hwpx_text_lines(p))
        elif kind == "DOCX":
            lines.extend(extract_docx_text_lines(p))
        else:
            raise ValueError("알 수 없는 형식입니다(.hwp/.hwpx/.docx)")

    # 2) OCR (메모리)
    if do_ocr:
        reader = init_easyocr(langs)
        if kind == "HWP5":
            rasters, vectors = extract_hwp_images_bytes(p, want_vectors=True)
        elif kind == "HWPX":
            rasters, vectors = extract_hwpx_images_bytes(p)
        elif kind == "DOCX":
            rasters, vectors = extract_docx_images_bytes(p)
        else:
            rasters, vectors = [], []

        if ocr_debug:
            lines.append(f"[OCR] images found (raster={len(rasters)}, vector={len(vectors)}) in {kind}")

        # 벡터를 가능한 한 PNG로 래스터화
        for vi, (ext, vblob) in enumerate(vectors, 1):
            png = rasterize_vector_to_png_bytes(vblob, ext)
            if png is not None:
                rasters.append(png)
                if ocr_debug:
                    lines.append(f"[OCR] vector {vi} -> raster ok ({ext})")
            else:
                if ocr_debug:
                    lines.append(f"[OCR] vector {vi} -> raster FAIL ({ext})")

        # 최종 래스터에 대해 OCR
        for i, blob in enumerate(rasters, 1):
            lines.extend(ocr_bytes_to_lines(reader, blob, table_mode, i, debug=ocr_debug))

    # 3) 후처리(캡션/라벨 제거)
    lines = _post_filter_lines(lines)
    return kind, lines


# ================= 메인 =================
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="문서 파서 + 이미지 텍스트(OCR) 포함 JSON(lines) 출력")
    ap.add_argument("input", help="입력 파일(.hwp/.hwpx/.docx)")
    ap.add_argument("--out", help="출력 JSON 경로")
    ap.add_argument("--pretty", action="store_true", help="JSON 들여쓰기")
    ap.add_argument("--hard-wrap", type=int, default=0, help="긴 줄 강제 줄바꿈 폭(0=해제)")
    ap.add_argument("--ocr", action="store_true", help="이미지 내 텍스트(OCR) 포함")
    ap.add_argument("--lang", default="ko,en", help="OCR 언어(콤마 구분, 기본 ko,en)")
    ap.add_argument("--ocr-table", action="store_true", help="OCR 결과를 표처럼 묶어 라인화")
    ap.add_argument("--no-text", action="store_true", help="본문 텍스트 제외(OCR만 포함)")
    ap.add_argument("--ocr-debug", action="store_true", help="OCR 디버그 마커(이미지 수/성공여부) lines에 포함")

    args = ap.parse_args(argv)

    p = Path(args.input).expanduser().resolve()
    langs = [x.strip() for x in args.lang.split(",") if x.strip()]

    kind, lines = parse_and_ocr_to_lines(
        p,
        do_ocr=args.ocr,
        langs=langs,
        table_mode=args.ocr_table,
        include_text=not args.no_text,
        ocr_debug=args.ocr_debug
    )

    if args.hard_wrap and args.hard_wrap > 0:
        lines = hard_wrap_lines(lines, args.hard_wrap)

    obj = {"file": str(p), "format": kind, "lines": lines}
    text = json.dumps(obj, ensure_ascii=False, indent=2 if args.pretty else None)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
