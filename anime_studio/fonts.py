"""Find a usable TrueType font on Windows, Linux or macOS."""

import re
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_LATIN = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
_CJK = [
    "C:/Windows/Fonts/YuGothB.ttc",
    "C:/Windows/Fonts/meiryob.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/msyhbd.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
]
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]")


def needs_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


@lru_cache(maxsize=64)
def get_font(size: int, cjk: bool = False):
    candidates = (_CJK + _LATIN) if cjk else (_LATIN + _CJK)
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def font_for(text: str, size: int):
    return get_font(size, needs_cjk(text))


def wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    """Greedy word wrap; also breaks long CJK runs without spaces."""
    lines = []
    for paragraph in text.split("\n"):
        tokens = paragraph.split(" ") if not needs_cjk(paragraph) else list(paragraph)
        joiner = " " if not needs_cjk(paragraph) else ""
        current = ""
        for token in tokens:
            trial = token if not current else current + joiner + token
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = token
        lines.append(current)
    return lines
