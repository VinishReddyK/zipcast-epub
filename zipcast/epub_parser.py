"""epub parsing and chapter extraction."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import ebooklib  # type: ignore
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning  # type: ignore
from ebooklib import epub  # type: ignore

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CONTENT_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"]
SKIP_TAGS = ["script", "style", "meta", "head", "link", "noscript", "nav", "header", "footer"]
MIN_CHAPTER_WORDS = 50
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class Chapter:
    """single chapter extracted from an epub."""

    index: int
    title: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def filename_base(self) -> str:
        """sanitized filename without extension."""
        safe_title = UNSAFE_FILENAME_CHARS.sub("_", self.title)
        safe_title = safe_title.strip().replace(" ", "_")[:50]
        return f"{self.index:02d}_{safe_title}"


@dataclass
class Book:
    """parsed epub book with metadata and chapters."""

    title: str
    author: str
    language: str
    chapters: list[Chapter]

    def to_metadata(self) -> dict:
        """return metadata dict for serialization (without chapter text)."""
        return {
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "chapters": [
                {
                    "index": c.index,
                    "title": c.title,
                    "filename_base": c.filename_base,
                    "word_count": c.word_count,
                }
                for c in self.chapters
            ],
        }


def extract_text_from_html(html_content: bytes) -> str:
    """convert html content to clean plain text, dropping nav/script/style noise."""
    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()

    paragraphs = []
    for tag in soup.find_all(CONTENT_TAGS):
        if tag.find(CONTENT_TAGS):
            # container with nested content tags: only pull text from the
            # non-content children so we don't double-count nested paragraphs
            parts = []
            for child in tag.children:
                name = getattr(child, "name", None)
                if name is None or name not in CONTENT_TAGS:
                    parts.append(child.get_text())
            text = " ".join("".join(parts).split())
            if text:
                paragraphs.append(text)
            continue

        text = " ".join(tag.get_text().split())
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def extract_title_from_html(html_content: bytes) -> str | None:
    """extract a chapter title from html (title tag, then first heading)."""
    soup = BeautifulSoup(html_content, "lxml")

    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return title_tag.string.strip()

    for tag in ["h1", "h2", "h3"]:
        heading = soup.find(tag)
        if heading:
            text = heading.get_text(strip=True)
            if text:
                return text

    return None


def extract_cover(book: epub.EpubBook) -> bytes | None:
    """extract cover image from epub, returns image bytes or None."""
    for cid in ["cover", "cover-image", "coverimage"]:
        if item := book.get_item_with_id(cid):
            return cast(bytes, item.get_content())

    if cover_meta := book.get_metadata("OPF", "cover"):
        if item := book.get_item_with_id(str(cover_meta[0][0])):
            return cast(bytes, item.get_content())

    images = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))
    for item in images:
        if "cover" in item.get_name().lower():
            return cast(bytes, item.get_content())

    return cast(bytes, images[0].get_content()) if images else None


def parse_epub(path: Path) -> tuple[Book, bytes | None]:
    """parse an epub file and extract all chapters with metadata and cover."""
    eb = epub.read_epub(str(path), options={"ignore_ncx": True})

    def get_meta(key: str, default: str = "") -> str:
        m = eb.get_metadata("DC", key)
        return str(m[0][0]) if m and m[0] else default

    book = Book(
        title=get_meta("title", path.stem),
        author=get_meta("creator", "Unknown"),
        language=get_meta("language", "en"),
        chapters=[],
    )

    for item in eb.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = cast(bytes, item.get_content())
        text = extract_text_from_html(content)
        if len(text.split()) < MIN_CHAPTER_WORDS:
            continue

        idx = len(book.chapters) + 1
        title = extract_title_from_html(content) or f"Chapter {idx}"
        book.chapters.append(Chapter(index=idx, title=title, text=text))

    return book, extract_cover(eb)
