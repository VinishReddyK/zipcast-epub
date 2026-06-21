"""zipcast: epub parsing shared between the (optional) local CLI and the Colab notebook."""

from .epub_parser import Book, Chapter, parse_epub

__all__ = ["Book", "Chapter", "parse_epub"]
