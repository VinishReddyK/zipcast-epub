"""zipcast: parse an epub into a clean text/metadata bundle, zipped for upload to Colab."""

from .epub_parser import Book, Chapter, parse_epub
from .pack import pack_to_zip

__all__ = ["Book", "Chapter", "parse_epub", "pack_to_zip"]
