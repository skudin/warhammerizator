import argparse
import tempfile
import re
from pathlib import Path
from zipfile import ZipFile
from typing import Dict

import fb2reader
from bs4 import BeautifulSoup

from warhammerizator.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to folder with books")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    parse_books(args.input, args.output)

    print("Done.")


def parse_books(input_path: Path, output_path: Path) -> None:
    for filename in input_path.rglob("*.fb2.zip"):
        books = processing_archive(filename)

        save_books(books, output_path)

        print(f"Archive {filename.name} processing is completed.")


def processing_archive(filename: Path) -> Dict[str, str]:
    result = dict()

    with ZipFile(filename, "r") as archive:
        books_names = [name for name in archive.namelist() if name.endswith(".fb2")]
        for book_name in books_names:
            with tempfile.TemporaryDirectory() as tmp_dir:
                archive.extract(book_name, tmp_dir)

                try:
                    book = fb2reader.fb2book(Path(tmp_dir) / book_name)
                except:
                    continue

                body = book.get_body()
                clean_text = clear_text(body)

                result[book_name] = clean_text

    return result


def clear_text(text: str):
    re_combine_whitespace = re.compile(r"\s+")
    re_combine_line_break = re.compile(r"\n+")
    re_combine_footnote = re.compile(r"\[\d+]")

    soup = BeautifulSoup(text, "html.parser")
    clean_paragraphs = []
    for p in soup.find_all("p"):
        p_text = p.text

        cleaner = BeautifulSoup(p_text, "html.parser")
        for tag in cleaner.find_all():
            if tag.name in ("b", "i", "u"):
                continue

            tag.decompose()

        clean_p_text = cleaner.get_text().strip().replace("\n", "").replace("+", "").replace("\xa0", " ").replace("\t", "")
        clean_p_text = re_combine_whitespace.sub(" ", clean_p_text)
        clean_p_text = re_combine_line_break.sub("", clean_p_text)
        clean_p_text = re_combine_footnote.sub("", clean_p_text)

        clean_paragraphs.append(clean_p_text)

    return " ".join(clean_paragraphs)


def save_books(books: Dict[str, str], output_path: Path) -> None:
    for book_name, content in books.items():
        book_filename = output_path / f"{book_name}.txt"

        with open(book_filename, "w") as fp:
            fp.write(content)


if __name__ == "__main__":
    main()
