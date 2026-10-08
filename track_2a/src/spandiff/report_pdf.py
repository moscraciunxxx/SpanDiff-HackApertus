"""Method and report PDFs built from the markdown sources."""

from __future__ import annotations

from pathlib import Path


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_report(path: Path, source: Path | None = None) -> None:
    source = source or path.parent / "technical_report.md"
    path.write_bytes(render_styled_report(source))


def paginate(lines: list[str], per_page: int = 40, max_pages: int = 2) -> list[list[str]]:
    if not lines:
        raise ValueError("PDF has no lines")
    pages = [lines[index : index + per_page] for index in range(0, len(lines), per_page)]
    if len(pages) > max_pages:
        raise ValueError(f"PDF would exceed {max_pages} pages")
    return pages


def _pages(lines: list[str], per_page: int = 40) -> list[list[str]]:
    return paginate(lines, per_page=per_page, max_pages=2)


def build_pdf_pages(pages: list[list[str]], max_pages: int = 2) -> bytes:
    """A PDF of 1 to max_pages pages. Page objects start at 3; content streams follow."""
    if not 1 <= len(pages) <= max_pages:
        raise ValueError(f"PDF must be 1 to {max_pages} pages")
    page_ids = []
    content_ids = []
    next_id = 3
    for _ in pages:
        page_ids.append(next_id)
        next_id += 1
        content_ids.append(next_id)
        next_id += 1
    font_id = next_id
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"),
    ]
    streams: list[bytes] = []
    for lines in pages:
        commands = ["BT", "/F1 11 Tf", "48 740 Td", "16 TL"]
        for index, line in enumerate(lines):
            if index:
                commands.append("T*")
            if line == GRADED_COMMAND:
                # 8 pt keeps the full graded command inside the page margins.
                commands.append("/F1 8 Tf")
                commands.append(f"({_escape(line)}) Tj")
                commands.append("/F1 11 Tf")
            else:
                commands.append(f"({_escape(line)}) Tj")
        commands.append("ET")
        try:
            streams.append("\n".join(commands).encode("latin-1"))
        except UnicodeEncodeError as exc:
            raise ValueError("PDF text is not Latin-1") from exc
    # placeholder slots filled in order: page, content, page, content, ...
    ordered: list[bytes] = []
    for page_id, content_id, stream in zip(page_ids, content_ids, streams):
        ordered.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode(
                "ascii"
            )
        )
        ordered.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    objects.extend(ordered)
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    if len(objects) != font_id:
        raise RuntimeError("PDF object count does not match the font id")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{number} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(out)


GRADED_COMMAND = (
    "python -m scripts.evaluate_predictions_admin "
    "data/evaluation/encoder_predictions/dev/SpanDiff_admin_ --split dev"
)


def _wrap(text: str, width: int) -> list[str]:
    if GRADED_COMMAND in text:
        before, after = text.split(GRADED_COMMAND, 1)
        lines: list[str] = []
        if before.strip():
            lines.extend(_wrap(before.strip(), width))
        # One PDF line so the graded command stays contiguous in the file bytes.
        lines.append(GRADED_COMMAND)
        if after.strip():
            lines.extend(_wrap(after.strip(), width))
        return lines
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def plain_lines(source: Path, width: int = 90) -> list[str]:
    lines: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        if text.startswith("#"):
            text = text.lstrip("#").strip()
        text = text.replace("**", "").replace("`", "")
        if not text or set(text) <= {"|", "-", " "}:
            continue
        lines.extend(_wrap(text, width))
    return lines


def method_lines(source: Path) -> list[str]:
    lines: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        lines.extend(_wrap(text, 90))
    return lines


def write_method(path: Path, source: Path | None = None) -> None:
    source = source or path.parent / "docs" / "method.md"
    pages = _pages(method_lines(source))
    path.write_bytes(build_pdf_pages(pages))


def _jpeg(png: Path) -> tuple[bytes, int, int]:
    import subprocess
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    handle.close()
    dest = Path(handle.name)
    subprocess.run(
        ["sips", "-s", "format", "jpeg", str(png), "--out", str(dest)],
        check=True,
        capture_output=True,
    )
    info = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(dest)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    width = height = 0
    for line in info.splitlines():
        if "pixelWidth" in line:
            width = int(line.split()[-1])
        if "pixelHeight" in line:
            height = int(line.split()[-1])
    data = dest.read_bytes()
    dest.unlink()
    return data, width, height


def _blocks(source: Path) -> list[tuple]:
    """Headings, paragraphs, tables, and image markers from the report markdown."""
    blocks: list[tuple] = []
    paragraph: list[str] = []
    table: list[list[str]] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(("p", " ".join(paragraph)))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            blocks.append(("table", [row[:] for row in table]))
            table.clear()

    for raw in source.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            flush_paragraph()
            flush_table()
            continue
        if text.startswith("|"):
            flush_paragraph()
            cells = [cell.strip() for cell in text.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            table.append(cells)
            continue
        flush_table()
        if text.startswith("#"):
            flush_paragraph()
            level = len(text) - len(text.lstrip("#"))
            blocks.append(("h", level, text.lstrip("#").strip()))
            continue
        if text.endswith(".png") or ".png" in text and text.startswith("!"):
            flush_paragraph()
            name = text.split("(")[-1].rstrip(")") if "(" in text else text
            blocks.append(("img", Path(name).name))
            continue
        paragraph.append(text.replace("**", "").replace("`", ""))
    flush_paragraph()
    flush_table()
    return blocks


def _wrap_width(text: str, size: float, bold: bool, limit: float) -> list[str]:
    factor = 0.56 if bold else 0.50
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if len(trial) * size * factor <= limit:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_styled_report(source: Path, max_pages: int = 6) -> bytes:
    """A short report with a drawn table, bold headings, and the layer curve."""
    content_width = 516.0
    left = 48.0
    bottom = 46.0
    blocks = _blocks(source)
    images: dict[str, tuple[bytes, int, int]] = {}
    pages: list[list[str]] = []
    commands: list[str] = []
    y = 748.0

    def new_page() -> None:
        nonlocal commands, y
        if commands:
            pages.append(commands)
        if len(pages) >= max_pages:
            raise ValueError(f"report PDF would exceed {max_pages} pages")
        commands = []
        y = 748.0

    def ensure(height: float) -> None:
        nonlocal y
        if y - height < bottom:
            new_page()

    def draw_text(text: str, size: float, bold: bool, x: float, baseline: float) -> None:
        font = "F2" if bold else "F1"
        commands.append(f"BT /{font} {size:.1f} Tf {x:.2f} {baseline:.2f} Td ({_escape(text)}) Tj ET")

    new_page()
    for kind, *payload in blocks:
        if kind == "h":
            level, title = payload
            size = 16.0 if level == 1 else 13.0
            ensure(size + 12)
            y -= 8 if level > 1 else 0
            draw_text(title, size, True, left, y - size)
            y -= size + 8
        elif kind == "p":
            (text,) = payload
            for line in _wrap_width(text, 10.0, False, content_width):
                ensure(14)
                draw_text(line, 10, False, left, y - 10)
                y -= 13
            y -= 4
        elif kind == "table":
            (rows,) = payload
            cols = max(len(row) for row in rows)
            if cols == 5:
                widths = [220.0, 74.0, 74.0, 74.0, 74.0]
            else:
                widths = [content_width / cols] * cols
            row_h = 16.0
            ensure(row_h * len(rows) + 4)
            for index, row in enumerate(rows):
                x = left
                top = y
                for col in range(cols):
                    width = widths[col]
                    if index == 0:
                        commands.append(
                            f"0.90 0.93 0.96 rg {x:.2f} {top - row_h:.2f} {width:.2f} {row_h:.2f} re f"
                        )
                    commands.append(
                        f"0.45 0.50 0.55 RG 0.6 w {x:.2f} {top - row_h:.2f} {width:.2f} {row_h:.2f} re S"
                    )
                    commands.append("0 0 0 rg")
                    label = row[col] if col < len(row) else ""
                    draw_text(label, 8.5, index == 0, x + 3, top - 11.5)
                    x += width
                y -= row_h
            commands.append("0 0 0 rg")
            y -= 8
        elif kind == "img":
            (name,) = payload
            png = source.parent / "docs" / name
            if not png.is_file():
                png = source.parent / name
            if not png.is_file():
                raise ValueError(f"report image is missing: {name}")
            data, width_px, height_px = _jpeg(png)
            images[name] = (data, width_px, height_px)
            display_w = content_width
            display_h = display_w * height_px / width_px
            if display_h > 250:
                display_h = 250
                display_w = display_h * width_px / height_px
            ensure(display_h + 8)
            y -= display_h
            alias = f"Im{len(images)}"
            commands.append(
                f"q {display_w:.2f} 0 0 {display_h:.2f} {left:.2f} {y:.2f} cm /{alias} Do Q"
            )
            y -= 10
    if commands:
        pages.append(commands)
    # new_page() already appended an empty starter before the first real commands
    # were added onto that same list, so drop a leading empty page if present.
    pages = [page for page in pages if page]
    if not 1 <= len(pages) <= max_pages:
        raise ValueError(f"report PDF has {len(pages)} pages")

    # Rebuild image aliases in the order they were registered.
    # The content stream already used Im1, Im2, ... in registration order.
    image_items = list(images.items())
    page_count = len(pages)
    # objects: catalog, pages, then page/content pairs, images, two fonts
    page_ids = []
    content_ids = []
    next_id = 3
    for _ in pages:
        page_ids.append(next_id)
        next_id += 1
        content_ids.append(next_id)
        next_id += 1
    image_ids = []
    for _ in image_items:
        image_ids.append(next_id)
        next_id += 1
    font1 = next_id
    font2 = next_id + 1
    xobject = " ".join(
        f"/Im{index + 1} {image_id} 0 R" for index, image_id in enumerate(image_ids)
    )
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"),
    ]
    for page_id, content_id, stream_lines in zip(page_ids, content_ids, pages):
        stream = "\n".join(stream_lines).encode("latin-1")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font1} 0 R /F2 {font2} 0 R >> "
                f"/XObject << {xobject} >> >> >>"
            ).encode("ascii")
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    for (data, width_px, height_px), _image_id in zip(
        (item[1] for item in image_items), image_ids
    ):
        objects.append(
            (
                f"<< /Type /XObject /Subtype /Image /Width {width_px} /Height {height_px} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\n"
                "stream\n"
            ).encode("ascii")
            + data
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{number} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(out)
