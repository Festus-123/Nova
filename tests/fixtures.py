"""Small, generated document fixtures (no binary files checked in)."""

import io
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def page_image(lines: List[str], fmt: str = "PNG", size=(1000, 1300), font_size: int = 34) -> bytes:
    """Render text lines onto a white page and encode it as PNG/JPEG/WEBP."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black", font=font)
        y += int(font_size * 1.6)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_pdf(pages: List[List[str]], title: Optional[str] = None) -> bytes:
    """Build a minimal valid PDF with a real text layer (one list of lines per page)."""
    objects: List[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog_id = add(b"")  # placeholder
    pages_id = add(b"")    # placeholder
    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for lines in pages:
        ops = ["BT", "/F1 14 Tf", "50 780 Td", "18 TL"]
        for line in lines:
            ops.append(f"({_escape_pdf_text(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content_id = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(
            add(
                (
                    f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 842] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
                ).encode()
            )
        )
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    info_id = None
    if title:
        info_id = add(f"<< /Title ({_escape_pdf_text(title)}) >>".encode())

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R"
    if info_id:
        trailer += f" /Info {info_id} 0 R"
    out.write((trailer + f" >>\nstartxref\n{xref}\n%%EOF\n").encode())
    return out.getvalue()


def image_pdf(page_lines: List[List[str]]) -> bytes:
    """A 'scanned' PDF: each page is only an image (no text layer)."""
    images = [Image.open(io.BytesIO(page_image(lines))).convert("RGB") for lines in page_lines]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def docx_file(sections: List[List[str]], table_rows: Optional[List[List[str]]] = None) -> bytes:
    """DOCX whose sections are separated by explicit page breaks."""
    import docx
    from docx.enum.text import WD_BREAK

    document = docx.Document()
    for number, lines in enumerate(sections):
        for line in lines:
            document.add_paragraph(line)
        if table_rows and number == 0:
            table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
            for r, row in enumerate(table_rows):
                for c, value in enumerate(row):
                    table.cell(r, c).text = value
        if number < len(sections) - 1:
            document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
