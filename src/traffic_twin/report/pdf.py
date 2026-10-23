"""docs/report/report.pdf from report.md with reportlab (D113).

markdown -> HTML (`markdown`: tables, fenced code) -> reportlab flowables. Fonts are the
DejaVu files that ship with matplotlib (portable Unicode: é ± ≤ → ×). The output is
invariant (`rl_config.invariant`): the same input on the same platform gives the same
bytes. Everything before `<!-- toc -->` is the cover page; the table of contents follows
it, and `## ` / `### ` headings become PDF outline entries. Every page carries the
footer ("Simulated data · code <sha>") and its number.

Tables: column widths come from each cell's longest unbreakable word (`minWidth`,
measured in the cell's own fonts, so inline code is measured in the mono font) and its
natural width; the text shrinks from 8 pt down to a floor of 6.5 pt, and a table that
does not fit at 6.5 pt is a build error (PdfError): select columns with `cols:` instead
of printing unreadable text.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path

import markdown
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.platypus.tableofcontents import TableOfContents

from traffic_twin.report.assemble import ReportError

TABLE_SIZES = (8.0, 7.5, 7.0, 6.5)  # pt; 6.5 is the floor (D113)
TABLE_FLOOR_PT = TABLE_SIZES[-1]
CELL_PAD = 3.0  # left and right padding of a cell, pt
DEFAULT_DPI = 144.0

INK = colors.HexColor("#1F2A33")
MUTED = colors.HexColor("#5B6770")
ACCENT = colors.HexColor("#0B6E69")  # green-wave teal
AMBER = colors.HexColor("#B57F00")
RULE = colors.HexColor("#C9D2D8")
TINT = colors.HexColor("#EEF3F5")
CODEBG = colors.HexColor("#F4F6F7")
CODE_INK = "#7A2E0E"

FONTS = {
    "Sans": "DejaVuSans.ttf",
    "Sans-B": "DejaVuSans-Bold.ttf",
    "Sans-I": "DejaVuSans-Oblique.ttf",
    "Sans-BI": "DejaVuSans-BoldOblique.ttf",
    "Mono": "DejaVuSansMono.ttf",
    "Mono-B": "DejaVuSansMono-Bold.ttf",
}


class PdfError(ReportError):
    """The PDF cannot be laid out within the rules (e.g. a table under 6.5 pt)."""


def register_fonts() -> None:
    import matplotlib

    if "Sans" in pdfmetrics.getRegisteredFontNames():
        return
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, f in FONTS.items():
        pdfmetrics.registerFont(TTFont(name, str(ttf / f)))
    pdfmetrics.registerFontFamily(
        "Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI"
    )
    pdfmetrics.registerFontFamily(
        "Mono", normal="Mono", bold="Mono-B", italic="Mono", boldItalic="Mono-B"
    )


def _style(name: str, **kw) -> ParagraphStyle:
    base = {"fontName": "Sans", "fontSize": 9.5, "leading": 13.5, "textColor": INK}
    base.update(kw)
    return ParagraphStyle(name, **base)


ST = {
    "p": _style("body", spaceAfter=5),
    "h1": _style(
        "h1",
        fontName="Sans-B",
        fontSize=20,
        leading=25,
        spaceAfter=10,
        spaceBefore=4,
        keepWithNext=1,
    ),
    "h2": _style(
        "h2",
        fontName="Sans-B",
        fontSize=14,
        leading=18,
        textColor=ACCENT,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=1,
    ),
    "h3": _style(
        "h3",
        fontName="Sans-B",
        fontSize=11,
        leading=14,
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=1,
    ),
    "h4": _style(
        "h4",
        fontName="Sans-B",
        fontSize=9.5,
        leading=13,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=1,
    ),
    "li": _style("li", leftIndent=12, bulletIndent=3, spaceAfter=2),
    "quote": _style(
        "quote",
        fontName="Sans-I",
        fontSize=9,
        leading=13,
        textColor=MUTED,
        leftIndent=10,
        borderPadding=(4, 6, 4, 6),
        backColor=TINT,
        spaceBefore=4,
        spaceAfter=8,
    ),
    "code": _style(
        "code",
        fontName="Mono",
        fontSize=7.6,
        leading=9.8,
        backColor=CODEBG,
        borderPadding=5,
        spaceBefore=3,
        spaceAfter=8,
    ),
    "cap": _style("cap", fontName="Sans-I", fontSize=8, leading=10, textColor=MUTED, spaceAfter=8),
    # cover page
    "cover-h1": _style(
        "cover-h1", fontName="Sans-B", fontSize=30, leading=36, spaceBefore=60, spaceAfter=14
    ),
    "cover-h2": _style(
        "cover-h2", fontName="Sans", fontSize=15, leading=20, textColor=ACCENT, spaceAfter=24
    ),
    "cover-p": _style("cover-p", fontSize=11, leading=16, spaceAfter=6),
    "cover-quote": _style(
        "cover-quote",
        fontName="Sans-B",
        fontSize=10,
        leading=14,
        textColor=AMBER,
        leftIndent=0,
        borderPadding=(8, 8, 8, 8),
        backColor=colors.HexColor("#FFF6DD"),
        spaceBefore=24,
        spaceAfter=10,
    ),
}


class Builder(HTMLParser):
    """Walks the HTML of `markdown` and emits flowables. Inline markup becomes
    reportlab's paragraph mini-language (<b>, <i>, <font>, <a>)."""

    INLINE = {
        "strong": ("<b>", "</b>"),
        "b": ("<b>", "</b>"),
        "em": ("<i>", "</i>"),
        "i": ("<i>", "</i>"),
    }

    def __init__(self, base: Path, width: float):
        super().__init__(convert_charrefs=True)
        self.base, self.width = Path(base), width
        self.story: list = []
        self.buf: list[str] = []
        self.block: str | None = None
        self.pre = False
        self.lists: list[list] = []  # [kind, counter]
        self.table: list[list[tuple[str, bool]]] | None = None
        self.row: list | None = None
        self.in_cell = False
        self.cell_header = False
        self.cover = True  # until <!-- toc -->
        self.toc_seen = False
        self.link_open = False

    # helpers
    def _st(self, name: str) -> ParagraphStyle:
        if self.cover and f"cover-{name}" in ST:
            return ST[f"cover-{name}"]
        return ST[name]

    def flush(self, style: str) -> None:
        txt = "".join(self.buf).strip()
        self.buf = []
        if not txt:
            return
        if style in ("h1", "h2", "h3", "h4"):
            p = Paragraph(txt, self._st(style))
            if not self.cover:
                p._toc = (int(style[1]), re.sub("<[^>]+>", "", txt))
            self.story.append(p)
        elif style == "li":
            kind, n = self.lists[-1] if self.lists else ("ul", 0)
            depth = max(0, len(self.lists) - 1)
            st = ParagraphStyle(
                f"li{depth}",
                parent=ST["li"],
                leftIndent=12 + 12 * depth,
                bulletIndent=3 + 12 * depth,
            )
            self.story.append(Paragraph(txt, st, bulletText="•" if kind == "ul" else f"{n}."))
        else:
            self.story.append(Paragraph(txt, self._st(style)))

    # parser hooks
    def handle_comment(self, data: str) -> None:
        d = data.strip()
        if d == "toc" and not self.toc_seen:
            self.toc_seen, self.cover = True, False
            toc = TableOfContents()
            toc.levelStyles = [
                _style("toc1", fontName="Sans-B", fontSize=10, leading=15),
                _style("toc2", fontSize=9, leading=12.5, leftIndent=14),
            ]
            self.story += [PageBreak(), Paragraph("Contents", ST["h1"]), toc, PageBreak()]
        elif d == "pagebreak":
            self.story.append(PageBreak())

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p"):
            if self.block not in ("li", "quote") and not self.in_cell:
                self.block = "h4" if tag in ("h5", "h6") else tag
            elif tag == "p" and self.block == "li" and self.buf:
                self.buf.append("<br/>")
        elif tag == "blockquote":
            self.block = "quote"
        elif tag in ("ul", "ol"):
            if self.block == "li":
                self.flush("li")
            self.lists.append([tag, 0])
        elif tag == "li":
            self.block = "li"
            if self.lists:
                self.lists[-1][1] += 1
        elif tag == "pre":
            self.pre = True
        elif tag == "code" and not self.pre:
            self.buf.append(f'<font face="Mono" color="{CODE_INK}">')
        elif tag in self.INLINE:
            self.buf.append(self.INLINE[tag][0])
        elif tag == "a":
            href = a.get("href", "")
            if re.match(r"^https?://", href):
                self.buf.append(f'<a href="{html.escape(href)}" color="#0B6E69">')
                self.link_open = True
            else:
                self.buf.append('<font color="#0B6E69">')
                self.link_open = False
        elif tag == "br":
            self.buf.append("<br/>")
        elif tag == "hr":
            self.story.append(
                HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=4, spaceAfter=6)
            )
        elif tag == "img":
            self.image(a.get("src", ""), a.get("alt", ""))
        elif tag == "table":
            self.table = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.in_cell, self.cell_header, self.buf = True, tag == "th", []

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p"):
            if self.in_cell or self.block in ("li",):
                return
            if self.block == "quote":
                if tag == "p":
                    self.buf.append("<br/><br/>")
                return
            self.flush("h4" if tag in ("h5", "h6") else tag)
            self.block = None
        elif tag == "blockquote":
            txt = "".join(self.buf).strip()
            self.buf = [re.sub(r"(<br/>)+$", "", txt)]
            self.flush("quote")
            self.block = None
        elif tag == "li":
            self.flush("li")
            self.block = "li" if len(self.lists) > 1 else None
        elif tag in ("ul", "ol"):
            if self.block == "li" and self.buf:
                self.flush("li")
            self.lists.pop()
            self.block = "li" if self.lists else None
            if not self.lists:
                self.block = None
                self.story.append(Spacer(1, 4))
        elif tag == "pre":
            code = "".join(self.buf).rstrip("\n")
            self.buf, self.pre = [], False
            self.story.append(Preformatted(code, ST["code"]))
        elif tag == "code" and not self.pre:
            self.buf.append("</font>")
        elif tag in self.INLINE:
            self.buf.append(self.INLINE[tag][1])
        elif tag == "a":
            self.buf.append("</a>" if self.link_open else "</font>")
            self.link_open = False
        elif tag in ("td", "th"):
            self.row.append(("".join(self.buf).strip(), self.cell_header))
            self.buf, self.in_cell = [], False
        elif tag == "tr":
            self.table.append(self.row)
        elif tag == "table":
            self.emit_table()
            self.table = None

    def handle_data(self, data):
        if self.pre:
            self.buf.append(data)  # Preformatted takes raw text
        elif self.block is None and not self.in_cell and not data.strip():
            return
        else:
            self.buf.append(html.escape(data, quote=False))

    # blocks
    def image(self, src: str, alt: str) -> None:
        from PIL import Image as PILImage

        p = (self.base / src).resolve()
        if not p.is_file():
            raise PdfError(f"figure {src} not found (relative to {self.base.as_posix()})")
        with PILImage.open(p) as im:
            w, h = im.size
            dpi = (im.info.get("dpi") or (DEFAULT_DPI, DEFAULT_DPI))[0] or DEFAULT_DPI
        dw = min(self.width, w / float(dpi) * 72)
        dh = dw * h / w
        maxh = 125 * mm
        if dh > maxh:
            dw, dh = dw * maxh / dh, maxh
        img = Image(str(p), width=dw, height=dh)
        img.hAlign = "CENTER"
        self.story.append(KeepTogether([img, Paragraph(html.escape(alt), ST["cap"])]))

    def emit_table(self) -> None:
        rows = [r for r in self.table if r]
        if not rows:
            return
        ncols = max(map(len, rows))
        rows = [r + [("", False)] * (ncols - len(r)) for r in rows]
        avail = self.width
        for size in TABLE_SIZES:
            styles = {
                h: ParagraphStyle(
                    f"cell{size}{h}",
                    fontName="Sans-B" if h else "Sans",
                    fontSize=size,
                    leading=size * 1.25,
                    textColor=colors.white if h else INK,
                )
                for h in (False, True)
            }
            cells = [[Paragraph(t or "", styles[h]) for t, h in r] for r in rows]
            mins = [0.0] * ncols
            prefs = [0.0] * ncols
            for r in cells:
                for i, p in enumerate(r):
                    mins[i] = max(mins[i], p.minWidth() + 2 * CELL_PAD)
                    p.wrap(1e6, 1e6)
                    widths = p.getActualLineWidths0() or [0.0]
                    prefs[i] = max(prefs[i], max(widths) + 2 * CELL_PAD + 0.5)
            if sum(mins) <= avail:
                break
        else:
            head = " | ".join(re.sub("<[^>]+>", "", t) for t, _ in rows[0])
            raise PdfError(
                f"table [{head}] needs {sum(mins):.0f} pt at {TABLE_FLOOR_PT} pt text, the page "
                f"has {avail:.0f} pt: include fewer columns (cols:)"
            )
        if sum(prefs) <= avail:
            widths = prefs
        else:
            spare = avail - sum(mins)
            extra = [p - m for p, m in zip(prefs, mins, strict=True)]
            total = sum(extra) or 1.0
            widths = [m + spare * e / total for m, e in zip(mins, extra, strict=True)]
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), CELL_PAD),
            ("RIGHTPADDING", (0, 0), (-1, -1), CELL_PAD),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
        t = LongTable(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle(style))
        t._font_pt = size
        self.story += [t, Spacer(1, 8)]


class Doc(BaseDocTemplate):
    def __init__(self, path: str, title: str, author: str, footer: str):
        super().__init__(
            path,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=title,
            author=author,
            subject="Urban Traffic Twin",
            creator="traffic_twin.report",
        )
        self.footer = footer
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._page)])

    def _page(self, canv, doc) -> None:
        canv.saveState()
        canv.setFont("Sans", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(self.leftMargin, 10 * mm, self.footer)
        canv.drawRightString(A4[0] - self.rightMargin, 10 * mm, str(doc.page))
        canv.restoreState()

    def afterFlowable(self, f) -> None:
        toc = getattr(f, "_toc", None)
        if not toc or toc[0] > 3:
            return
        level = max(0, toc[0] - 2)
        key = f"h{self.seq}"
        self.seq += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(toc[1], key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, toc[1], self.page, key))

    def handle_documentBegin(self) -> None:
        self.seq = 0  # stable bookmark keys on every multiBuild pass
        super().handle_documentBegin()


def story(md_text: str, base: Path, width: float) -> list:
    html_text = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    b = Builder(base, width)
    b.feed(html_text)
    b.close()
    return b.story


def render_pdf(
    md_text: str,
    base: Path,
    out: Path,
    footer: str,
    *,
    title: str = "Urban Traffic Twin: final report",
    author: str = "Urban Traffic Twin",
) -> int:
    """Write the PDF of `md_text` (images resolved against `base`, the report's folder)
    to `out`; returns the page count. Invariant output (same input, same bytes)."""
    register_fonts()
    rl_config.invariant = 1
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Doc(str(out), title=title, author=author, footer=footer)
    flow = story(md_text, base, doc.width)
    doc.multiBuild(flow)
    return doc.page


def table_font_sizes(md_text: str, base: Path) -> list[float]:
    """The text size of every table of `md_text` (tests: each >= 6.5 pt)."""
    register_fonts()
    width = A4[0] - 36 * mm
    return [f._font_pt for f in story(md_text, base, width) if hasattr(f, "_font_pt")]
