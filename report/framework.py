"""Small document layer on top of python-docx for the NV62010 report.

Content is collected as blocks first, then numbered and rendered, so that forward
references work and the numbering rules can be checked automatically.

Inline markup in text:
    [[c:key1,key2]]  IEEE citation(s), numbered in order of first citation
    [[F:key]]  [[T:key]]  [[E:key]]  references to a figure, table or equation
    $latex$  inline equation (real Word equation)
    **bold**  *italic*
    <<name>>  a value from results/ (see values.py); unknown names stop the build
"""
import re
from copy import deepcopy
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from latex2mathml.converter import convert as latex_to_mathml
from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
FONT = "Times New Roman"
LINE = 1.158   # line spacing of the student's ML lab file
TOKEN = re.compile(r"(\[\[[cFTE]:[^\]]+\]\]|\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)")
_transform = etree.XSLT(etree.parse(str(XSL)))


M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def omath(latex):
    root = etree.fromstring(latex_to_mathml(latex))
    math = deepcopy(etree.fromstring(etree.tostring(_transform(root))))
    m = lambda tag: f"{{{M_NS}}}{tag}"
    # matrices: use real, stretching brackets rather than plain "(" and ")" characters
    for matrix in list(math.iter(m("m"))):
        before, after = matrix.getprevious(), matrix.getnext()
        if before is None or after is None or before.tag != m("r") or after.tag != m("r"):
            continue
        if "".join(before.itertext()).strip() != "(" or "".join(after.itertext()).strip() != ")":
            continue
        parent = matrix.getparent()
        delimiter = etree.Element(m("d"))
        element = etree.SubElement(delimiter, m("e"))
        parent.insert(parent.index(before), delimiter)
        parent.remove(before)
        parent.remove(after)
        element.append(matrix)
    # numbers and brackets stay upright, as in normal mathematical typesetting
    for run in math.iter(m("r")):
        text = "".join(t.text or "" for t in run.iter(m("t")))
        if text and re.fullmatch(r"[0-9.,+\-=()\[\]\s−]+", text) and run.find(m("rPr")) is None:
            props = etree.Element(m("rPr"))
            etree.SubElement(props, m("sty")).set(m("val"), "p")
            run.insert(0, props)
    return math


def set_column_widths(table, widths):
    """Set both the grid and the cells, because WPS Office reads the grid."""
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
        for cell in column.cells:
            cell.width = Inches(width)


def set_run_font(run, name=FONT, size=None, bold=None, italic=None):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def shade(element_pr, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    element_pr.append(shd)


def add_field(paragraph, instruction, placeholder="Update this field", size=None, font=FONT):
    def char(kind):
        run = paragraph.add_run()
        set_run_font(run, name=font, size=size)
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
        run._r.append(el)
        return el
    char("begin").set(qn("w:dirty"), "true")
    run = paragraph.add_run()
    set_run_font(run, name=font, size=size)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    run._r.append(instr)
    char("separate")
    set_run_font(paragraph.add_run(placeholder), name=font, size=size)
    char("end")


class Report:
    def __init__(self, values, references):
        self.values = values
        self.references = references      # key -> IEEE text
        self.blocks = []
        self.problems = []

    # ------------------------------------------------------------------ content API
    def _add(self, kind, **kw):
        self.blocks.append({"kind": kind, **kw})

    def h1(self, text, numbered=True): self._add("h1", text=text, numbered=numbered)
    def h2(self, text): self._add("h2", text=text)
    def h3(self, text): self._add("h3", text=text)
    def p(self, text, style=None): self._add("p", text=text, style=style)
    def review(self, text): self._add("p", text=text, style="Review")
    def reference(self, text): self._add("p", text=text, style="Reference")
    def bullets(self, items): self._add("bullets", items=items)
    def numbered(self, items): self._add("numbered", items=items)
    def code(self, text, caption=None): self._add("code", text=text, caption=caption)
    def page_break(self): self._add("page_break")
    def centered(self, text, size=12, bold=False, after=6): self._add("centered", text=text, size=size, bold=bold, after=after)
    def toc(self, instruction): self._add("toc", instruction=instruction)
    def begin_body(self): self._add("begin_body")
    def begin_landscape(self): self._add("orientation", landscape=True)
    def end_landscape(self): self._add("orientation", landscape=False)
    def references_here(self): self._add("references")

    def fig(self, key, files, caption, width=5.5):
        files = [files] if isinstance(files, str) else files
        self._add("fig", key=key, files=files, caption=caption, width=width)

    def table(self, key, caption, header, rows, align=None, widths=None, size=10):
        self._add("table", key=key, caption=caption, header=header, rows=rows, align=align,
                  widths=widths, size=size)

    def eq(self, key, latex): self._add("eq", key=key, latex=latex)

    # ------------------------------------------------------------------ numbering pass
    def _fill(self, text):
        def sub(m):
            name = m.group(1)
            if name not in self.values:
                self.problems.append(f"unknown value <<{name}>>")
                return "??"
            return str(self.values[name])
        return re.sub(r"<<([^<>]+)>>", sub, text)

    def _block_texts(self, block):
        k = block["kind"]
        if k in ("p", "h1", "h2", "h3"):
            yield block["text"]
        elif k in ("bullets", "numbered"):
            yield from block["items"]
        elif k == "table":
            for row in block["rows"]:
                yield from (str(c) for c in row)
        elif k == "references":
            return

    def number(self):
        self.fig_no, self.tab_no, self.eq_no, self.cite_no = {}, {}, {}, {}
        section, nf, nt, ne = 0, 0, 0, 0
        h2, h3 = 0, 0
        seen_refs = set()
        for block in self.blocks:
            k = block["kind"]
            if k == "h1":
                if block["numbered"]:
                    section += 1
                    block["label"] = f"{section}."
                nf = nt = h2 = h3 = 0
            elif k == "h2":
                h2 += 1
                h3 = 0
                block["label"] = f"{section}.{h2}"
            elif k == "h3":
                h3 += 1
                block["label"] = f"{section}.{h2}.{h3}"
            for text in self._block_texts(block):
                text = self._fill(text)
                for token in re.findall(r"\[\[([cFTE]):([^\]]+)\]\]", text):
                    if token[0] == "c":
                        for key in token[1].split(","):
                            key = key.strip()
                            if key not in self.references:
                                self.problems.append(f"citation to unknown reference {key}")
                            self.cite_no.setdefault(key, len(self.cite_no) + 1)
                    else:
                        seen_refs.add(f"{token[0]}:{token[1]}")
            if k == "fig":
                nf += 1
                self.fig_no[block["key"]] = f"Fig. {section}.{nf}"
                if f"F:{block['key']}" not in seen_refs:
                    self.problems.append(f"figure {block['key']} not referred to before it appears")
            elif k == "table":
                nt += 1
                self.tab_no[block["key"]] = f"TABLE {section}.{nt}"
                if f"T:{block['key']}" not in seen_refs:
                    self.problems.append(f"table {block['key']} not referred to before it appears")
            elif k == "eq":
                ne += 1
                self.eq_no[block["key"]] = f"({ne})"
        for key in self.references:
            if key not in self.cite_no:
                self.problems.append(f"reference {key} is never cited")
        for ref in seen_refs:
            kind, key = ref.split(":")
            store = {"F": self.fig_no, "T": self.tab_no, "E": self.eq_no}[kind]
            if key not in store:
                self.problems.append(f"reference to missing {ref}")

    def _citation_text(self, keys):
        numbers = sorted(self.cite_no[k.strip()] for k in keys.split(","))
        groups, start, prev = [], numbers[0], numbers[0]
        for n in numbers[1:] + [None]:
            if n is not None and n == prev + 1:
                prev = n
                continue
            if prev - start >= 2:
                groups.append(f"[{start}]\u2013[{prev}]")
            else:
                groups.extend(f"[{x}]" for x in range(start, prev + 1))
            start = prev = n
        return ", ".join(groups)

    # ------------------------------------------------------------------ rendering
    def _inline(self, paragraph, text, size=None, bold=None):
        text = self._fill(text)
        for part in TOKEN.split(text):
            if not part:
                continue
            if part.startswith("[["):
                kind, key = part[2:-2].split(":", 1)
                if kind == "c":
                    shown = self._citation_text(key)
                else:
                    table = {"F": self.fig_no, "T": self.tab_no, "E": self.eq_no}[kind]
                    shown = table.get(key, "??")
                    if kind == "E":
                        shown = f"Eq. {shown}"
                    elif kind == "T":
                        shown = shown.replace("TABLE", "Table")
                set_run_font(paragraph.add_run(shown), size=size, bold=bold)
            elif part.startswith("$"):
                paragraph._p.append(omath(part[1:-1]))
            elif part.startswith("**"):
                self._inline(paragraph, part[2:-2], size=size, bold=True)
            elif part.startswith("*"):
                set_run_font(paragraph.add_run(part[1:-1]), size=size, bold=bold, italic=True)
            else:
                set_run_font(paragraph.add_run(part), size=size, bold=bold)

    def _styles(self, doc):
        def base(style, size, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.5, before=0, after=6):
            style.font.name = FONT
            style.font.size = Pt(size)
            style.font.bold = bold
            style.font.italic = False
            style.font.color.rgb = RGBColor(0, 0, 0)
            rpr = style.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                fonts.set(qn(attr), FONT)
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if fonts.get(qn(attr)) is not None:
                    del fonts.attrib[qn(attr)]
            pf = style.paragraph_format
            pf.alignment = align
            pf.line_spacing = line
            pf.space_before = Pt(before)
            pf.space_after = Pt(after)
        styles = doc.styles
        # Body and headings follow the student's ML lab file: Normal 12 pt with 1.16 line
        # spacing and 8 pt after; blue headings at 20, 16 and 14 pt.
        base(styles["Normal"], 12, line=LINE, after=8)
        for name, size, before, after in (("Heading 1", 20, 0, 10), ("Heading 2", 16, 12, 4), ("Heading 3", 14, 8, 4)):
            base(styles[name], size, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.08, before=before, after=after)
            styles[name].font.color.rgb = RGBColor(0x2F, 0x55, 0x97)
            styles[name].paragraph_format.keep_with_next = True
        styles["Heading 1"].paragraph_format.page_break_before = True
        for name in ("List Bullet", "List Number"):
            base(styles[name], 12, line=LINE, after=4)
        from docx.enum.style import WD_STYLE_TYPE
        def new(name, size, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.5, before=0, after=6, bold=False):
            style = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style.base_style = styles["Normal"]
            base(style, size, bold=bold, align=align, line=line, before=before, after=after)
            return style
        new("Fig Caption", 11, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, before=2, after=10)
        tc = new("Table Caption", 11, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, before=8, after=4)
        tc.paragraph_format.keep_with_next = True
        new("Review", 12, line=LINE, after=8)
        ref = new("Reference", 11, line=1.0, after=4)
        ref.paragraph_format.left_indent = Cm(0.9)
        ref.paragraph_format.first_line_indent = Cm(-0.9)
        code = new("Code Block", 9.5, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.0, before=0, after=0)
        code.font.name = "Consolas"
        fonts = code.element.get_or_add_rPr().find(qn("w:rFonts"))
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            fonts.set(qn(attr), "Consolas")

    def _page_setup(self, section, landscape=False):
        a4_w, a4_h = Cm(21.0), Cm(29.7)
        section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
        section.page_width, section.page_height = (a4_h, a4_w) if landscape else (a4_w, a4_h)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Inches(0.9))
        section.header_distance = Cm(1.25)
        section.footer_distance = Cm(1.25)
        # page border as in the student's ML lab file
        sect = section._sectPr
        for el in sect.findall(qn("w:pgBorders")):
            sect.remove(el)
        borders = OxmlElement("w:pgBorders")
        borders.set(qn("w:offsetFrom"), "page")
        for side in ("top", "left", "bottom", "right"):
            edge = OxmlElement(f"w:{side}")
            for key, value in (("val", "single"), ("sz", "18"), ("space", "24"), ("color", "000000")):
                edge.set(qn(f"w:{key}"), value)
            borders.append(edge)
        sect.find(qn("w:pgMar")).addnext(borders)

    def _page_number(self, section, fmt=None, start=None):
        sect = section._sectPr
        for el in sect.findall(qn("w:pgNumType")):
            sect.remove(el)
        for el in sect.findall(qn("w:titlePg")):
            sect.remove(el)
        if fmt:
            pg = OxmlElement("w:pgNumType")
            pg.set(qn("w:fmt"), fmt)
            if start is not None:
                pg.set(qn("w:start"), str(start))
            sect.find(qn("w:pgBorders")).addnext(pg)

    def _new_section(self, doc, landscape=False, fmt="decimal", start=None):
        section = doc.add_section(WD_SECTION.NEW_PAGE)
        self._page_setup(section, landscape)
        self._page_number(section, fmt, start)
        return section

    def _header_footer(self, section, short_title, course):
        head = section.header.paragraphs[0]
        head.alignment = WD_ALIGN_PARAGRAPH.CENTER
        head.paragraph_format.line_spacing = 1.0
        set_run_font(head.add_run(short_title), size=14)
        foot = section.footer.paragraphs[0]
        foot.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        foot.paragraph_format.line_spacing = 1.0
        add_field(foot, "PAGE", "1", size=11, font="Arial")

    def _cover(self, doc, block):
        """Cover page laid out like the student's ML lab file."""
        def borderless(rows, cols):
            table = doc.add_table(rows=rows, cols=cols)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = False
            return table
        def line(para, text, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=6, bold=None, color=None):
            para.alignment = align
            para.paragraph_format.line_spacing = 1.0
            para.paragraph_format.space_after = Pt(after)
            run = para.add_run(text)
            set_run_font(run, size=size, bold=bold)
            if color:
                run.font.color.rgb = RGBColor.from_string(color)
        assets = Path(__file__).resolve().parent / "assets"
        logos = borderless(1, 2)
        set_column_widths(logos, (3.61, 2.85))
        for cell, name, height_note in ((logos.cell(0, 0), "sharda_logo.png", 2.9), (logos.cell(0, 1), "naac_badge.png", 1.05)):
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.line_spacing = 1.0
            para.add_run().add_picture(str(assets / name), width=Inches(height_note))
        line(doc.add_paragraph(), "", 12, after=14)
        line(doc.add_paragraph(), block["school"], 12, after=14, bold=True)
        line(doc.add_paragraph(), block["department"], 18, after=34, color="1F4E79")
        box = doc.add_table(rows=1, cols=1)
        box.style = "Table Grid"
        box.alignment = WD_TABLE_ALIGNMENT.CENTER
        box.autofit = False
        set_column_widths(box, (5.8,))
        cell = box.cell(0, 0)
        line(cell.paragraphs[0], block["title"], 18, after=10, bold=True)
        line(cell.add_paragraph(), block["course"], 14, after=8)
        line(doc.add_paragraph(), "", 12, after=24)
        for text, after in block["lines"]:
            line(doc.add_paragraph(), text, 13, after=after)
        people = borderless(1, 2)
        set_column_widths(people, (3.61, 2.85))
        for cell, heading, rows in ((people.cell(0, 0), "Submitted By:", block["by"]), (people.cell(0, 1), "Submitted To:", block["to"])):
            line(cell.paragraphs[0], heading, 12, WD_ALIGN_PARAGRAPH.LEFT, bold=True)
            for text in rows:
                line(cell.add_paragraph(), text, 12, WD_ALIGN_PARAGRAPH.LEFT)

    def _figure(self, doc, block):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.line_spacing = 1.0
        para.paragraph_format.keep_with_next = True
        para.paragraph_format.space_after = Pt(2)
        widths = block["width"] if isinstance(block["width"], list) else [block["width"]] * len(block["files"])
        for i, name in enumerate(block["files"]):
            path = FIG_DIR / f"{name}.png"
            if not path.exists():
                self.problems.append(f"missing figure file {name}")
                continue
            image = Image.open(path)
            w, h = image.size
            width = min(widths[i], 6.0, 8.2 * w / h)
            # embed at 300 dpi for the printed size (source files can be far larger)
            target = int(width * 300)
            buffer = BytesIO()
            if w > target:
                image = image.convert("RGB").resize((target, round(h * target / w)), Image.LANCZOS)
            image.save(buffer, format="PNG", dpi=(300, 300), optimize=True)
            buffer.seek(0)
            run = para.add_run()
            run.add_picture(buffer, width=Inches(width))
            if i < len(block["files"]) - 1:
                para.add_run(" ")
        cap = doc.add_paragraph(style="Fig Caption")
        self._inline(cap, f"{self.fig_no[block['key']]}. {block['caption']}")

    def _table(self, doc, block):
        cap = doc.add_paragraph(style="Table Caption")
        self._inline(cap, f"{self.tab_no[block['key']]}. {block['caption']}")
        header, rows = block["header"], block["rows"]
        cols = len(header)
        table = doc.add_table(rows=1 + len(rows), cols=cols)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        align = block["align"] or "l" * cols
        if block["widths"]:
            set_column_widths(table, block["widths"])
        keep_together = len(rows) <= 20       # short tables stay on one page
        lookup = {"l": WD_ALIGN_PARAGRAPH.LEFT, "r": WD_ALIGN_PARAGRAPH.RIGHT, "c": WD_ALIGN_PARAGRAPH.CENTER}
        for r, row in enumerate([header] + rows):
            tr = table.rows[r]._tr
            trpr = tr.get_or_add_trPr()
            trpr.append(OxmlElement("w:cantSplit"))
            if r == 0:
                trpr.append(OxmlElement("w:tblHeader"))
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                if block["widths"]:
                    cell.width = Inches(block["widths"][c])
                para = cell.paragraphs[0]
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER if r == 0 else lookup[align[c]]
                para.paragraph_format.keep_with_next = keep_together and r < len(rows)
                para.paragraph_format.line_spacing = 1.0
                para.paragraph_format.space_after = Pt(1)
                para.paragraph_format.space_before = Pt(1)
                self._inline(para, str(value), size=block["size"], bold=True if r == 0 else None)
                if r == 0:
                    shade(cell._tc.get_or_add_tcPr(), "D9E2F3")
        after = doc.add_paragraph()
        after.paragraph_format.space_after = Pt(4)
        after.paragraph_format.line_spacing = 1.0

    def _equation(self, doc, block):
        table = doc.add_table(rows=1, cols=3)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        set_column_widths(table, (0.25, 5.65, 0.55))
        middle, right = table.cell(0, 1).paragraphs[0], table.cell(0, 2).paragraphs[0]
        middle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        middle.paragraph_format.line_spacing = 1.0
        wrapper = OxmlElement("m:oMathPara")
        wrapper.append(omath(block["latex"]))
        middle._p.append(wrapper)
        right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        right.paragraph_format.line_spacing = 1.0
        set_run_font(right.add_run(self.eq_no[block["key"]]))
        from docx.enum.table import WD_ALIGN_VERTICAL
        for cell in table.rows[0].cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        after = doc.add_paragraph()
        after.paragraph_format.space_after = Pt(2)
        after.paragraph_format.line_spacing = 1.0

    def _code(self, doc, block):
        lines = block["text"].strip("\n").split("\n")
        if len(lines) > 20:
            self.problems.append(f"code excerpt has {len(lines)} lines (limit 20): {lines[0][:40]}")
        for i, line in enumerate(lines):
            para = doc.add_paragraph(style="Code Block")
            para.paragraph_format.keep_together = True
            para.paragraph_format.keep_with_next = i < len(lines) - 1
            shade(para._p.get_or_add_pPr(), "F2F2F2")
            set_run_font(para.add_run(line if line else " "), name="Consolas", size=9.5)
        if block["caption"]:
            cap = doc.add_paragraph(style="Fig Caption")
            self._inline(cap, block["caption"])
        else:
            doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def render(self, path, short_title, course):
        self.number()
        doc = Document()
        self._styles(doc)
        first = doc.sections[0]
        self._page_setup(first)
        self._page_number(first, "lowerRoman", 1)
        first._sectPr.find(qn("w:cols")).addnext(OxmlElement("w:titlePg"))
        self._header_footer(first, short_title, course)
        for block in self.blocks:
            k = block["kind"]
            if k == "h1":
                label = f"{block['label']} " if block.get("label") else ""
                para = doc.add_paragraph(style="Heading 1")
                self._inline(para, label + block["text"], bold=True, size=20)
            elif k in ("h2", "h3"):
                level, size = (2, 16) if k == "h2" else (3, 14)
                para = doc.add_paragraph(style=f"Heading {level}")
                self._inline(para, f"{block['label']} {block['text']}", bold=True, size=size)
            elif k == "p":
                para = doc.add_paragraph(style=block["style"] or "Normal")
                self._inline(para, block["text"])
            elif k == "bullets":
                for item in block["items"]:
                    self._inline(doc.add_paragraph(style="List Bullet"), item)
            elif k == "numbered":
                for n, item in enumerate(block["items"], start=1):
                    para = doc.add_paragraph()
                    para.paragraph_format.left_indent = Cm(0.9)
                    para.paragraph_format.first_line_indent = Cm(-0.7)
                    para.paragraph_format.space_after = Pt(3)
                    self._inline(para, f"{n}.  {item}")
            elif k == "centered":
                para = doc.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para.paragraph_format.line_spacing = 1.15
                para.paragraph_format.space_after = Pt(block["after"])
                self._inline(para, block["text"], size=block["size"], bold=block["bold"] or None)
            elif k == "page_break":
                para = doc.add_paragraph()
                para.paragraph_format.page_break_before = True
                para.paragraph_format.line_spacing = 1.0
                para.paragraph_format.space_after = Pt(0)
            elif k == "toc":
                para = doc.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                add_field(para, block["instruction"])
            elif k == "begin_body":
                self._new_section(doc, fmt="decimal", start=1)
            elif k == "orientation":
                self._new_section(doc, landscape=block["landscape"], fmt="decimal")
            elif k == "fig":
                self._figure(doc, block)
            elif k == "table":
                self._table(doc, block)
            elif k == "eq":
                self._equation(doc, block)
            elif k == "code":
                self._code(doc, block)
            elif k == "cover":
                self._cover(doc, block)
            elif k == "abbrev":
                table = doc.add_table(rows=1 + len(block["rows"]), cols=2)
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.autofit = False
                set_column_widths(table, (1.4, 4.8))
                for r, row in enumerate([["Abbreviation", "Meaning"]] + block["rows"]):
                    for c, value in enumerate(row):
                        cell = table.cell(r, c)
                        para = cell.paragraphs[0]
                        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        para.paragraph_format.line_spacing = 1.0
                        para.paragraph_format.space_after = Pt(0)
                        set_run_font(para.add_run(value), size=10.5, bold=True if r == 0 else None)
                        if r == 0:
                            shade(cell._tc.get_or_add_tcPr(), "D9E2F3")
            elif k == "references":
                for key, number in sorted(self.cite_no.items(), key=lambda kv: kv[1]):
                    para = doc.add_paragraph(style="Reference")
                    self._inline(para, f"[{number}]\u00a0{self.references[key]}")
        # mark the file as a current Word document (otherwise it opens in Compatibility Mode)
        settings = doc.settings.element
        compat = settings.find(qn("w:compat"))
        if compat is None:
            compat = OxmlElement("w:compat")
            settings.append(compat)
        for old in compat.findall(qn("w:compatSetting")):
            if old.get(qn("w:name")) == "compatibilityMode":
                compat.remove(old)
        mode = OxmlElement("w:compatSetting")
        mode.set(qn("w:name"), "compatibilityMode")
        mode.set(qn("w:uri"), "http://schemas.microsoft.com/office/word")
        mode.set(qn("w:val"), "15")
        compat.append(mode)
        doc.core_properties.author = ""
        doc.core_properties.title = short_title
        doc.save(path)
        return self.problems
