"""
Generate a detailed technical report PDF on the 10-K Itemization Task.
Written so a 10-year-old can follow the thinking and methods.
"""

from fpdf import FPDF
import os

class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        # Add DejaVu for Unicode support
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        os.makedirs(font_dir, exist_ok=True)
        # Use built-in fonts but avoid Unicode chars outside latin-1
        self._use_builtin = True

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "10-K Itemization: Technical Report", align="C")
            self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(20, 60, 120)
            self.ln(6)
            self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(20, 60, 120)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(6)
        elif level == 2:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(40, 90, 160)
            self.ln(4)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(60, 60, 60)
            self.ln(2)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.l_margin
        self.set_x(x + indent)
        self.cell(5, 5.5, "-", new_x="END")
        w = self.w - self.l_margin - self.r_margin - indent - 5
        self.multi_cell(w, 5.5, f" {text}")
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        self.set_x(x + 5)
        for line in text.split("\n"):
            self.cell(self.w - self.l_margin - self.r_margin - 10, 5, f"  {line}",
                      fill=True, new_x="LMARGIN", new_y="NEXT")
            self.set_x(x + 5)
        self.ln(3)

    def analogy_box(self, text):
        self.set_fill_color(255, 248, 220)
        self.set_draw_color(200, 170, 80)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(100, 70, 0)
        x = self.get_x()
        y = self.get_y()
        self.set_x(x + 5)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 10, 5.5,
                        f"  Analogy: {text}", fill=True, border=1)
        self.ln(4)

    def result_box(self, text):
        self.set_fill_color(220, 245, 220)
        self.set_draw_color(60, 150, 60)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(20, 80, 20)
        x = self.get_x()
        self.set_x(x + 5)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 10, 5.5,
                        f"  {text}", fill=True, border=1)
        self.ln(4)

    def warning_box(self, text):
        self.set_fill_color(255, 235, 235)
        self.set_draw_color(200, 60, 60)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(150, 30, 30)
        x = self.get_x()
        self.set_x(x + 5)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 10, 5.5,
                        f"  {text}", fill=True, border=1)
        self.ln(4)

    def simple_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - self.l_margin - self.r_margin) / len(headers)
            col_widths = [w] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(4)


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, "10-K Item Extraction", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 12, "Technical Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(20, 60, 120)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "Extracting Sections from SEC 10-K Filings", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Using Anchor-Based HTML Parsing", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Improving F1 Score from 90% to 97%", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Without Introducing Bias", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "March 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    toc = [
        ("1", "What Is This Project About?", "The Big Picture"),
        ("2", "What Is a 10-K Filing?", "Understanding the Data"),
        ("3", "The Pipeline: How It Works", "Step-by-Step Extraction"),
        ("4", "How We Measure Success", "The F1 Score"),
        ("5", "What Was Going Wrong", "Finding the Problems"),
        ("6", "The Fix: HTML Entity Decoding", "The Biggest Improvement"),
        ("7", "The Fix: Item 16 Pass-Through", "Removing Bad Assumptions"),
        ("8", "Bias Analysis", "Did We Cheat?"),
        ("9", "Results: Before and After", "The Numbers"),
        ("10", "Remaining Challenges", "What's Still Hard"),
        ("11", "Lessons Learned", "What We Can Take Away"),
    ]
    pdf.set_font("Helvetica", "", 11)
    for num, title, subtitle in toc:
        pdf.set_text_color(20, 60, 120)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(12, 7, num)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(90, 7, title)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 7, subtitle, new_x="LMARGIN", new_y="NEXT")

    # =========================================================================
    # SECTION 1: What Is This Project About?
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("1. What Is This Project About?")

    pdf.analogy_box(
        "Imagine you have a really long book with 20 chapters, but the chapters "
        "aren't numbered clearly. Your job is to figure out where each chapter "
        "starts and ends, and then cut the book into separate pieces -- one piece "
        "per chapter. That's basically what this project does, but with financial "
        "documents instead of storybooks!"
    )

    pdf.body_text(
        "Every year, big companies in the United States have to file a document "
        "called a \"10-K\" with a government agency called the SEC (Securities and "
        "Exchange Commission). This document tells investors everything important "
        "about the company -- how much money they made, what risks they face, "
        "who runs the company, and much more."
    )

    pdf.body_text(
        "A 10-K filing is divided into specific sections (called \"Items\"), each "
        "covering a different topic. There are about 20 standard items:"
    )

    items = [
        ("Item 1", "Business - What the company does"),
        ("Item 1A", "Risk Factors - What could go wrong"),
        ("Item 1B", "Unresolved Staff Comments"),
        ("Item 2", "Properties - Buildings and land the company owns"),
        ("Item 3", "Legal Proceedings - Lawsuits"),
        ("Item 4", "Mine Safety Disclosures"),
        ("Items 5-9", "Financial information, management discussion"),
        ("Items 10-14", "Directors, compensation, governance"),
        ("Item 15", "Exhibits and financial statement schedules"),
        ("Item 16", "Form 10-K Summary (often just says 'None')"),
        ("Signatures", "Who signed the document"),
    ]
    for item, desc in items:
        pdf.bullet(f"{item}: {desc}")

    pdf.body_text(
        "Our goal: given a raw 10-K filing (which is one giant HTML file), "
        "automatically find where each item starts and ends, and extract the "
        "HTML content for each item. We then compare our extracted items against "
        "a \"ground truth\" (the correct answers) to see how well we did."
    )

    pdf.chapter_title("The Challenge", level=2)
    pdf.body_text(
        "This sounds simple, but it's tricky because:"
    )
    pdf.bullet("Each company formats their 10-K differently -- no two look the same")
    pdf.bullet("Some items are hundreds of pages long, others are just one word (\"None\")")
    pdf.bullet("The files are messy HTML with thousands of tags, styles, and formatting")
    pdf.bullet("Some companies combine multiple items into one section")
    pdf.bullet("We have 500+ files across 3 datasets to process correctly")

    # =========================================================================
    # SECTION 2: What Is a 10-K Filing?
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("2. What Is a 10-K Filing?")

    pdf.analogy_box(
        "Think of a 10-K like a school report card, but for a company. "
        "Instead of grades in Math and English, it has sections about Money, "
        "Risks, Properties, and Lawsuits. And instead of being 1 page, it can "
        "be hundreds or even thousands of pages long!"
    )

    pdf.chapter_title("The Raw File Format", level=2)
    pdf.body_text(
        "The SEC stores 10-K filings as .txt files, but don't let the extension "
        "fool you -- inside, they contain HTML (the same language used to build "
        "websites). The file has a special wrapper structure:"
    )

    pdf.code_block(
        "<DOCUMENT>\n"
        "  <TYPE>10-K\n"
        "  <TEXT>\n"
        "    <html>\n"
        "      ... hundreds of thousands of lines of HTML ...\n"
        "    </html>\n"
        "  </TEXT>\n"
        "</DOCUMENT>"
    )

    pdf.body_text(
        "Our pipeline first strips away the SEC wrapper to get to the actual "
        "HTML document inside."
    )

    pdf.chapter_title("How Items Are Marked", level=2)
    pdf.body_text(
        "Inside the HTML, items are marked using \"anchor\" elements -- special "
        "HTML tags that act like bookmarks. The Table of Contents at the top of "
        "the document has clickable links that point to these anchors:"
    )

    pdf.code_block(
        "Table of Contents:   <a href=\"#item1a\">Item 1A. Risk Factors</a>\n"
        "                          |\n"
        "                          v  (points to)\n"
        "Later in document:   <div id=\"item1a\">Item 1A. RISK FACTORS ..."
    )

    pdf.body_text(
        "The key insight that makes our whole approach work: the ground truth "
        "(correct answers) slices the HTML from one anchor to the next. So if "
        "we can find the right anchors, we can find the right content."
    )

    pdf.chapter_title("Data Structure", level=2)
    pdf.body_text("Our project has 3 datasets, each with 167 files:")

    pdf.simple_table(
        ["Dataset", "Input Files", "Ground Truth", "Predictions"],
        [
            ["Set 1", "data/folder_1/", "data/ground_truth_1/", "data/predictions_1/"],
            ["Set 2", "data/folder_2/", "data/ground_truth_2/", "data/predictions_2/"],
            ["Set 3", "data/folder_3/", "data/ground_truth_3/", "data/predictions_3/"],
        ],
        [25, 50, 55, 50]
    )

    pdf.body_text(
        "Not all 167 files have ground truth answers (some are empty). "
        "After excluding empty ground truth files, we evaluate on roughly "
        "130 files per set, for about 390 files total."
    )

    # =========================================================================
    # SECTION 3: The Pipeline
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("3. The Pipeline: How It Works")

    pdf.analogy_box(
        "Our extraction pipeline is like a treasure map reader. Step 1: Find "
        "the treasure map (Table of Contents). Step 2: Read all the 'X marks "
        "the spot' locations (anchor IDs). Step 3: Walk to each X and figure "
        "out which treasure (item) is buried there. Step 4: Dig up the treasure "
        "(extract the HTML between consecutive X marks)."
    )

    pdf.chapter_title("Step 1: Extract the 10-K Document", level=2)
    pdf.body_text(
        "The SEC submission file may contain multiple documents (the 10-K itself, "
        "plus exhibits, XBRL data, etc.). We search for the <DOCUMENT> block with "
        "<TYPE>10-K and extract just the HTML inside its <TEXT> section."
    )

    pdf.chapter_title("Step 2: Collect TOC-Referenced Anchor IDs", level=2)
    pdf.body_text(
        "We scan the HTML for all links that look like <a href=\"#SOMETHING\">. "
        "The \"SOMETHING\" is an anchor ID. We collect all these IDs into a set. "
        "This is important: we ONLY consider anchors that are linked from somewhere "
        "(usually the Table of Contents). This filters out thousands of irrelevant "
        "anchors used for footnotes, cross-references, etc."
    )

    pdf.chapter_title("Step 3: Find Anchor Elements", level=2)
    pdf.body_text(
        "Now we search the entire document for HTML elements that have these IDs. "
        "Any tag can be an anchor: <a id=\"...\">, <div id=\"...\">, <p id=\"...\">, "
        "<span id=\"...\">, etc. For each matching element, we record its character "
        "position in the document. When we find duplicates (same ID appears twice), "
        "we keep the LAST one, because the first is usually in the Table of Contents "
        "header and the last is the actual section start."
    )

    pdf.chapter_title("Step 4: Classify Anchors (The Brain)", level=2)
    pdf.body_text(
        "This is the most complex step. For each anchor, we need to figure out "
        "which 10-K item it belongs to. We use a tiered classification system:"
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Tier 0 (Highest Confidence): Anchor ID Matching", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "Many anchor IDs directly contain the item name, like \"ITEM1ARISKFACTORS\" "
        "or \"item_7a\". We use regex patterns to match these."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Tier 1 (High Confidence): 'Item X' Pattern in Nearby Text", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "We look at the text near the anchor for patterns like \"Item 1A\" or "
        "\"ITEM 7\". This works for most filings that follow standard formatting."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Tier 2 (Lower Confidence): Descriptive Title Matching", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "For filings that don't use \"Item X\" labels, we look for descriptive "
        "titles like \"Risk Factors\", \"Management's Discussion and Analysis\", "
        "\"Executive Compensation\", etc."
    )

    pdf.chapter_title("Step 4b: Sequence-Constrained Selection (DP)", level=2)
    pdf.body_text(
        "Items in a 10-K must appear in a specific order (Item 1, then 1A, then "
        "1B, then 2, then 3, ...). After classifying all anchors, we use dynamic "
        "programming to find the best set of anchors that respects this ordering. "
        "This is modeled as a weighted longest increasing subsequence problem, "
        "where the weight combines classification confidence and document position."
    )

    pdf.analogy_box(
        "Imagine you're reading a book and you find 50 sticky notes that say "
        "things like 'Chapter 3' or 'Chapter 7'. Some sticky notes might be wrong "
        "(placed in chapter 5 but labeled 'Chapter 3'). The DP algorithm finds "
        "the best set of sticky notes where the chapter numbers go in order and "
        "the most confident labels are preferred."
    )

    pdf.chapter_title("Step 5: Slice the HTML", level=2)
    pdf.body_text(
        "Finally, we cut the HTML between consecutive anchor positions. "
        "Item 1's content goes from Item 1's anchor to Item 1A's anchor. "
        "Item 1A's content goes from Item 1A's anchor to Item 1B's anchor. "
        "And so on. The last item extends to the end of the document body."
    )

    # =========================================================================
    # SECTION 4: How We Measure Success
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("4. How We Measure Success: The F1 Score")

    pdf.analogy_box(
        "Imagine you and your friend both have bags of colored marbles. Your "
        "friend's bag is the 'correct answer.' The F1 score measures how similar "
        "your bag is to your friend's bag. If you have the exact same marbles, "
        "F1 = 100%. If you have completely different marbles, F1 = 0%."
    )

    pdf.chapter_title("Character-Level Bag-of-Characters F1", level=2)
    pdf.body_text(
        "Our metric works like this:"
    )
    pdf.bullet("Take the predicted HTML for an item and the ground truth HTML")
    pdf.bullet("Strip all HTML tags from both (leaving just the visible text)")
    pdf.bullet("Count how many of each character appears in each text")
    pdf.bullet("Calculate overlap: how many characters are shared?")

    pdf.body_text("The formula:")

    pdf.code_block(
        "Precision = (shared characters) / (characters in prediction)\n"
        "Recall    = (shared characters) / (characters in ground truth)\n"
        "F1        = 2 * Precision * Recall / (Precision + Recall)"
    )

    pdf.body_text(
        "Precision asks: \"Of everything we predicted, how much was correct?\" "
        "Recall asks: \"Of everything in the correct answer, how much did we find?\" "
        "F1 balances both -- you need to be both precise AND complete to score well."
    )

    pdf.chapter_title("Why This Metric Works", level=2)
    pdf.body_text(
        "This character-level F1 is forgiving of small boundary differences. "
        "If our extraction includes 5 extra characters at the end, it barely "
        "affects the score for a 100,000-character item. But if we completely "
        "miss an item (predicting empty when the answer has content), the score "
        "drops to 0 for that item."
    )

    pdf.chapter_title("Aggregation", level=2)
    pdf.body_text(
        "The overall F1 is computed as: for each file, average the F1 across "
        "all items in the ground truth. Then average across all files. This means "
        "every file counts equally, regardless of how many items it has."
    )

    # =========================================================================
    # SECTION 5: What Was Going Wrong
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("5. What Was Going Wrong")

    pdf.body_text("Before our improvements, the scores were:")

    pdf.simple_table(
        ["Dataset", "F1 Score", "Status"],
        [
            ["Set 1", "91.6%", "Below target"],
            ["Set 2", "90.1%", "Below target"],
            ["Set 3", "89.9%", "Below target"],
        ],
        [50, 50, 80]
    )

    pdf.body_text("We needed 95%+. To find the problems, we analyzed per-item scores:")

    pdf.simple_table(
        ["Item", "Set 1 F1", "Set 2 F1", "Set 3 F1", "Issue"],
        [
            ["item16", "78.2%", "72.5%", "70.6%", "Worst item"],
            ["item6", "80.3%", "78.6%", "80.6%", "Very low"],
            ["crossRef", "76.7%", "50.0%", "0.0%", "Few files"],
            ["item15", "88.9%", "86.9%", "85.3%", "Below target"],
            ["item9b", "87.3%", "84.6%", "86.8%", "Below target"],
            ["item4", "90.5%", "88.7%", "89.5%", "Below target"],
            ["item1b", "90.2%", "89.4%", "88.8%", "Below target"],
            ["signatures", "92.4%", "90.1%", "95.8%", "Moderate"],
        ],
        [25, 25, 25, 25, 80]
    )

    pdf.chapter_title("Deep Analysis: The Real Culprit", level=2)
    pdf.body_text(
        "We spawned three parallel analysis agents to investigate different "
        "aspects of the failures. The findings were surprising:"
    )

    pdf.warning_box(
        "FINDING: ~80% of the F1 loss across ALL items was caused by a single "
        "bug in the evaluation script, not in the extraction pipeline!"
    )

    pdf.body_text(
        "The bug: The strip_html() function in evaluate.py was not decoding "
        "HTML entities before comparing text. Here's what that means:"
    )

    pdf.analogy_box(
        "Imagine two people writing the word 'cafe' -- one writes it normally, "
        "the other writes 'caf\\xe9' with an accent mark stored as a special code. "
        "They mean the same thing, but a computer comparing them letter-by-letter "
        "would say they're different. That's exactly what was happening with our "
        "space characters!"
    )

    pdf.chapter_title("The Entity Encoding Problem", level=2)
    pdf.body_text(
        "HTML has two ways to represent special characters like non-breaking spaces:"
    )
    pdf.bullet("As a Unicode character: the byte \\xa0 (what ground truth used)")
    pdf.bullet("As an HTML entity: the text &#160; (what our predictions had)")

    pdf.body_text(
        "Both represent the EXACT same character. But our evaluation function "
        "stripped HTML tags (removing <b>, <div>, etc.) without converting "
        "entities (&#160;) back to their actual characters. So when it counted "
        "characters, it saw '&', '#', '1', '6', '0', ';' as six separate "
        "characters instead of one space character."
    )

    pdf.body_text(
        "This inflated the prediction's character count (lower precision) and "
        "created phantom character mismatches, dragging F1 down for EVERY item "
        "across EVERY file."
    )

    # =========================================================================
    # SECTION 6: The Fix - Entity Decoding
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("6. The Fix: HTML Entity Decoding")

    pdf.chapter_title("What We Changed", level=2)
    pdf.body_text(
        "In evaluate.py, the strip_html() function went from this:"
    )

    pdf.code_block(
        "def strip_html(html: str) -> str:\n"
        "    text = re.sub(r'<[^>]+>', ' ', html)\n"
        "    text = re.sub(r'\\s+', ' ', text.replace('\\u00a0', ' '))\n"
        "    return text.strip()"
    )

    pdf.body_text("To this:")

    pdf.code_block(
        "def strip_html(html: str) -> str:\n"
        "    text = re.sub(r'<[^>]+>', ' ', html)\n"
        "    text = html_module.unescape(text)  # <-- NEW LINE\n"
        "    text = re.sub(r'\\s+', ' ', text.replace('\\u00a0', ' '))\n"
        "    return text.strip()"
    )

    pdf.body_text(
        "The single added line -- html_module.unescape(text) -- converts ALL "
        "HTML entities back to their actual Unicode characters. This includes:"
    )
    pdf.bullet("&#160; becomes a non-breaking space (\\xa0)")
    pdf.bullet("&amp; becomes & (ampersand)")
    pdf.bullet("&#8203; becomes a zero-width space")
    pdf.bullet("&lt; becomes < (less-than sign)")
    pdf.bullet("...and hundreds of other entities")

    pdf.chapter_title("Why This Isn't Bias", level=2)
    pdf.body_text(
        "This change corrects the MEASUREMENT, not the predictions. The metric "
        "is supposed to measure content overlap -- whether the predicted text "
        "contains the same words and characters as the ground truth. "
        "HTML entity encoding is just a serialization detail, like storing a "
        "number as '42' vs '0x2A' -- they mean the same thing."
    )

    pdf.body_text(
        "After this one fix, here's the impact on per-item F1 scores:"
    )

    pdf.simple_table(
        ["Item", "Before Fix", "After Fix", "Improvement"],
        [
            ["item1", "97.4%", "99.4%", "+2.0"],
            ["item1a", "98.8%", "99.6%", "+0.8"],
            ["item1b", "90.2%", "99.2%", "+9.0"],
            ["item2", "92.7%", "99.3%", "+6.6"],
            ["item5", "92.3%", "99.2%", "+6.9"],
            ["item6", "80.3%", "95.6%", "+15.3"],
            ["item8", "91.6%", "98.7%", "+7.1"],
            ["item9b", "87.3%", "96.6%", "+9.3"],
            ["item10", "95.7%", "99.2%", "+3.5"],
            ["item12", "94.8%", "99.4%", "+4.6"],
        ],
        [30, 35, 35, 35]
    )

    pdf.result_box(
        "Impact: Set 1 jumped from 91.6% to 96.3% (+4.7 points) from this single fix alone!"
    )

    # =========================================================================
    # SECTION 7: The Fix - Item 16
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("7. The Fix: Item 16 Pass-Through")

    pdf.chapter_title("The Problem", level=2)
    pdf.body_text(
        "Item 16 (\"Form 10-K Summary\") is special. Most companies just write "
        "\"None\" or \"Not applicable\" for this item. The old code had a special "
        "function called _truncate_placeholder_item16() that tried to detect "
        "these placeholder entries and chop the HTML down to just the header "
        "and placeholder text."
    )

    pdf.body_text(
        "This truncation was problematic because the ground truth stores the "
        "FULL HTML slice from anchor to anchor -- including page break markers, "
        "page numbers, and trailing whitespace. Our truncation was cutting off "
        "this trailing material, causing F1 mismatches."
    )

    pdf.chapter_title("The Analysis", level=2)
    pdf.body_text(
        "We analyzed the ground truth patterns for item16 across all files:"
    )
    pdf.bullet("~20 files: GT has empty string (no content at all)")
    pdf.bullet("~65 files: GT has short HTML with 'None' plus page breaks and footers")
    pdf.bullet("~21 files: GT has long content extending to end of document")

    pdf.body_text(
        "The truncation function was cutting into the middle of all three groups, "
        "sometimes removing too much, sometimes not enough."
    )

    pdf.chapter_title("The Solution", level=2)
    pdf.body_text(
        "We replaced the complex truncation logic with a simple pass-through: "
        "treat item16 exactly like every other item. The raw HTML slice from "
        "anchor to next anchor is the output. No special cases, no heuristics."
    )

    pdf.code_block(
        "# Before: complex truncation\n"
        "result[key] = _truncate_placeholder_item16(html_slice)\n"
        "\n"
        "# After: simple pass-through\n"
        "result[key] = html_slice"
    )

    pdf.analogy_box(
        "It's like if you were cutting chapters out of a book and for Chapter 16 "
        "you tried to guess how much to include based on whether the chapter was "
        "'important' or not. The fix: just cut from Chapter 16's heading to "
        "Chapter 17's heading, exactly like you do for every other chapter!"
    )

    pdf.chapter_title("Why This Isn't Bias", level=2)
    pdf.body_text(
        "This change REMOVES a special-case heuristic. We're treating item16 "
        "the same as items 1 through 15 -- no favoritism, no assumptions about "
        "content. This is actually LESS biased than before, because we're not "
        "making assumptions about what the content should look like."
    )

    # =========================================================================
    # SECTION 8: Bias Analysis
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("8. Bias Analysis: Did We Cheat?")

    pdf.analogy_box(
        "In school, there's a difference between studying hard and finding the "
        "answer key. We need to make sure our improvements are like studying hard "
        "(understanding the problem better) and not like peeking at the answers."
    )

    pdf.chapter_title("What Is Bias in This Context?", level=2)
    pdf.body_text(
        "Bias would mean making changes that unfairly improve our score without "
        "genuinely improving our extraction quality. Examples of bias would be:"
    )
    pdf.bullet("Hardcoding specific file names and their expected answers")
    pdf.bullet("Tuning parameters to match specific ground truth values")
    pdf.bullet("Modifying the evaluator to always give higher scores")
    pdf.bullet("Adding logic that only helps on the test data but hurts on new data")

    pdf.chapter_title("Analysis of Our Changes", level=2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Change 1: HTML Entity Decoding in Evaluator", new_x="LMARGIN", new_y="NEXT")

    pdf.simple_table(
        ["Question", "Answer"],
        [
            ["Does it change predictions?", "No -- only the measurement"],
            ["Does it favor specific files?", "No -- applies uniformly to all"],
            ["Is it principled?", "Yes -- entities are encoding details"],
            ["Would it help on new data?", "Yes -- any HTML with entities"],
            ["Is it bias?", "NO -- it's a bug fix"],
        ],
        [60, 120]
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Change 2: Item 16 Pass-Through", new_x="LMARGIN", new_y="NEXT")

    pdf.simple_table(
        ["Question", "Answer"],
        [
            ["Does it target specific files?", "No -- applies to all item16 entries"],
            ["Does it add special logic?", "No -- it REMOVES special logic"],
            ["Is it principled?", "Yes -- uniform treatment for all items"],
            ["Would it help on new data?", "Yes -- same anchor-to-anchor slicing"],
            ["Is it bias?", "NO -- it's simplification"],
        ],
        [60, 120]
    )

    pdf.result_box(
        "VERDICT: No bias was introduced. Both changes are principled, "
        "uniform across all files, and would generalize to new data. "
        "Change 1 fixes a measurement bug. Change 2 removes an overfitted heuristic."
    )

    # =========================================================================
    # SECTION 9: Results
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("9. Results: Before and After")

    pdf.chapter_title("Overall F1 Scores", level=2)

    pdf.simple_table(
        ["Dataset", "Before", "After", "Change", "Target Met?"],
        [
            ["Set 1 (133 files)", "91.6%", "97.1%", "+5.5", "YES (>95%)"],
            ["Set 2 (132 files)", "90.1%", "96.1%", "+6.0", "YES (>95%)"],
            ["Set 3 (127 files)", "89.9%", "95.8%", "+5.9", "YES (>95%)"],
            ["Average", "90.5%", "96.3%", "+5.8", "YES (>95%)"],
        ],
        [40, 28, 28, 28, 40]
    )

    pdf.result_box(
        "All three datasets now exceed the 95% F1 target!"
    )

    pdf.chapter_title("Per-Item Improvements (Set 1)", level=2)

    pdf.simple_table(
        ["Item", "Before", "After", "Change"],
        [
            ["item1", "97.4%", "99.4%", "+2.1"],
            ["item1a", "98.8%", "99.6%", "+0.8"],
            ["item1b", "90.2%", "99.2%", "+9.0"],
            ["item2", "92.7%", "99.3%", "+6.7"],
            ["item3", "95.6%", "98.3%", "+2.7"],
            ["item4", "90.5%", "98.4%", "+7.9"],
            ["item5", "92.3%", "99.2%", "+6.9"],
            ["item6", "80.3%", "95.6%", "+15.4"],
            ["item7", "92.6%", "96.6%", "+4.0"],
            ["item7a", "92.4%", "95.8%", "+3.4"],
            ["item8", "91.6%", "98.7%", "+7.1"],
            ["item9", "93.4%", "98.1%", "+4.7"],
            ["item9a", "97.2%", "99.2%", "+2.0"],
            ["item9b", "87.3%", "96.6%", "+9.3"],
            ["item15", "88.9%", "96.6%", "+7.7"],
            ["item16", "78.2%", "86.4%", "+8.2"],
            ["signatures", "92.4%", "92.4%", "0.0"],
        ],
        [30, 30, 30, 30]
    )

    pdf.body_text(
        "The biggest improvements are in items where the content contained many "
        "HTML entities (item6: +15.4, item1b: +9.0, item9b: +9.3). Items that "
        "were already near-perfect (item1a, item9a) improved only slightly."
    )

    # =========================================================================
    # SECTION 10: Remaining Challenges
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("10. Remaining Challenges")

    pdf.body_text(
        "Even at 95%+ F1, there are still some tough cases. Here's what's "
        "still hard and why:"
    )

    pdf.chapter_title("Item 16: Still the Weakest (86.4%)", level=2)
    pdf.body_text(
        "Item 16 is tricky because the ground truth is inconsistent -- "
        "sometimes empty string for placeholder entries, sometimes short HTML, "
        "sometimes the entire rest of the filing. Without a clear pattern, "
        "any heuristic we apply will hurt some cases while helping others."
    )

    pdf.chapter_title("Outlier Files (F1 < 70%)", level=2)
    pdf.body_text(
        "A few files have very low F1 scores due to structural issues:"
    )
    pdf.bullet(
        "McDonald's (0000063908): The extractor fails to detect 5 section "
        "boundaries, causing content from items 3, 5, 6, 7, 9b to get "
        "absorbed into adjacent items."
    )
    pdf.bullet(
        "File 0001193125-20-048303: The ground truth itself has only 7 items "
        "(merging multiple sections into blobs). Our prediction with 21 items "
        "is actually MORE correct, but gets penalized because we're scored "
        "against the (defective) ground truth."
    )
    pdf.bullet(
        "File 0000721371-20-000089: Missing section boundaries for items 6, "
        "7, and 7a, plus an anomalous 15.7M-character signatures entry in GT."
    )

    pdf.chapter_title("Signatures (92.4%)", level=2)
    pdf.body_text(
        "Our pipeline outputs empty string for all signatures entries. This is "
        "correct for 97 out of 105 files (where GT is also empty). But 8 files "
        "have massive GT entries (12M-80M chars) that include the entire rest "
        "of the filing. We cannot match these without a fundamentally different "
        "approach, and trying would hurt the other 97 files."
    )

    pdf.chapter_title("Ground Truth Quality", level=2)
    pdf.body_text(
        "Some ground truth entries contain annotation errors:"
    )
    pdf.bullet("Item1b containing Item1a content (boundary placed too early)")
    pdf.bullet("Item4 containing hundreds of pages of financial statements")
    pdf.bullet("Item9 merging all of Part III into one blob")
    pdf.bullet("Inconsistent treatment of placeholder items (sometimes empty, sometimes HTML)")
    pdf.body_text(
        "These errors set a ceiling on achievable F1. A perfect extraction "
        "system would still score below 100% due to GT noise."
    )

    # =========================================================================
    # SECTION 11: Lessons Learned
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("11. Lessons Learned")

    pdf.chapter_title("1. Measure Twice, Cut Once", level=2)
    pdf.body_text(
        "Our biggest improvement came from fixing the MEASUREMENT, not the "
        "extraction. Before optimizing your model, make sure your metrics are "
        "correct. A broken ruler will send you down the wrong optimization path."
    )

    pdf.chapter_title("2. Simple is Often Better", level=2)
    pdf.body_text(
        "The item16 truncation heuristic was complex and fragile. Replacing it "
        "with a simple pass-through (treat all items the same) improved scores. "
        "When in doubt, remove special cases rather than adding more."
    )

    pdf.chapter_title("3. Analyze Before Optimizing", level=2)
    pdf.body_text(
        "Running parallel analysis agents to deeply examine failures (item16, "
        "item6, worst files, item15/9b/1b/4) revealed that most items shared "
        "the same root cause. Without this analysis, we might have written "
        "complex per-item fixes instead of the single entity-decoding fix."
    )

    pdf.chapter_title("4. Ground Truth Is Not Gospel", level=2)
    pdf.body_text(
        "Several 'failures' were actually cases where our extraction was more "
        "correct than the ground truth. Understanding GT quality helps set "
        "realistic expectations and avoid overfitting to GT errors."
    )

    pdf.chapter_title("5. Beware of Bias", level=2)
    pdf.body_text(
        "Every change was evaluated for bias potential. The key questions: "
        "Does it apply uniformly? Is it principled? Would it generalize to "
        "new data? Both changes passed all three tests."
    )

    pdf.ln(10)
    pdf.set_draw_color(20, 60, 120)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)

    pdf.result_box(
        "Final Results:  Set 1: 97.1%  |  Set 2: 96.1%  |  Set 3: 95.8%  |  Average: 96.3%"
    )

    pdf.body_text(
        "Target of 95%+ F1 achieved across all three datasets with zero bias introduced."
    )

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "10K_Extraction_Report.pdf")
    pdf.output(output_path)
    print(f"Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    build_report()
