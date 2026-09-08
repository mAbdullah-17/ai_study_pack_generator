import io
import json
from typing import Any, Dict

from pypdf import PdfReader
from docx import Document


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_uploaded_file(
    uploaded_file,
) -> str:

    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    try:

        if filename.endswith(".pdf"):

            return extract_pdf(
                uploaded_file
            )

        if filename.endswith(".docx"):

            return extract_docx(
                uploaded_file
            )

        if filename.endswith(".txt"):

            return extract_txt(
                uploaded_file
            )

        raise ValueError(
            "Unsupported file type."
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read '{uploaded_file.name}': {exc}"
        ) from exc


# ============================================================
# PDF
# ============================================================

def extract_pdf(
    uploaded_file,
) -> str:

    data = uploaded_file.read()

    reader = PdfReader(
        io.BytesIO(data)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:

            pages.append(
                text.strip()
            )

    return "\n\n".join(pages).strip()


# ============================================================
# DOCX
# ============================================================

def extract_docx(
    uploaded_file,
) -> str:

    data = uploaded_file.read()

    document = Document(
        io.BytesIO(data)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(text)

    return "\n\n".join(
        paragraphs
    ).strip()


# ============================================================
# TXT
# ============================================================

def extract_txt(
    uploaded_file,
) -> str:

    data = uploaded_file.read()

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "cp1252",
    ):

        try:

            return data.decode(
                encoding
            ).strip()

        except UnicodeDecodeError:

            continue

    raise ValueError(
        "The text file could not be decoded."
    )


# ============================================================
# MARKDOWN EXPORT
# ============================================================

def export_as_markdown(
    pack: Dict[str, Any],
) -> str:

    metadata = pack.get(
        "metadata",
        {},
    )

    lines = []

    lines.append(
        f"# {metadata.get('title', 'Study Pack')}"
    )

    lines.append("")

    lines.append(
        f"**Subject:** "
        f"{metadata.get('subject', '')}"
    )

    lines.append(
        f"**Topic:** "
        f"{metadata.get('topic', '')}"
    )

    lines.append(
        f"**Difficulty:** "
        f"{metadata.get('difficulty', '')}"
    )

    lines.append("")

    lines.append("## Overview")

    lines.append(
        pack.get(
            "overview",
            "",
        )
    )

    lines.append("")

    lines.append("## Key Concepts")

    for concept in pack.get(
        "key_concepts",
        [],
    ):

        if isinstance(concept, dict):

            lines.append(
                f"### {concept.get('name', '')}"
            )

            lines.append(
                concept.get(
                    "explanation",
                    "",
                )
            )

    lines.append("")

    lines.append("## Definitions")

    for definition in pack.get(
        "definitions",
        [],
    ):

        if isinstance(definition, dict):

            lines.append(
                f"- **{definition.get('term', '')}:** "
                f"{definition.get('definition', '')}"
            )

    lines.append("")

    lines.append("## Detailed Notes")

    for note in pack.get(
        "detailed_notes",
        [],
    ):

        if isinstance(note, dict):

            lines.append(
                f"### {note.get('heading', '')}"
            )

            lines.append(
                note.get(
                    "content",
                    "",
                )
            )

    lines.append("")

    lines.append("## Examples")

    for example in pack.get(
        "examples",
        [],
    ):

        if isinstance(example, dict):

            lines.append(
                f"### {example.get('title', '')}"
            )

            lines.append(
                f"**Problem:** "
                f"{example.get('problem', '')}"
            )

            lines.append(
                f"**Solution:** "
                f"{example.get('solution', '')}"
            )

    lines.append("")

    lines.append("## Formulas")

    for formula in pack.get(
        "formulas",
        [],
    ):

        if isinstance(formula, dict):

            lines.append(
                f"### {formula.get('name', '')}"
            )

            lines.append(
                f"`{formula.get('formula', '')}`"
            )

            lines.append(
                formula.get(
                    "explanation",
                    "",
                )
            )

    lines.append("")

    lines.append("## Common Mistakes")

    for mistake in pack.get(
        "common_mistakes",
        [],
    ):

        lines.append(
            f"- {mistake}"
        )

    lines.append("")

    lines.append("## MCQs")

    for index, question in enumerate(
        pack.get("mcqs", []),
        start=1,
    ):

        if not isinstance(question, dict):
            continue

        lines.append(
            f"### {index}. "
            f"{question.get('question', '')}"
        )

        for option in question.get(
            "options",
            [],
        ):

            lines.append(
                f"- {option}"
            )

        lines.append(
            f"**Answer:** "
            f"{question.get('correct_answer', '')}"
        )

        lines.append(
            f"**Explanation:** "
            f"{question.get('explanation', '')}"
        )

    lines.append("")

    lines.append("## Short Questions")

    for index, question in enumerate(
        pack.get(
            "short_questions",
            [],
        ),
        start=1,
    ):

        if isinstance(question, dict):

            lines.append(
                f"### {index}. "
                f"{question.get('question', '')}"
            )

            lines.append(
                f"**Answer:** "
                f"{question.get('answer', '')}"
            )

    lines.append("")

    lines.append("## Long Questions")

    for index, question in enumerate(
        pack.get(
            "long_questions",
            [],
        ),
        start=1,
    ):

        if isinstance(question, dict):

            lines.append(
                f"### {index}. "
                f"{question.get('question', '')}"
            )

            lines.append(
                f"**Answer:** "
                f"{question.get('answer', '')}"
            )

    lines.append("")

    lines.append("## Flashcards")

    for index, card in enumerate(
        pack.get(
            "flashcards",
            [],
        ),
        start=1,
    ):

        if isinstance(card, dict):

            lines.append(
                f"### Card {index}"
            )

            lines.append(
                f"**Front:** "
                f"{card.get('front', '')}"
            )

            lines.append(
                f"**Back:** "
                f"{card.get('back', '')}"
            )

    lines.append("")

    lines.append("## Quick Revision")

    lines.append(
        pack.get(
            "quick_revision",
            "",
        )
    )

    lines.append("")

    lines.append("## Revision Checklist")

    for item in pack.get(
        "revision_checklist",
        [],
    ):

        lines.append(
            f"- [ ] {item}"
        )

    return "\n".join(lines)


# ============================================================
# TXT EXPORT
# ============================================================

def export_as_txt(
    pack: Dict[str, Any],
) -> str:

    markdown = export_as_markdown(
        pack
    )

    return markdown.replace(
        "#",
        "",
    )


# ============================================================
# JSON EXPORT
# ============================================================

def export_as_json(
    pack: Dict[str, Any],
) -> str:

    return json.dumps(
        pack,
        indent=2,
        ensure_ascii=False,
    )
