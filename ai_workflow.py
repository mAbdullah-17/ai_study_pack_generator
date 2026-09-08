import json
import re
from typing import Any, Dict

import streamlit as st
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Keep uploaded material reasonably small.
# This is characters, not tokens.
MAX_MATERIAL_CHARS = 12000

# Retry only twice after the first attempt.
MAX_RETRIES = 2


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    return Groq(api_key=api_key)


def get_model() -> str:
    return st.secrets.get(
        "GROQ_MODEL",
        DEFAULT_MODEL
    )


# ============================================================
# JSON HANDLING
# ============================================================

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Safely extract a JSON object from an AI response.
    """

    if not text:
        raise ValueError(
            "The AI returned an empty response."
        )

    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    text = text.strip()

    # Try normal JSON first
    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Find first JSON object
    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object was found in the AI response."
        )

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):

        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\" and in_string:
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:

                candidate = text[
                    start:i + 1
                ]

                try:
                    result = json.loads(
                        candidate
                    )

                    if isinstance(result, dict):
                        return result

                except json.JSONDecodeError:
                    break

    raise ValueError(
        "The AI returned invalid JSON."
    )


# ============================================================
# GROQ JSON REQUEST
# ============================================================

def call_groq_json(
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:

    client = get_groq_client()

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:

            retry_instruction = ""

            if attempt > 0:

                retry_instruction = """

IMPORTANT:
Your previous response could not be parsed.

Return ONLY a valid JSON object.
Do not use Markdown.
Do not use ```json.
Do not add explanations.
Use double quotes.
Do not use trailing commas.
"""

            response = client.chat.completions.create(
                model=get_model(),

                messages=[
                    {
                        "role": "system",
                        "content": (
                            system_prompt
                            + retry_instruction
                        )
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=0.1,

                response_format={
                    "type": "json_object"
                }
            )

            if not response.choices:
                raise RuntimeError(
                    "Groq returned no response choices."
                )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            return extract_json_object(
                content
            )

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:
                continue

            raise RuntimeError(
                "AI JSON generation failed after "
                f"{MAX_RETRIES + 1} attempts: "
                f"{last_error}"
            ) from last_error


# ============================================================
# MATERIAL LIMITER
# ============================================================

def limit_material(
    material: str
) -> str:

    if not material:
        return ""

    material = material.strip()

    if len(material) <= MAX_MATERIAL_CHARS:
        return material

    return (
        material[:MAX_MATERIAL_CHARS]
        + "\n\n[Additional material omitted.]"
    )


# ============================================================
# SMALL CONTEXT HELPERS
# ============================================================

def compact_student_context(
    context: Dict[str, Any]
) -> Dict[str, Any]:

    student = context["student"]

    return {
        "subject": student["subject"],
        "topic": student["topic"],
        "level": student["education_level"],
        "difficulty": student["difficulty"],
        "language": student["language"],
        "goal": student["study_goal"],
        "exam": student["exam_type"],
    }


def compact_settings(
    settings: Dict[str, Any]
) -> Dict[str, Any]:

    return {
        "mcq_count": settings.get(
            "mcq_count",
            8
        ),
        "short_count": settings.get(
            "short_count",
            5
        ),
        "long_count": settings.get(
            "long_count",
            3
        ),
        "flashcard_count": settings.get(
            "flashcard_count",
            10
        ),
    }


# ============================================================
# STAGE 1 — PLANNING
# ============================================================

def create_study_plan(
    subject: str,
    topic: str,
    education_level: str,
    difficulty: str,
    language: str,
    study_goal: str,
    exam_type: str,
    study_material: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:

    system_prompt = """
You are an expert educational planner.

Create a concise personalized study plan.

Focus only on:
- learning objectives
- important topics
- subtopics
- concepts
- assessment priorities
- revision strategy

Do not create the full study pack.

Return ONLY valid JSON.
"""

    user_prompt = f"""
Student:

Subject: {subject}
Topic: {topic}
Level: {education_level}
Difficulty: {difficulty}
Language: {language}
Goal: {study_goal or "General learning"}
Exam: {exam_type}

Settings:
{json.dumps(
    compact_settings(settings),
    ensure_ascii=False
)}

Study material:
{study_material or "None"}

Return:

{{
    "learning_objectives": [],
    "key_topics": [],
    "subtopics": [],
    "important_concepts": [],
    "assessment_focus": [],
    "revision_strategy": []
}}
"""

    return call_groq_json(
        system_prompt,
        user_prompt
    )


# ============================================================
# STAGE 2 — CONTENT GENERATION
# ============================================================

def generate_learning_content(
    context: Dict[str, Any]
) -> Dict[str, Any]:

    system_prompt = """
You are an expert teacher.

Create concise but useful study notes based on
the supplied student profile and study plan.

Avoid unnecessary repetition.

Return ONLY valid JSON.
"""

    student = compact_student_context(
        context
    )

    plan = context[
        "study_plan"
    ]

    material = context.get(
        "study_material",
        ""
    )

    user_prompt = f"""
Student:
{json.dumps(
    student,
    ensure_ascii=False
)}

Study plan:
{json.dumps(
    plan,
    ensure_ascii=False
)}

Study material:
{material or "None"}

Create:

{{
    "overview": "",
    "learning_objectives": [],
    "key_concepts": [
        {{
            "name": "",
            "explanation": ""
        }}
    ],
    "definitions": [
        {{
            "term": "",
            "definition": ""
        }}
    ],
    "detailed_notes": [
        {{
            "heading": "",
            "content": ""
        }}
    ],
    "examples": [
        {{
            "title": "",
            "problem": "",
            "solution": ""
        }}
    ],
    "formulas": [
        {{
            "name": "",
            "formula": "",
            "explanation": ""
        }}
    ],
    "memory_tips": [],
    "common_mistakes": []
}}
"""

    return call_groq_json(
        system_prompt,
        user_prompt
    )


# ============================================================
# STAGE 3 — ASSESSMENT
# ============================================================

def generate_assessment(
    context: Dict[str, Any]
) -> Dict[str, Any]:

    system_prompt = """
You are an expert exam-question designer.

Create assessment questions from the supplied
learning content.

Questions must be:
- relevant
- non-duplicated
- appropriate for the student's level
- appropriate for the selected difficulty

Return ONLY valid JSON.
"""

    settings = compact_settings(
        context["settings"]
    )

    student = compact_student_context(
        context
    )

    learning = context[
        "learning_content"
    ]

    user_prompt = f"""
Student:
{json.dumps(
    student,
    ensure_ascii=False
)}

Learning content:
{json.dumps(
    learning,
    ensure_ascii=False
)}

Requested quantities:
{json.dumps(
    settings,
    ensure_ascii=False
)}

Return:

{{
    "mcqs": [
        {{
            "question": "",
            "options": [],
            "correct_answer": "",
            "explanation": "",
            "difficulty": ""
        }}
    ],

    "short_questions": [
        {{
            "question": "",
            "answer": "",
            "difficulty": ""
        }}
    ],

    "long_questions": [
        {{
            "question": "",
            "answer": "",
            "difficulty": ""
        }}
    ],

    "flashcards": [
        {{
            "front": "",
            "back": ""
        }}
    ]
}}
"""

    return call_groq_json(
        system_prompt,
        user_prompt
    )


# ============================================================
# STAGE 4 — REVIEW
# ============================================================

def review_study_pack(
    context: Dict[str, Any]
) -> Dict[str, Any]:

    system_prompt = """
You are a strict educational quality reviewer.

Review the generated study pack.

Do NOT rewrite the study pack.

Identify only:
- factual problems
- missing important concepts
- unclear explanations
- duplicate questions
- incorrect answers
- difficulty problems
- important improvements

Keep the review concise.

Return ONLY valid JSON.
"""

    learning = context[
        "learning_content"
    ]

    assessment = context[
        "assessment"
    ]

    user_prompt = f"""
Study content:

{json.dumps(
    learning,
    ensure_ascii=False
)}

Assessment:

{json.dumps(
    assessment,
    ensure_ascii=False
)}

Return:

{{
    "overall_score": 0,
    "passed": true,
    "issues": [],
    "missing_concepts": [],
    "content_corrections": [],
    "question_corrections": [],
    "recommended_refinements": []
}}
"""

    return call_groq_json(
        system_prompt,
        user_prompt
    )


# ============================================================
# STAGE 5 — REFINEMENT
# ============================================================

def refine_study_pack(
    context: Dict[str, Any]
) -> Dict[str, Any]:

    system_prompt = """
You are an educational editor.

Correct the study pack using ONLY the issues identified
in the review.

Do not unnecessarily rewrite correct information.

Keep the output concise.

Return ONLY valid JSON.
"""

    # IMPORTANT:
    # We deliberately do NOT send:
    # - full student profile
    # - full study plan
    #
    # This significantly reduces token usage.

    learning = context[
        "learning_content"
    ]

    assessment = context[
        "assessment"
    ]

    review = context[
        "review"
    ]

    user_prompt = f"""
CURRENT CONTENT:

{json.dumps(
    learning,
    ensure_ascii=False
)}

CURRENT ASSESSMENT:

{json.dumps(
    assessment,
    ensure_ascii=False
)}

QUALITY REVIEW:

{json.dumps(
    review,
    ensure_ascii=False
)}

Return the corrected complete content:

{{
    "learning_content": {{
        "overview": "",
        "learning_objectives": [],
        "key_concepts": [],
        "definitions": [],
        "detailed_notes": [],
        "examples": [],
        "formulas": [],
        "memory_tips": [],
        "common_mistakes": []
    }},

    "assessment": {{
        "mcqs": [],
        "short_questions": [],
        "long_questions": [],
        "flashcards": []
    }}
}}
"""

    return call_groq_json(
        system_prompt,
        user_prompt
    )


# ============================================================
# QUICK REVISION
# ============================================================

def build_quick_revision(
    learning: Dict[str, Any]
) -> str:

    sections = []

    overview = learning.get(
        "overview",
        ""
    )

    if overview:
        sections.append(
            "OVERVIEW\n" + overview
        )

    concepts = learning.get(
        "key_concepts",
        []
    )

    concept_lines = []

    for item in concepts:

        if isinstance(item, dict):

            name = item.get(
                "name",
                ""
            )

            explanation = item.get(
                "explanation",
                ""
            )

            if name:
                concept_lines.append(
                    f"- {name}: {explanation}"
                )

    if concept_lines:

        sections.append(
            "KEY CONCEPTS\n"
            + "\n".join(
                concept_lines
            )
        )

    tips = learning.get(
        "memory_tips",
        []
    )

    if tips:

        sections.append(
            "MEMORY TIPS\n"
            + "\n".join(
                f"- {tip}"
                for tip in tips
            )
        )

    mistakes = learning.get(
        "common_mistakes",
        []
    )

    if mistakes:

        sections.append(
            "COMMON MISTAKES\n"
            + "\n".join(
                f"- {item}"
                for item in mistakes
            )
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# REVISION CHECKLIST
# ============================================================

def build_revision_checklist(
    learning: Dict[str, Any]
):

    checklist = []

    for objective in learning.get(
        "learning_objectives",
        []
    ):

        if objective:

            checklist.append(
                f"Understand: {objective}"
            )

    for concept in learning.get(
        "key_concepts",
        []
    ):

        if isinstance(
            concept,
            dict
        ):

            name = concept.get(
                "name",
                ""
            )

            if name:

                checklist.append(
                    f"Revise: {name}"
                )

    checklist.extend([
        "Review the notes",
        "Review memory tips",
        "Review common mistakes",
        "Practice MCQs",
        "Practice short questions",
        "Practice long questions",
        "Review flashcards"
    ])

    return checklist


# ============================================================
# FINAL PACK
# ============================================================

def build_final_pack(
    context: Dict[str, Any]
) -> Dict[str, Any]:

    student = context[
        "student"
    ]

    learning = context[
        "learning_content"
    ]

    assessment = context[
        "assessment"
    ]

    review = context[
        "review"
    ]

    return {

        "metadata": {
            "title": (
                f"{student['subject']} - "
                f"{student['topic']}"
            ),
            "subject": student[
                "subject"
            ],
            "topic": student[
                "topic"
            ],
            "education_level": student[
                "education_level"
            ],
            "difficulty": student[
                "difficulty"
            ],
            "language": student[
                "language"
            ],
            "study_goal": student[
                "study_goal"
            ],
            "exam_type": student[
                "exam_type"
            ]
        },

        "overview": learning.get(
            "overview",
            ""
        ),

        "learning_objectives": learning.get(
            "learning_objectives",
            []
        ),

        "key_concepts": learning.get(
            "key_concepts",
            []
        ),

        "definitions": learning.get(
            "definitions",
            []
        ),

        "detailed_notes": learning.get(
            "detailed_notes",
            []
        ),

        "examples": learning.get(
            "examples",
            []
        ),

        "formulas": learning.get(
            "formulas",
            []
        ),

        "memory_tips": learning.get(
            "memory_tips",
            []
        ),

        "common_mistakes": learning.get(
            "common_mistakes",
            []
        ),

        "mcqs": assessment.get(
            "mcqs",
            []
        ),

        "short_questions": assessment.get(
            "short_questions",
            []
        ),

        "long_questions": assessment.get(
            "long_questions",
            []
        ),

        "flashcards": assessment.get(
            "flashcards",
            []
        ),

        "quick_revision": build_quick_revision(
            learning
        ),

        "revision_checklist":
            build_revision_checklist(
                learning
            ),

        "quality_score": review.get(
            "overall_score",
            0
        ),

        "quality_review": review,

        "workflow_log": context.get(
            "workflow_log",
            []
        )
    }


# ============================================================
# MAIN WORKFLOW
# ============================================================

def generate_study_pack(
    subject: str,
    topic: str,
    education_level: str,
    difficulty: str,
    language: str,
    study_goal: str,
    exam_type: str,
    study_material: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:

    """
    Five-stage AI study-pack workflow:

    1. Planning
    2. Content Generation
    3. Assessment
    4. Review
    5. Refinement

    Context is passed between stages,
    but unnecessary information is removed
    before large AI calls.
    """

    material = limit_material(
        study_material
    )

    context = {

        "student": {
            "subject": subject,
            "topic": topic,
            "education_level":
                education_level,
            "difficulty":
                difficulty,
            "language":
                language,
            "study_goal":
                study_goal,
            "exam_type":
                exam_type
        },

        "study_material":
            material,

        "settings":
            settings,

        "study_plan":
            None,

        "learning_content":
            None,

        "assessment":
            None,

        "review":
            None,

        "workflow_log":
            []
    }

    # ========================================================
    # STAGE 1
    # ========================================================

    context[
        "workflow_log"
    ].append(
        "Planning started"
    )

    try:

        context[
            "study_plan"
        ] = create_study_plan(
            subject,
            topic,
            education_level,
            difficulty,
            language,
            study_goal,
            exam_type,
            material,
            settings
        )

    except Exception as exc:

        raise RuntimeError(
            f"Planning stage failed: {exc}"
        ) from exc

    context[
        "workflow_log"
    ].append(
        "Planning completed"
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    context[
        "workflow_log"
    ].append(
        "Content generation started"
    )

    try:

        context[
            "learning_content"
        ] = generate_learning_content(
            context
        )

    except Exception as exc:

        raise RuntimeError(
            f"Content generation stage failed: {exc}"
        ) from exc

    context[
        "workflow_log"
    ].append(
        "Content generation completed"
    )

    # ========================================================
    # STAGE 3
    # ========================================================

    context[
        "workflow_log"
    ].append(
        "Assessment generation started"
    )

    try:

        context[
            "assessment"
        ] = generate_assessment(
            context
        )

    except Exception as exc:

        raise RuntimeError(
            f"Assessment stage failed: {exc}"
        ) from exc

    context[
        "workflow_log"
    ].append(
        "Assessment generation completed"
    )

    # ========================================================
    # STAGE 4
    # ========================================================

    context[
        "workflow_log"
    ].append(
        "Quality review started"
    )

    try:

        context[
            "review"
        ] = review_study_pack(
            context
        )

    except Exception as exc:

        raise RuntimeError(
            f"Review stage failed: {exc}"
        ) from exc

    context[
        "workflow_log"
    ].append(
        "Quality review completed"
    )

    # ========================================================
    # STAGE 5
    # ========================================================

    # Only refine if the reviewer found problems.

    if not context[
        "review"
    ].get(
        "passed",
        False
    ):

        context[
            "workflow_log"
        ].append(
            "Refinement started"
        )

        try:

            refined = refine_study_pack(
                context
            )

            if refined.get(
                "learning_content"
            ):

                context[
                    "learning_content"
                ] = refined[
                    "learning_content"
                ]

            if refined.get(
                "assessment"
            ):

                context[
                    "assessment"
                ] = refined[
                    "assessment"
                ]

        except Exception as exc:

            raise RuntimeError(
                f"Refinement stage failed: {exc}"
            ) from exc

        context[
            "workflow_log"
        ].append(
            "Refinement completed"
        )

    else:

        context[
            "workflow_log"
        ].append(
            "Refinement not required"
        )

    # ========================================================
    # FINAL
    # ========================================================

    context[
        "workflow_log"
    ].append(
        "Study pack finalized"
    )

    return build_final_pack(
        context
    )


# ============================================================
# CONNECTION TEST
# ============================================================

def check_groq_connection() -> bool:

    try:

        result = call_groq_json(

            system_prompt="""
You are a connection test.

Return ONLY valid JSON.
""",

            user_prompt="""
Return:

{
    "status": "OK"
}
"""
        )

        return (
            result.get("status")
            == "OK"
        )

    except Exception:

        return False
