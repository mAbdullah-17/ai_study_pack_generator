import streamlit as st

from ai_workflow import generate_study_pack, check_groq_connection
from utils import (
    extract_uploaded_file,
    export_as_markdown,
    export_as_txt,
    export_as_json,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #ffffff;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .app-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            color: #64748B;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .section-title {
            color: #0F172A;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1rem;
        }

        .success-box {
            padding: 1rem;
            border-radius: 10px;
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
        }

        .info-box {
            padding: 1rem;
            border-radius: 10px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            background: white;
            text-align: center;
        }

        .stage-complete {
            color: #2E7D32;
            font-weight: 600;
        }

        .stage-pending {
            color: #64748B;
        }

        .question-card {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            background: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "study_pack" not in st.session_state:
    st.session_state.study_pack = None

if "workflow_log" not in st.session_state:
    st.session_state.workflow_log = []

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0

if "flashcard_flipped" not in st.session_state:
    st.session_state.flashcard_flipped = False


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">AI Study Pack Generator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Turn your topic or study material into a personalized, "
    "structured study pack using a multi-stage AI workflow."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Navigation")

    page = st.radio(
        "Go to",
        [
            "Create Study Pack",
            "Study Pack",
            "Quiz",
            "Flashcards",
            "Export",
            "Settings",
        ],
    )

    st.divider()

    st.subheader("AI Workflow")

    st.caption("The generator uses five controlled stages:")

    st.write("1. Planning")
    st.write("2. Content Generation")
    st.write("3. Assessment")
    st.write("4. AI Review")
    st.write("5. Refinement")


# ============================================================
# CREATE STUDY PACK PAGE
# ============================================================

if page == "Create Study Pack":

    st.markdown(
        '<div class="section-title">Create Your Study Pack</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Provide your subject and topic. Uploading study material "
        "is optional."
    )

    col1, col2 = st.columns(2)

    with col1:

        subject = st.text_input(
            "Subject *",
            placeholder="e.g. Physics",
        )

    with col2:

        topic = st.text_input(
            "Topic *",
            placeholder="e.g. Newton's Laws of Motion",
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        education_level = st.selectbox(
            "Education Level",
            [
                "School",
                "Matric / O-Level",
                "Intermediate / A-Level",
                "Undergraduate",
                "Graduate",
                "General",
            ],
        )

    with col2:

        difficulty = st.selectbox(
            "Difficulty",
            [
                "Easy",
                "Medium",
                "Hard",
            ],
            index=1,
        )

    with col3:

        language = st.selectbox(
            "Language",
            [
                "English",
                "Urdu",
            ],
        )

    study_goal = st.text_input(
        "Study Goal",
        placeholder="e.g. Prepare for an upcoming exam",
    )

    exam_type = st.selectbox(
        "Exam Type",
        [
            "General Study",
            "Class Test",
            "Midterm",
            "Final Exam",
            "Competitive Exam",
        ],
    )

    st.markdown("### Study Material")

    st.info(
        "Study material is optional. If you provide material, "
        "the AI will prioritize it when generating the study pack."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
    )

    pasted_text = st.text_area(
        "Or paste study material here",
        height=180,
        placeholder="Paste notes, textbook content, lecture notes, etc.",
    )

    st.markdown("### Generation Settings")

    col1, col2, col3 = st.columns(3)

    with col1:

        mcq_count = st.slider(
            "MCQs",
            min_value=3,
            max_value=15,
            value=8,
        )

    with col2:

        short_count = st.slider(
            "Short Questions",
            min_value=2,
            max_value=10,
            value=5,
        )

    with col3:

        long_count = st.slider(
            "Long Questions",
            min_value=1,
            max_value=5,
            value=3,
        )

    col1, col2 = st.columns(2)

    with col1:

        flashcard_count = st.slider(
            "Flashcards",
            min_value=5,
            max_value=20,
            value=10,
        )

    with col2:

        pack_depth = st.selectbox(
            "Explanation Depth",
            [
                "Concise",
                "Balanced",
                "Detailed",
            ],
            index=1,
        )

    include_formulas = st.checkbox(
        "Include formulas where relevant",
        value=True,
    )

    include_examples = st.checkbox(
        "Include examples",
        value=True,
    )

    include_mistakes = st.checkbox(
        "Include common mistakes",
        value=True,
    )

    st.divider()

    generate_button = st.button(
        "Generate Study Pack",
        type="primary",
        use_container_width=True,
    )

    if generate_button:

        if not subject.strip():
            st.error("Please enter a subject.")

        elif not topic.strip():
            st.error("Please enter a topic.")

        else:

            with st.status(
                "Creating your personalized study pack...",
                expanded=True,
            ) as status:

                try:

                    st.write("Processing your input...")

                    study_material = ""

                    if uploaded_file is not None:

                        extracted = extract_uploaded_file(uploaded_file)

                        if extracted:
                            study_material += extracted

                    if pasted_text.strip():

                        if study_material:
                            study_material += "\n\n"

                        study_material += pasted_text.strip()

                    if not study_material:

                        st.write(
                            "No study material supplied. "
                            "The AI will generate from the requested topic."
                        )

                    settings = {
                        "mcq_count": mcq_count,
                        "short_count": short_count,
                        "long_count": long_count,
                        "flashcard_count": flashcard_count,
                        "pack_depth": pack_depth,
                        "include_formulas": include_formulas,
                        "include_examples": include_examples,
                        "include_mistakes": include_mistakes,
                    }

                    result = generate_study_pack(
                        subject=subject.strip(),
                        topic=topic.strip(),
                        education_level=education_level,
                        difficulty=difficulty,
                        language=language,
                        study_goal=study_goal.strip(),
                        exam_type=exam_type,
                        study_material=study_material,
                        settings=settings,
                    )

                    st.session_state.study_pack = result

                    if result.get("workflow_log"):
                        st.session_state.workflow_log = result[
                            "workflow_log"
                        ]

                    status.update(
                        label="Study pack completed!",
                        state="complete",
                    )

                    st.success(
                        "Your personalized study pack has been generated."
                    )

                    st.info(
                        "Open the 'Study Pack' page from the sidebar "
                        "to view your results."
                    )

                except Exception as exc:

                    status.update(
                        label="Generation failed",
                        state="error",
                    )

                    st.error(
                        f"Unable to generate the study pack: {exc}"
                    )


# ============================================================
# STUDY PACK PAGE
# ============================================================

elif page == "Study Pack":

    st.markdown(
        '<div class="section-title">Your Study Pack</div>',
        unsafe_allow_html=True,
    )

    pack = st.session_state.study_pack

    if not pack:

        st.info(
            "No study pack has been generated yet. "
            "Go to 'Create Study Pack' to begin."
        )

    else:

        metadata = pack.get("metadata", {})

        st.title(
            metadata.get(
                "title",
                f"{metadata.get('subject', '')} — "
                f"{metadata.get('topic', '')}",
            )
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Subject",
                metadata.get("subject", "N/A"),
            )

        with col2:
            st.metric(
                "Difficulty",
                metadata.get("difficulty", "N/A"),
            )

        with col3:
            st.metric(
                "Level",
                metadata.get("education_level", "N/A"),
            )

        with col4:
            st.metric(
                "Quality",
                f"{pack.get('quality_score', 'N/A')}",
            )

        tabs = st.tabs(
            [
                "Overview",
                "Concepts",
                "Notes",
                "Examples",
                "Formulas",
                "Mistakes",
                "Questions",
                "Quick Revision",
                "Checklist",
            ]
        )

        # ----------------------------------------------------
        # OVERVIEW
        # ----------------------------------------------------

        with tabs[0]:

            st.subheader("Overview")

            st.write(
                pack.get(
                    "overview",
                    "No overview available.",
                )
            )

            objectives = pack.get(
                "learning_objectives",
                [],
            )

            if objectives:

                st.subheader("Learning Objectives")

                for objective in objectives:
                    st.checkbox(
                        objective,
                        value=False,
                        key=f"objective_{objective}",
                    )

        # ----------------------------------------------------
        # CONCEPTS
        # ----------------------------------------------------

        with tabs[1]:

            st.subheader("Key Concepts")

            concepts = pack.get(
                "key_concepts",
                [],
            )

            if not concepts:

                st.info("No key concepts available.")

            else:

                for concept in concepts:

                    if isinstance(concept, dict):

                        name = concept.get(
                            "name",
                            "Concept",
                        )

                        explanation = concept.get(
                            "explanation",
                            "",
                        )

                        with st.expander(name):

                            st.write(explanation)

                    else:

                        st.write(f"• {concept}")

        # ----------------------------------------------------
        # NOTES
        # ----------------------------------------------------

        with tabs[2]:

            st.subheader("Detailed Notes")

            notes = pack.get(
                "detailed_notes",
                [],
            )

            if isinstance(notes, list):

                for note in notes:

                    if isinstance(note, dict):

                        st.markdown(
                            f"### {note.get('heading', 'Topic')}"
                        )

                        st.write(
                            note.get(
                                "content",
                                "",
                            )
                        )

                    else:

                        st.write(note)

            else:

                st.write(notes)

        # ----------------------------------------------------
        # EXAMPLES
        # ----------------------------------------------------

        with tabs[3]:

            st.subheader("Examples")

            examples = pack.get(
                "examples",
                [],
            )

            if not examples:

                st.info("No examples were generated.")

            else:

                for index, example in enumerate(
                    examples,
                    start=1,
                ):

                    if isinstance(example, dict):

                        st.markdown(
                            f"### Example {index}: "
                            f"{example.get('title', '')}"
                        )

                        st.write(
                            example.get(
                                "problem",
                                "",
                            )
                        )

                        st.write(
                            example.get(
                                "solution",
                                "",
                            )
                        )

                    else:

                        st.write(example)

        # ----------------------------------------------------
        # FORMULAS
        # ----------------------------------------------------

        with tabs[4]:

            st.subheader("Formulas")

            formulas = pack.get(
                "formulas",
                [],
            )

            if not formulas:

                st.info(
                    "No dedicated formulas are required "
                    "for this topic."
                )

            else:

                for formula in formulas:

                    if isinstance(formula, dict):

                        st.markdown(
                            f"**{formula.get('name', 'Formula')}**"
                        )

                        st.code(
                            formula.get(
                                "formula",
                                "",
                            )
                        )

                        st.write(
                            formula.get(
                                "explanation",
                                "",
                            )
                        )

                    else:

                        st.code(str(formula))

        # ----------------------------------------------------
        # COMMON MISTAKES
        # ----------------------------------------------------

        with tabs[5]:

            st.subheader("Common Mistakes")

            mistakes = pack.get(
                "common_mistakes",
                [],
            )

            if not mistakes:

                st.info("No common mistakes were generated.")

            else:

                for mistake in mistakes:
                    st.warning(str(mistake))

        # ----------------------------------------------------
        # QUESTIONS
        # ----------------------------------------------------

        with tabs[6]:

            st.subheader("MCQs")

            mcqs = pack.get(
                "mcqs",
                [],
            )

            for index, question in enumerate(
                mcqs,
                start=1,
            ):

                if not isinstance(question, dict):
                    continue

                st.markdown(
                    f"**{index}. "
                    f"{question.get('question', '')}**"
                )

                options = question.get(
                    "options",
                    [],
                )

                for option in options:
                    st.write(f"- {option}")

                with st.expander("Show answer"):

                    st.success(
                        question.get(
                            "correct_answer",
                            "N/A",
                        )
                    )

                    st.write(
                        question.get(
                            "explanation",
                            "",
                        )
                    )

            st.divider()

            st.subheader("Short Questions")

            short_questions = pack.get(
                "short_questions",
                [],
            )

            for index, question in enumerate(
                short_questions,
                start=1,
            ):

                if isinstance(question, dict):

                    st.markdown(
                        f"**{index}. "
                        f"{question.get('question', '')}**"
                    )

                    with st.expander("Show answer"):

                        st.write(
                            question.get(
                                "answer",
                                "",
                            )
                        )

            st.divider()

            st.subheader("Long Questions")

            long_questions = pack.get(
                "long_questions",
                [],
            )

            for index, question in enumerate(
                long_questions,
                start=1,
            ):

                if isinstance(question, dict):

                    st.markdown(
                        f"**{index}. "
                        f"{question.get('question', '')}**"
                    )

                    with st.expander("Show answer"):

                        st.write(
                            question.get(
                                "answer",
                                "",
                            )
                        )

        # ----------------------------------------------------
        # QUICK REVISION
        # ----------------------------------------------------

        with tabs[7]:

            st.subheader("Quick Revision Sheet")

            st.write(
                pack.get(
                    "quick_revision",
                    "No quick revision sheet available.",
                )
            )

        # ----------------------------------------------------
        # CHECKLIST
        # ----------------------------------------------------

        with tabs[8]:

            st.subheader("Revision Checklist")

            checklist = pack.get(
                "revision_checklist",
                [],
            )

            if checklist:

                for item in checklist:

                    st.checkbox(
                        item,
                        key=f"check_{item}",
                    )

            else:

                st.info("No checklist available.")


# ============================================================
# QUIZ PAGE
# ============================================================

elif page == "Quiz":

    st.markdown(
        '<div class="section-title">Practice Quiz</div>',
        unsafe_allow_html=True,
    )

    pack = st.session_state.study_pack

    if not pack:

        st.info(
            "Generate a study pack first."
        )

    else:

        mcqs = pack.get(
            "mcqs",
            [],
        )

        if not mcqs:

            st.warning(
                "This study pack does not contain MCQs."
            )

        else:

            for index, question in enumerate(
                mcqs,
                start=1,
            ):

                if not isinstance(question, dict):
                    continue

                st.markdown(
                    f"### Question {index}"
                )

                st.write(
                    question.get(
                        "question",
                        "",
                    )
                )

                options = question.get(
                    "options",
                    [],
                )

                answer = st.radio(
                    "Select an answer:",
                    options,
                    key=f"quiz_{index}",
                    index=None,
                )

                st.session_state.quiz_answers[
                    index
                ] = answer

            if st.button(
                "Submit Quiz",
                type="primary",
            ):

                score = 0

                for index, question in enumerate(
                    mcqs,
                    start=1,
                ):

                    selected = st.session_state.quiz_answers.get(
                        index
                    )

                    correct = question.get(
                        "correct_answer"
                    )

                    if selected == correct:
                        score += 1

                st.session_state.quiz_submitted = True

                st.success(
                    f"Your score: {score}/{len(mcqs)}"
                )

                percentage = (
                    score / len(mcqs) * 100
                )

                st.metric(
                    "Percentage",
                    f"{percentage:.1f}%",
                )

                st.subheader("Answer Review")

                for index, question in enumerate(
                    mcqs,
                    start=1,
                ):

                    selected = st.session_state.quiz_answers.get(
                        index
                    )

                    correct = question.get(
                        "correct_answer"
                    )

                    if selected == correct:

                        st.success(
                            f"Question {index}: Correct"
                        )

                    else:

                        st.error(
                            f"Question {index}: Incorrect"
                        )

                        st.write(
                            f"Correct answer: {correct}"
                        )

                        st.write(
                            question.get(
                                "explanation",
                                "",
                            )
                        )

            if st.button("Reset Quiz"):

                st.session_state.quiz_answers = {}

                st.session_state.quiz_submitted = False

                st.rerun()


# ============================================================
# FLASHCARDS PAGE
# ============================================================

elif page == "Flashcards":

    st.markdown(
        '<div class="section-title">Flashcards</div>',
        unsafe_allow_html=True,
    )

    pack = st.session_state.study_pack

    if not pack:

        st.info(
            "Generate a study pack first."
        )

    else:

        cards = pack.get(
            "flashcards",
            [],
        )

        if not cards:

            st.info(
                "No flashcards were generated."
            )

        else:

            total = len(cards)

            if st.session_state.flashcard_index >= total:

                st.session_state.flashcard_index = 0

            index = st.session_state.flashcard_index

            card = cards[index]

            st.caption(
                f"Card {index + 1} of {total}"
            )

            if isinstance(card, dict):

                if st.session_state.flashcard_flipped:

                    st.markdown(
                        "### Answer"
                    )

                    st.info(
                        card.get(
                            "back",
                            "",
                        )
                    )

                else:

                    st.markdown(
                        "### Question"
                    )

                    st.info(
                        card.get(
                            "front",
                            "",
                        )
                    )

            col1, col2, col3 = st.columns(3)

            with col1:

                if st.button(
                    "Previous",
                    use_container_width=True,
                ):

                    st.session_state.flashcard_index = (
                        index - 1
                    ) % total

                    st.session_state.flashcard_flipped = False

                    st.rerun()

            with col2:

                if st.button(
                    "Flip Card",
                    use_container_width=True,
                ):

                    st.session_state.flashcard_flipped = (
                        not st.session_state.flashcard_flipped
                    )

                    st.rerun()

            with col3:

                if st.button(
                    "Next",
                    use_container_width=True,
                ):

                    st.session_state.flashcard_index = (
                        index + 1
                    ) % total

                    st.session_state.flashcard_flipped = False

                    st.rerun()


# ============================================================
# EXPORT PAGE
# ============================================================

elif page == "Export":

    st.markdown(
        '<div class="section-title">Export Study Pack</div>',
        unsafe_allow_html=True,
    )

    pack = st.session_state.study_pack

    if not pack:

        st.info(
            "Generate a study pack before exporting."
        )

    else:

        st.write(
            "Download your study pack in your preferred format."
        )

        markdown_content = export_as_markdown(pack)

        txt_content = export_as_txt(pack)

        json_content = export_as_json(pack)

        st.download_button(
            "Download Markdown",
            data=markdown_content,
            file_name="study_pack.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.download_button(
            "Download TXT",
            data=txt_content,
            file_name="study_pack.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.download_button(
            "Download JSON",
            data=json_content,
            file_name="study_pack.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# SETTINGS PAGE
# ============================================================

elif page == "Settings":

    st.markdown(
        '<div class="section-title">Settings</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Groq API")

    if st.button("Test Groq Connection"):

        with st.spinner("Testing connection..."):

            try:

                result = check_groq_connection()

                if result:

                    st.success(
                        "Groq API connection is working."
                    )

                else:

                    st.error(
                        "Groq API connection failed."
                    )

            except Exception as exc:

                st.error(
                    f"Connection test failed: {exc}"
                )

    st.info(
        "Your Groq API key is stored in Streamlit Secrets "
        "and is never displayed."
    )

    st.subheader("Application")

    st.write(
        "AI Study Pack Generator uses a multi-stage AI "
        "workflow to plan, generate, assess, review, "
        "and refine personalized study material."
    )
