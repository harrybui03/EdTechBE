from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Iterable

import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document

from graph.graph import app


# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)


def _format_metadata(metadata: dict | None) -> str:
    if not metadata:
        return ""
    lines = []
    for key, value in metadata.items():
        if value is None:
            continue
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def _format_documents(documents: Iterable[Document] | None, preview_chars: int = 600) -> str:
    docs = list(documents or [])
    if not docs:
        return "_No documents retrieved (fallback to web or empty result)._"

    chunks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        metadata_md = _format_metadata(doc.metadata or {})
        content_preview = (doc.page_content or "").strip()
        if preview_chars and len(content_preview) > preview_chars:
            content_preview = content_preview[:preview_chars].rstrip() + "..."
        section = [
            f"### Document {idx}",
            metadata_md or "_No metadata_",
            "",
            "```text",
            content_preview or "(empty)",
            "```",
        ]
        chunks.append("\n".join(section))
    return "\n\n---\n\n".join(chunks)


def answer_question(
    question: str,
    user_id: str | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> tuple[str, str, list, str, list[tuple[str, str]]]:
    """
    Answer a question with conversation history support.
    
    Returns:
        tuple: (answer, trace, sources, formatted_docs, updated_chat_history)
    """
    if not question or not question.strip():
        return (
            "Please enter a question.",
            "",
            [],
            "_No documents retrieved._",
            chat_history or [],
        )

    buf = io.StringIO()
    payload = {"question": question.strip()}
    if user_id and user_id.strip():
        payload["user_id"] = user_id.strip()
    if chat_history:
        payload["chat_history"] = chat_history

    with redirect_stdout(buf):
        result = app.invoke(input=payload)

    trace = buf.getvalue()
    answer = result.get("generation", str(result))
    sources = result.get("sources", [])
    formatted_docs = _format_documents(result.get("documents"))
    updated_history = result.get("chat_history", chat_history or [])
    
    return answer, trace, sources, formatted_docs, updated_history


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Agentic RAG QA") as demo:
        gr.Markdown("# Agentic RAG • Conversational Q&A")
        
        # Store chat history in component state
        chat_history_state = gr.State(value=[])
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                )
                with gr.Row():
                    question = gr.Textbox(
                        label="Question",
                        placeholder="Ask something... (supports conversation context)",
                        scale=4,
                        show_label=False,
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)
                    clear = gr.Button("Clear", scale=1)
                
                user_id = gr.Textbox(
                    label="User ID (optional)",
                    placeholder="UUID of learner",
                    visible=False,
                )
                
            with gr.Column(scale=1):
                trace = gr.Textbox(label="RAG Trace", lines=15)
                sources = gr.JSON(label="Sources (metadata)")
                documents_md = gr.Markdown(label="Retrieved Documents")

        def chat_fn(message: str, history: list, user_id_input: str | None):
            """Handle chat interaction"""
            if not message or not message.strip():
                return history, history, "", [], "_No documents retrieved._"
            
            # Call answer_question with history
            answer, trace_output, sources_output, docs_output, updated_history = answer_question(
                question=message,
                user_id=user_id_input,
                chat_history=history,
            )
            
            # Use updated_history from answer_question (already includes current Q&A)
            return updated_history, updated_history, trace_output, sources_output, docs_output

        def clear_chat():
            """Clear chat history"""
            return [], "", [], "_No documents retrieved._"

        submit.click(
            fn=chat_fn,
            inputs=[question, chat_history_state, user_id],
            outputs=[chatbot, chat_history_state, trace, sources, documents_md],
        ).then(lambda: "", outputs=question)  # Clear question input after submit

        question.submit(
            fn=chat_fn,
            inputs=[question, chat_history_state, user_id],
            outputs=[chatbot, chat_history_state, trace, sources, documents_md],
        ).then(lambda: "", outputs=question)  # Clear question input after submit

        clear.click(
            fn=clear_chat,
            outputs=[chatbot, trace, sources, documents_md],
        ).then(lambda: [], outputs=chat_history_state)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    # Try to launch on localhost, fallback to share if needed
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except ValueError:
        # If localhost is not accessible, use share mode
        print("Localhost not accessible, launching with share=True...")
        demo.launch(share=True)

