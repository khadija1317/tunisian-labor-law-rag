import gradio as gr
from src.agents.dispatcher import handle_query  # adjust path if different

def query_assistant(question):
    if not question or not question.strip():
        return "Veuillez entrer une question.", "", ""

    result = handle_query(question)

    answer = result.get("final_answer", "")
    citations = result.get("citations", []) or []
    status = result.get("status", "unknown")

    citations_text = "\n".join(f"- {c}" for c in citations) if citations else "Aucune citation."

    return answer, citations_text, f"Status: {status}"

with gr.Blocks(title="Tunisian Labor Law Assistant") as demo:
    gr.Markdown("# Tunisian Labor Law & CNSS Assistant")
    gr.Markdown("Ask a question about the Code du Travail or CNSS procedures (loans, pensions).")

    question_input = gr.Textbox(
        label="Your question",
        placeholder="e.g. Quelle est la durée du préavis de licenciement ?",
        lines=2
    )
    submit_btn = gr.Button("Ask", variant="primary")

    answer_output = gr.Textbox(label="Answer", lines=6)
    with gr.Row():
        citations_output = gr.Textbox(label="Citations", lines=3)
        status_output = gr.Textbox(label="System status", lines=1)

    submit_btn.click(
        fn=query_assistant,
        inputs=question_input,
        outputs=[answer_output, citations_output, status_output]
    )

if __name__ == "__main__":
    demo.launch()