import gradio as gr

# For rendering the PDF.
import fitz
from PIL import Image

N = 0

def render_file(file):
    global N
    doc = fitz.open(file.name)
    page = doc[N]
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image


def render_first(file):
    global N
    N = 0
    return render_file(file)

with gr.Blocks() as demo:
    # First row: chatbot and PDF image.
    with gr.Row():
        with gr.Column(scale=0.5):
            chatbot = gr.Chatbot(value=[], elem_id="chatbot").style(height=650)
        with gr.Column(scale=0.5):
            show_img = gr.Image(label="Upload PDF", tool="select").style(height=680)
    # Second row: query, submit, and upload buttons.
    with gr.Row():
        with gr.Column(scale=0.7):
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter"
            ).style(container=False)
        with gr.Column(scale=0.15):
            submit_btn = gr.Button("Submit")
        with gr.Column(scale=0.15):
            upload_btn = gr.UploadButton("Upload a PDF", file_types=[".pdf"]).style()

    # Set up event handlers.
    upload_btn.upload(fn=render_first, inputs=[upload_btn], outputs=[show_img])

demo.queue()