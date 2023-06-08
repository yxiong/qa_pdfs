"""
Usage: `gradio app.py`

Ref.: https://www.analyticsvidhya.com/blog/2023/05/build-a-chatgpt-for-pdfs-with-langchain/#h-render-image-of-a-pdf-file
"""

import gradio as gr

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# For rendering the PDF.
import fitz
from PIL import Image

COUNT = 0
N = 0
chat_history = []
chain = None

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


def process_file(file):
    loader = PyPDFLoader(file.name)
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    pdfsearch = Chroma.from_documents(documents, embeddings)
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3),
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True)
    return chain

def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history

def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain

    if COUNT == 0:
        chain = process_file(btn)
        COUNT += 1
    result = chain({"question": query, "chat_history": chat_history},
                   return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    N = list(result["source_documents"][0])[1][1]["page"]
    for char in result['answer']:
        history[-1][-1] += char

        # Yield the updated history and an empty string
        yield history, ''

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
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, upload_btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[upload_btn],
        outputs=[show_img]
    )

demo.queue()