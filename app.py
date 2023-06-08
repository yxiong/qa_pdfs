import gradio as gr
import openai
import time

messages = []
cost = 0

def add_text(history, text):
    global messages
    history = history + [(text, '')]
    messages = messages + [{"role": "user", "content": text}]
    return history, ""

def generate_response(history, model):
    global messages, cost

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2)

    response_msg = response.choices[0].message.content
    cost = cost + (response.usage['total_tokens']) * (0.002 / 1000)
    messages = messages + [{"role": 'assistant', 'content': response_msg}]

    for char in response_msg:
        history[-1][1] += char
        time.sleep(0.05)
        yield history

def calc_cost():
    return cost

with gr.Blocks() as demo:
    radio = gr.Radio(value='gpt-3.5-turbo', choices=['gpt-3.5-turbo','gpt-4'], label='models')
    chatbot = gr.Chatbot(value=[], elem_id="chatbot").style(height=650)
    with gr.Row():
        with gr.Column(scale=0.9):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
        with gr.Column(scale=0.10):
            cost_view = gr.Textbox(label='usage in $', value=0)

    txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        generate_response, inputs =[chatbot,radio],outputs = chatbot).then(
        calc_cost, outputs=cost_view)


demo.queue()