import datetime
import os
import random
import re
from io import StringIO
from threading import Thread

import gradio as gr
import pandas as pd
from huggingface_hub import upload_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

from dialogues import DialogueTemplate

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_TOKEN = os.environ.get("API_TOKEN", None)
DIALOGUES_DATASET = "smangrul/codegen-25-instrcut-dialogues"
ENDPOINT = os.environ.get("ENDPOINT", None)
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully \
as possible, while being safe. Your answers should not include any harmful, \
unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that \
your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why \
instead of answering something not correct. If you don’t know the answer to a \
question, please don’t share false information."""

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", torch_device)
print("CPU threads:", torch.get_num_threads())

base_model_name_or_path = "bigcode/starcoderplus"
peft_models = [
    "smangrul/peft-lora-starcoderplus-chat-asst-A100-40GB-colab",
    "smangrul/peft-lora-starcoderplus-personal-copilot-A100-40GB-colab",
]

tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_auth_token=HF_TOKEN,
    trust_remote_code=True,
)

for model_name_or_path in peft_models:
    model = PeftModel.from_pretrained(model, model_name_or_path)
    model = model.merge_and_unload()

model.to(torch_device)
model.eval()
print(model)


def randomize_seed_generator():
    seed = random.randint(0, 1000000)
    return seed


def wrap_html_code(text):
    pattern = r"<.*?>"
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        return f"```{text}```"
    else:
        return text


def has_no_history(chatbot, history):
    return not chatbot and not history


def run_generate(
    RETRY_FLAG,
    user_message,
    chatbot,
    history,
    temperature,
    top_k,
    top_p,
    max_new_tokens,
    repetition_penalty,
):

    # Don't return meaningless message when the input is empty
    if not user_message:
        print("Empty input")

    if not RETRY_FLAG:
        history.append(user_message)
        seed = 42
    else:
        seed = randomize_seed_generator()

    past_messages = []
    for data in chatbot:
        user_data, model_data = data

        past_messages.extend(
            [{"role": "user", "content": user_data}, {"role": "assistant", "content": model_data.rstrip()}]
        )

    if len(past_messages) < 1:
        dialogue_template = DialogueTemplate(
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_message}]
        )
        prompt = dialogue_template.get_inference_prompt()
    else:
        dialogue_template = DialogueTemplate(
            system="", messages=past_messages + [{"role": "user", "content": user_message}]
        )
        prompt = dialogue_template.get_inference_prompt()

    generate_kwargs = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    # Get the model and tokenizer, and tokenize the user text.
    model_inputs = tokenizer([prompt], return_tensors="pt").to(torch_device)

    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
    # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
    streamer = TextIteratorStreamer(tokenizer, timeout=5.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Pull the generated text from the streamer, and update the model output.
    output = ""
    for idx, new_text in enumerate(streamer):
        output += new_text
        if idx == 0:
            history.append(output)
        else:
            history[-1] = output

        chat = [
            (wrap_html_code(history[i].strip()), wrap_html_code(history[i + 1].strip()))
            for i in range(0, len(history) - 1, 2)
        ]

        # chat = [(history[i].strip(), history[i + 1].strip()) for i in range(0, len(history) - 1, 2)]

        yield chat, history, user_message, ""

    return chat, history, user_message, ""


examples = [
    "How can I write a Python function to generate the nth Fibonacci number?",
    "How do I get the current date using shell commands? Explain how it works.",
    "What's the meaning of life?",
    "Write a function in Javascript to reverse words in a given string.",
    "Give the following data {'Name':['Tom', 'Brad', 'Kyle', 'Jerry'], 'Age':[20, 21, 19, 18], 'Height' : [6.1, 5.9, 6.0, 6.1]}. Can you plot one graph with two subplots as columns. The first is a bar graph showing the height of each person. The second is a bargraph showing the age of each person? Draw the graph in seaborn talk mode.",
    "Create a regex to extract dates from logs",
    "How to decode JSON into a typescript object",
    "Write a list into a jsonlines file and save locally",
]


def clear_chat():
    return [], []


def delete_last_turn(chat, history):
    if chat and history:
        chat.pop(-1)
        history.pop(-1)
        history.pop(-1)
    return chat, history


def process_example(args):
    for [x, y] in run_generate(args):
        pass
    return [x, y]


# Regenerate response
def retry_last_answer(
    user_message,
    chat,
    history,
    temperature,
    top_k,
    top_p,
    max_new_tokens,
    repetition_penalty,
):
    if chat and history:
        # Removing the previous conversation from chat
        chat.pop(-1)
        # Removing bot response from the history
        history.pop(-1)
        # Setting up a flag to capture a retry
        RETRY_FLAG = True
        # Getting last message from user
        user_message = history[-1]

    yield from run_generate(
        RETRY_FLAG,
        user_message,
        chat,
        history,
        temperature,
        top_k,
        top_p,
        max_new_tokens,
        repetition_penalty,
    )


title = """<h1 align="center">PEFT StarCodePlus Chat Playground 💬</h1>"""
custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""

with gr.Blocks(analytics_enabled=False, css=custom_css) as demo:
    gr.HTML(title)

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
            ⚠️ **Intended Use**: this app is provided as educational tools to explain large language model fine-tuning; not to serve as replacement for human expertise.

            ⚠️ **Known Failure Modes**: the model used has not been aligned to human preferences with techniques like RLHF, so it can produce problematic outputs (especially when prompted to do so). Since the base model was pretrained on a large corpus of code, it may produce code snippets that are syntactically valid but semantically incorrect.  For example, it may produce code that does not compile or that produces incorrect results.  It may also produce code that is vulnerable to security exploits.  We have observed the model also has a tendency to produce false URLs which should be carefully inspected before clicking. For more details on the model's limitations in terms of factuality and biases.

            """
            )
    with gr.Row():
        with gr.Box():
            output = gr.Markdown()
            chatbot = gr.Chatbot(elem_id="chat-message", label="Chat")

    with gr.Row():
        with gr.Column(scale=3):
            user_message = gr.Textbox(placeholder="Enter your message here", show_label=False, elem_id="q-input")
            with gr.Row():
                send_button = gr.Button("Send", elem_id="send-btn", visible=True)

                regenerate_button = gr.Button("Regenerate", elem_id="retry-btn", visible=True)

                delete_turn_button = gr.Button("Delete last turn", elem_id="delete-btn", visible=True)

                clear_chat_button = gr.Button("Clear chat", elem_id="clear-btn", visible=True)

            with gr.Accordion(label="Parameters", open=False, elem_id="parameters-accordion"):
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.6,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    interactive=True,
                    info="Higher values produce more diverse outputs",
                )
                top_k = gr.Slider(
                    label="Top-k",
                    value=50,
                    minimum=0.0,
                    maximum=100,
                    step=1,
                    interactive=True,
                    info="Sample from a shortlist of top-k tokens",
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    value=0.95,
                    minimum=0.0,
                    maximum=1,
                    step=0.05,
                    interactive=True,
                    info="Higher values sample more low-probability tokens",
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=512,
                    minimum=0,
                    maximum=1024,
                    step=4,
                    interactive=True,
                    info="The maximum numbers of new tokens",
                )
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    value=1.2,
                    minimum=0.0,
                    maximum=10,
                    step=0.1,
                    interactive=True,
                    info="The parameter for repetition penalty. 1.0 means no penalty.",
                )
            with gr.Row():
                gr.Examples(
                    examples=examples,
                    inputs=[user_message],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )

    history = gr.State([])
    RETRY_FLAG = gr.Checkbox(value=False, visible=False)

    # To clear out "message" input textbox and use this to regenerate message
    last_user_message = gr.State("")

    user_message.submit(
        run_generate,
        inputs=[
            RETRY_FLAG,
            user_message,
            chatbot,
            history,
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            repetition_penalty,
        ],
        outputs=[chatbot, history, last_user_message, user_message],
    )

    send_button.click(
        run_generate,
        inputs=[
            RETRY_FLAG,
            user_message,
            chatbot,
            history,
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            repetition_penalty,
        ],
        outputs=[chatbot, history, last_user_message, user_message],
    )

    regenerate_button.click(
        retry_last_answer,
        inputs=[
            user_message,
            chatbot,
            history,
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            repetition_penalty,
        ],
        outputs=[chatbot, history, last_user_message, user_message],
    )

    delete_turn_button.click(delete_last_turn, [chatbot, history], [chatbot, history])
    clear_chat_button.click(clear_chat, outputs=[chatbot, history])

    demo.queue(concurrency_count=16).launch(debug=True, share=True)
