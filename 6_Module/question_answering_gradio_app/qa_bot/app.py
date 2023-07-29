import argparse
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import hnswlib
from typing import Iterator

import gradio as gr
import pandas as pd
import torch

from agent import get_input_token_length, run

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_TOKEN = os.environ.get("API_TOKEN", None)
ENDPOINT = os.environ.get("ENDPOINT", None)
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000
EMBED_DIM = 1024
K = 10
EF = 100
SEARCH_INDEX = "search_index.bin"
DOCUMENT_DATASET = "chunked_data.parquet"

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", torch_device)
print("CPU threads:", torch.get_num_threads())

biencoder = SentenceTransformer("intfloat/e5-large-v2", device=torch_device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device=torch_device)

# load the index for the PEFT docs
def load_hnsw_index(index_file):
    # Load the HNSW index from the specified file
    index = hnswlib.Index(space="ip", dim=EMBED_DIM)
    index.load_index(index_file)
    return index


def create_query_embedding(query):
    # Encode the query to get its embedding
    embedding = biencoder.encode([query], normalize_embeddings=True)[0]
    return embedding


def find_nearest_neighbors(query_embedding):
    search_index.set_ef(EF)
    # Find the k-nearest neighbors for the query embedding
    labels, _ = search_index.knn_query(query_embedding, k=K)
    relevant_chunks = data_df.iloc[labels[0]]["chunk_content"].tolist()
    return relevant_chunks


def rerank_chunks_with_cross_encoder(query, chunks):
    # Create a list of tuples, each containing a query-chunk pair
    pairs = [(query, chunk) for chunk in chunks]

    # Get scores for each query-chunk pair using the cross encoder
    scores = cross_encoder.predict(pairs)

    # Sort the chunks based on their scores in descending order
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]

    return sorted_chunks


def create_qa_prompt(query, relevant_chunks):
    stuffed_context = " ".join(relevant_chunks)
    return f"""\
Use the following pieces of context given in to answer the question at the end. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. \
Keep the answer short and succinct.
        
Context:<{stuffed_context}
Question:<{query}>
Helpful Answer:\
"""


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000

DESCRIPTION = """
# Llama-2 7B Chat

This Space demonstrates model [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta, a Llama 2 model with 7B parameters fine-tuned for chat instructions. Feel free to play with it, or duplicate to run generations without a queue! If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://huggingface.co/inference-endpoints).

🔎 For more details about the Llama 2 family of models and how to use them with `transformers`, take a look [at our blog post](https://huggingface.co/blog/llama2).

🔨 Looking for an even more powerful model? Check out the [13B version](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat) or the large [70B model demo](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI).
"""

LICENSE = """
<p/>

---
As a derivate work of [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/USE_POLICY.md).
"""

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return "", message


def display_input(message: str, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ""))
    return history


def delete_prev_fn(history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ""
    return history, message or ""


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    query_embedding = create_query_embedding(message)
    relevant_chunks = find_nearest_neighbors(query_embedding)
    reranked_relevant_chunks = rerank_chunks_with_cross_encoder(message, relevant_chunks)
    qa_prompt = create_qa_prompt(message, reranked_relevant_chunks)
    # print(f"{qa_prompt=}")

    history = history_with_input[:-1]
    generator = run(qa_prompt, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, "")]
    for response in generator:
        yield history + [(message, response)]


def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 0.2, 0.95, 50)
    for x in generator:
        pass
    return "", x


def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    input_token_length = get_input_token_length(message, chat_history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(
            f"The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again."
        )


search_index = load_hnsw_index(SEARCH_INDEX)
data_df = pd.read_parquet(DOCUMENT_DATASET).reset_index()
with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")

    with gr.Group():
        chatbot = gr.Chatbot(label="Chatbot")
        with gr.Row():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder="Type a message...",
                scale=10,
            )
            submit_button = gr.Button("Submit", variant="primary", scale=1, min_width=0)
    with gr.Row():
        retry_button = gr.Button("🔄  Retry", variant="secondary")
        undo_button = gr.Button("↩️ Undo", variant="secondary")
        clear_button = gr.Button("🗑️  Clear", variant="secondary")

    saved_input = gr.State()

    with gr.Accordion(label="Advanced options", open=False):
        system_prompt = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=6)
        max_new_tokens = gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )
        temperature = gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.2,
        )
        top_p = gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        )
        top_k = gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        )

    gr.Examples(
        examples=[
            "What is 🤗 PEFT?",
            "How do I create a LoraConfig using 🤗 PEFT?",
            "What are the different prompt learning methods supported in PEFT?",
            "How do I use DeepSpeed with 🤗 PEFT?",
            "How do I combine multiple LoRA adapters?",
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        # fn=process_example,
        cache_examples=True,
    )

    gr.Markdown(LICENSE)

    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(fn=display_input, inputs=[saved_input, chatbot], outputs=chatbot, api_name=False, queue=False,).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = (
        submit_button.click(
            fn=clear_and_save_textbox,
            inputs=textbox,
            outputs=[textbox, saved_input],
            api_name=False,
            queue=False,
        )
        .then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        )
        .then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        )
        .success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(fn=display_input, inputs=[saved_input, chatbot], outputs=chatbot, api_name=False, queue=False,).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )

demo.queue(max_size=20).launch(debug=True, share=True)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Script to create and use an HNSW index for similarity search.")
#     parser.add_argument("--input_file", help="Input file containing text chunks in a Parquet format")
#     parser.add_argument("--index_file", help="HNSW index file with .bin extension")
#     args = parser.parse_args()

#     data_df = pd.read_parquet(args.input_file).reset_index()
#     search_index = load_hnsw_index(args.index_file)
#     main()
