import os
from threading import Thread
from typing import Iterator

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

DESCRIPTION = """\
# DeepSeek-6.7B-Chat (CPU Version)

This Space demonstrates model [DeepSeek-Coder](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) by DeepSeek.
**Running on CPU - Expect slower response times**
"""

# Load model with CPU optimizations
model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    device_map="auto",           # Let accelerate handle device placement
    low_cpu_mem_usage=True       # Reduce memory footprint
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.use_default_system_prompt = False

def generate(
    message: str,
    chat_history: list,
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        print(f"Trimmed input to {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=300.0, skip_prompt=True, skip_special_tokens=True)  # Increased timeout for CPU
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id
    )
    
    try:
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
    except Exception as e:
        yield f"Error: {str(e)}"
        return

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs).replace("回答道", "")

chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0,
            maximum=1.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Implement a simple calculator in Python"],
        ["Explain recursion with a simple example"],
        ["Write a Python function to reverse a string"],
    ],
)

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)