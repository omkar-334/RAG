import os
import subprocess
import time
from pathlib import Path
import requests

import modal
from modal import App, Image, Mount, asgi_app, build, enter, method, wsgi_app

MODEL = os.environ.get("MODEL", "llama3")


def pull(model: str = MODEL):
    # subprocess.run(["systemctl", "daemon-reload"])
    # subprocess.run(["systemctl", "enable", "ollama"])
    # subprocess.run(["systemctl", "start", "ollama"])
    time.sleep(2)
    subprocess.run(["ollama", "run", model], stdout=subprocess.PIPE)


image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands(  # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &"
        # "curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama",
        # "chmod +x /usr/bin/ollama",
        # "useradd -r -s /bin/false -m -d /usr/share/ollama ollama",
    )
    # .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install("ollama")
    .pip_install("flask")
    .pip_install("openai")
    .run_function(pull)
)
app = modal.App(name="fastapi", image=image)
with image.imports():
    import ollama
    from openai import OpenAI


@app.cls(gpu="a10g", container_idle_timeout=300)
class Ollama:
    @build()
    def pull(self): ...

    @enter()
    def load(self):
        subprocess.run(["systemctl", "start", "ollama"])
    

    @method()
    def infer(self, text: str):
        stream = ollama.chat(
            model=MODEL, messages=[{"role": "user", "content": text}], stream=True
        )
        for chunk in stream:
            yield chunk["message"]["content"]
            print(chunk["message"]["content"], end="", flush=True)

    @wsgi_app()
    def f(self):
        import ollama
        from flask import Flask, request
        from openai import OpenAI

        options = dict(
            num_keep=5,
            seed=42,
            num_predict=100,
            top_k=20,
            top_p=0.9,
            tfs_z=0.5,
            typical_p=0.7,
            repeat_last_n=33,
            temperature=0.8,
            repeat_penalty=1.2,
            presence_penalty=1.5,
            frequency_penalty=1.0,
            mirostat=1,
            mirostat_tau=0.8,
            mirostat_eta=0.6,
            penalize_newline=True,
            stop=["\n", "user:"],
            numa=False,
            num_ctx=1024,
            num_batch=2,
            num_gpu=1,
            main_gpu=0,
            low_vram=False,
            f16_kv=True,
            vocab_only=False,
            use_mmap=True,
            use_mlock=False,
            num_thread=8,
        )

        web_app = Flask(__name__)

        @web_app.get("/")
        def home():
            return "Hello Flask World!"

        @web_app.route("/api/chat", methods=["POST"])
        async def ollama_chat():
            output = request.json
            try:
                messages = output["messages"]
            except Exception:
                return "INPUT Error - 'messages' not found in request."

            values_option = output.get("options", {})
            for i in values_option:
                options[i] = values_option[i]
            ollama.
            try:
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    stream=False,
                    options=options,
                )
            except Exception as e:
                return "Error - " + str(e)[:100]

            # uncomment this line if only conversation is to be returned and no other paramters
            # return_output = [{'role': 'user', 'content': content},chat_output['message']]
            return response

        @web_app.route("/api/generate", methods=["POST"])
        async def ollama_generate():
            output = request.json
            try:
                prompt = output["prompt"]
            except Exception:
                return "INPUT Error - 'prompt' not found in request."

            values_option = output.get("options", {})
            for i in values_option:
                options[i] = values_option[i]
            try:
                response = ollama.generate(
                    model=MODEL, prompt=prompt, stream=False, options=options
                )
            except Exception as e:
                return "Error - " + str(e)[:100]

            return response

        @web_app.route("/v1/chat/completions", methods=["POST"])
        async def openai_chat():
            output = request.json
            try:
                messages = output["messages"]
            except Exception:
                return "INPUT Error - 'messages' not found in request."
            try:
                client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                )
                response = client.chat.completions.create(
                    model=MODEL, messages=messages
                )
            except Exception as e:
                return "Error - " + str(e)[:100]
            return response
    
        @web_app.route("/test", methods=["POST"])
        async def test():
            output = request.json
            messages = output["prompt"]

            response = requests.post("http://localhost:11434/v1")
            return response.content

        return web_app