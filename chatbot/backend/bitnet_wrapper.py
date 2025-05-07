import subprocess
import os
import platform

class BitNetCppInterface:
    def __init__(self, model_path="./models/ggml-model-i2_s.gguf", binary_path="../bitnet/BitNet/build/bin/llama-cli"):
        self.model_path = model_path
        if platform.system() == "Windows":
            self.binary_path = os.path.join("build", "bin", "Release", "llama-cli.exe")
        else:
            self.binary_path = binary_path

    def generate(self, prompt, n_predict=128, threads=2, ctx_size=2048, temperature=0.8, conversation=False):
        command = [
            self.binary_path,
            "-m", self.model_path,
            "-n", str(n_predict),
            "-t", str(threads),
            "-p", prompt,
            "-ngl", "0",
            "-c", str(ctx_size),
            "--temp", str(temperature),
            "-b", "1"
        ]
        if conversation:
            command.append("-cnv")

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"BitNet error: {result.stderr}")
            return result.stdout.strip()
        except Exception as e:
            return f"BitNet subprocess error: {str(e)}"
