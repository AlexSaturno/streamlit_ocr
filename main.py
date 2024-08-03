import subprocess
import streamlit
import os
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        r"C:\Users\alexa\Documents\Github\streamlit_ocr\01_Processamento_Padrão.py",
        "--server.port",
        "8501",
        "--server.address",
        "127.0.0.1",
    ]
)
