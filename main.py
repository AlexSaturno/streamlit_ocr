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
        r"/home/azureuser/poc-fpo/",
        "--server.port",
        "8501",
        "--server.address",
        "51.12.49.102",
    ]
)
