import subprocess
import streamlit
import os
import sys

subprocess.run([sys.executable, "-m", "streamlit", "run", r"C:\Projetos\Asimov Academy\Projetos\POCs banco\Projetos\FPO\FPO\02_staging\UI\01_Processamento_Padrão.py", "--server.port", "8100", "--server.address", "127.0.0.1"])
