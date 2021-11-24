import setup_env
import subprocess
import os

subprocess.check_call(
    ['python3', 'gesture_recognization.py'], env=dict(os.environ))
