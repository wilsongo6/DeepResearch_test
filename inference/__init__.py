"""
Initialize the inference package by loading environment variables from a local
.env file. This ensures tools that rely on SUMMARY_* and other secrets see the
expected values even when the shell environment hasn't been explicitly sourced.
"""

from dotenv import load_dotenv

# Load variables defined in .env into os.environ once when the package loads.
load_dotenv()
