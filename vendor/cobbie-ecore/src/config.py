import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from typing import Literal

load_dotenv(find_dotenv())

# Path. A fresh clone must be importable without a workstation-specific .env;
# callers can still override this explicitly for legacy layouts.
ROOT_PATH = os.environ.get("ROOT_PATH", str(Path(__file__).resolve().parents[1]))
SRC_PATH = os.path.join(ROOT_PATH, "src")
TEST_IFC_PATH = os.path.join(ROOT_PATH, "src/db/bim_models/duplex/arc.ifc")
# Public source releases intentionally omit the protected benchmark database.
# Paper/evaluation runners may bind a separately verified, read-only/relocated
# database without overwriting the ignored repository placeholder.
DB_PATH = os.environ.get("COBBIE_DB_PATH", os.path.join(ROOT_PATH, "src/db/db.db"))
DIRECTORY_IFC_MODELS_PATH = os.path.join(ROOT_PATH, "src/db/bim_models")
CREATED_TOOLS_PATH = os.path.join(ROOT_PATH, "src/tools/created")
INITIAL_TOOLS_PATH = os.path.join(ROOT_PATH, "src/tools/initial")
MANUAL_TOOLS_PATH = os.path.join(ROOT_PATH, "src/tools/manual")

# URI
MLFLOW_URI = "http://127.0.0.1:5000"

# Credentials are optional at import time so offline verification and package
# inspection remain credential-free. Paid/live entry points validate the
# provider-specific key before making a request.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# Boilerplate code for the toolmaker
FUNCTION_BOILERPLATE = """
import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.shape
import ifcopenshell.util.placement
import ifcopenshell.util.geolocation
import ifcopenshell.util.system
import ifcopenshell.geom
import math
import json
from typing import *
"""

# Configure logger
LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
