"""Entry point: uvicorn app.main:app"""

import logging

from dotenv import load_dotenv

from app.api import create_app

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()
