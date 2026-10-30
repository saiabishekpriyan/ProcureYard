"""Application configuration — reads from environment / .env file."""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg2://qd:qd_secret@localhost:5432/qddb"
    payswift_base_url: str = "http://localhost:8081"
    gemini_api_key: str = ""
    # Gemini model name. Override with GEMINI_MODEL env var.
    # gemini-3.8-flash verified available 2026-10-29 via genai.list_models().
    gemini_model: str = "gemini-3.8-flash"

    riders_csv: str = "/data/riders.csv"
    trips_csv: str = "/data/trips.csv"
    payout_lines_csv: str = "/data/payout_lines.csv"
    conversations_json: str = "/data/conversations.json"

    payswift_max_retries: int = 3
    payswift_timeout: int = 10
    # Gemini API call timeout in seconds (per send_message call, not total).
    # Set via GEMINI_TIMEOUT env var. 45 s gives the model time to call tools
    # and still respond, without hanging the HTTP request indefinitely.
    gemini_timeout: int = 45


settings = Settings()
