# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Gmail
    gmail_credentials_path: str
    gmail_token_path: str
    gmail_check_interval: int = 60
    
    # Slack
    slack_bot_token: str
    slack_channel_id: str
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()