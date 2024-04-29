from pydantic_settings import BaseSettings

class BaseSettingsWrapper(BaseSettings):

    class Config:
        env_file = ".env"