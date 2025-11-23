from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Database Configuration
    database_url: str = "postgresql://user:password@host:port/dbname" # Default, should be overridden by env

    # Stock Lists
    global_stocks: List[str] = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"] # Example global stocks
    local_stocks: List[str] = ["OGDC.KA", "LUCK.KA", "HUBC.KA"] # Example local stocks
    
    # Scheduler Settings
    fetch_interval_seconds: int = 900 # 15 minutes
    local_update_interval_hours: int = 24

    # Model Settings
    retrain_threshold: float = 0.1

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore" # Allow extra fields in .env

    @classmethod
    def customise_sources(cls, init_settings, env_settings, file_secret_settings):
        from pydantic import Field
        from pydantic_settings.sources import EnvSettingsSource

        class CustomEnvSource(EnvSettingsSource):
            def get_field_value(self, field: Field, field_name: str):
                value, source = super().get_field_value(field, field_name)
                if field_name in ["global_stocks", "local_stocks"] and isinstance(value, str):
                    value = [s.strip() for s in value.split(",")]
                return value, source

        return (
            init_settings,
            CustomEnvSource,
            file_secret_settings,
        )
    
settings = Settings()