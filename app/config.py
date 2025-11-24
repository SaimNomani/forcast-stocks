from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Database Configuration
    database_url: str = "postgresql://user:password@host:port/dbname"

    # Stock Lists
    global_stocks: List[str] = ["AAPL", "GOOGL", "MSFT"]
    local_stocks: List[str] = ["OGDC.KA", "LUCK.KA", "HUBC.KA"]
    
    # Scheduler Settings
    fetch_interval_seconds: int = 900  # 15 minutes
    local_update_interval_hours: int = 24

    # Model Settings
    retrain_threshold: float = 0.1

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Custom sources to handle comma-separated lists in env vars."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    def _init_(self, **kwargs):
        super()._init_(**kwargs)
        # Parse comma-separated strings into lists
        if isinstance(self.global_stocks, str):
            self.global_stocks = [s.strip() for s in self.global_stocks.split(",")]
        if isinstance(self.local_stocks, str):
            self.local_stocks = [s.strip() for s in self.local_stocks.split(",")]


settings=Settings()