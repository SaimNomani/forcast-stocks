from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Target PSX Stocks
    symbols: List[str] = ["OGDC.KA", "LUCK.KA", "HUBC.KA"]
    
    # 24 Hours in seconds (Daily Update)
    fetch_interval: int = 86400 
    retrain_threshold: float = 0.1

    class Config:
        env_file = ".env"
        case_sensitive = False

    @classmethod
    def customise_sources(cls, init_settings, env_settings, file_secret_settings):
        from pydantic import Field
        from pydantic_settings.sources import EnvSettingsSource

        class CustomEnvSource(EnvSettingsSource):
            def get_field_value(self, field: Field, field_name: str):
                value, source = super().get_field_value(field, field_name)
                if field_name == "symbols" and isinstance(value, str):
                    value = [s.strip() for s in value.split(",")]
                return value, source

        return (
            init_settings,
            CustomEnvSource,
            file_secret_settings,
        )
    
settings = Settings()