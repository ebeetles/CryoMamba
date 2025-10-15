import os
from typing import Optional

class Settings:
    """Application settings"""
    def __init__(self):
        self.app_name: str = os.getenv("APP_NAME", "CryoMamba Inference Server")
        self.app_version: str = os.getenv("APP_VERSION", "1.0.0")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        
        # API settings
        self.api_v1_prefix: str = os.getenv("API_V1_PREFIX", "/v1")
        
        # Logging settings
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Development settings
        cors_origins_str = os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]')
        try:
            import json
            self.cors_origins = json.loads(cors_origins_str) if cors_origins_str.startswith('[') else ["*"]
        except (json.JSONDecodeError, ValueError):
            self.cors_origins = ["*"]

        # Upload settings
        self.upload_base_dir: str = os.getenv("UPLOAD_BASE_DIR", "/tmp/cryomamba_uploads")
        self.max_upload_size_gb: int = int(os.getenv("MAX_UPLOAD_SIZE_GB", "20"))
        self.max_chunk_size_mb: int = int(os.getenv("MAX_CHUNK_SIZE_MB", "64"))

        # nnU-Net settings
        self.nnunet_model_dir: Optional[str] = os.getenv(
            "NNUNET_MODEL_DIR",
            "/Users/elwinli/Downloads/pretrained_weights/nnUNetTrainer__nnUNetPlans__3d_fullres",
        )
        self.nnunet_device: str = os.getenv("NNUNET_DEVICE", "cuda")

# Global settings instance
settings = Settings()
