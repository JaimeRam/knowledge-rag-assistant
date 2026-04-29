from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Optional
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class DigimonMetadata(Base):
    """Store structured metadata about Digimons."""
    __tablename__ = "digimon_metadata"
    
    id = Column(Integer, primary_key=True)
    dapi_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    level = Column(String(100))
    type = Column(String(100))
    attribute = Column(String(100))
    description = Column(Text)
    image_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConversationLog(Base):
    """Log conversation queries and responses for analytics."""
    __tablename__ = "conversation_logs"
    
    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    digimon_ids = Column(Text)  # JSON array of digimon IDs referenced
    latency_ms = Column(Integer)
    token_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class PostgresManager:
    def __init__(self):
        settings = get_settings()
        self.engine = create_engine(
            f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Created PostgreSQL tables")
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def close(self):
        """Close the engine."""
        self.engine.dispose()
