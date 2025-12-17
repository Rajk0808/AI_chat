from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Interaction(Base):
    """Store all user interactions"""
    __tablename__ = "interactions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    pet_id = Column(String, index=True)
    session_id = Column(String)
    timestamp = Column(DateTime, index=True)
    
    query = Column(String)
    response = Column(String)
    citations = Column(JSON)
    
    module = Column(String, index=True)
    model_used = Column(String)
    rag_used = Column(Boolean)
    fine_tuned_model = Column(Boolean)
    
    confidence_score = Column(Float)
    response_quality = Column(JSON)
    
    cost_usd = Column(Float)
    tokens_generated = Column(Integer)
    tokens_prompt = Column(Integer)
    
    timing = Column(JSON)
    success = Column(Boolean)
    errors = Column(JSON)
    
    feedback_rating = Column(Integer, default=None)  # 1-5 stars
    feedback_comment = Column(String, default=None)


class FineTuningJob(Base):
    """Track fine-tuning jobs"""
    __tablename__ = "fine_tuning_jobs"
    
    id = Column(String, primary_key=True)
    openai_job_id = Column(String, unique=True)
    
    created_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    status = Column(String)  # queued, running, succeeded, failed
    model_id = Column(String)  # Fine-tuned model ID
    
    examples_count = Column(Integer)
    training_file_id = Column(String)
    
    performance_score = Column(Float, default=0.0)
    is_deployed = Column(Boolean, default=False)
    
    metadata = Column(JSON)


class AccumulatedExample(Base):
    """Store accumulated training examples"""
    __tablename__ = "accumulated_examples"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime)
    
    user_query = Column(String)
    ai_response = Column(String)
    user_rating = Column(Integer)  # 1-5 stars
    module = Column(String)
    
    pet_id = Column(String)
    user_id = Column(String)
    feedback = Column(String)


class Model(Base):
    """Track available models"""
    __tablename__ = "models"
    
    id = Column(String, primary_key=True)
    name = Column(String)
    type = Column(String)  # base, fine-tuned
    status = Column(String)  # active, inactive, deprecated
    
    performance_score = Column(Float)
    accuracy = Column(Float)
    cost_per_token = Column(Float)
    
    created_at = Column(DateTime)
    deployed_at = Column(DateTime)


class FinetuningBudget(Base):
    """Track fine-tuning budget"""
    __tablename__ = "fine_tuning_budget"
    
    id = Column(String, primary_key=True)
    month = Column(String)
    total_budget = Column(Float)
    spent = Column(Float)
    remaining = Column(Float)
    updated_at = Column(DateTime)