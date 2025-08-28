from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# --- Database Configuration ---
# For simplicity, using SQLite. In production, consider PostgreSQL or MySQL.
DATABASE_URL = "sqlite:///./trash_detection.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    surname = Column(String)
    aadhar_id = Column(String, unique=True, index=True)
    face_id = Column(String, unique=True, index=True) # Unique ID for face recognition
    points = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    registered_at = Column(DateTime, default=datetime.datetime.now)

class DisposalRecord(Base):
    __tablename__ = "disposal_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True) # Foreign key to User
    cctv_location = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    trash_type = Column(String)
    disposed_properly = Column(Boolean)
    points_awarded = Column(Integer)
    footage_url = Column(String) # URL to short disposal footage

class IssueReport(Base):
    __tablename__ = "issue_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True) # Foreign key to User
    issue_type = Column(String) # e.g., webapp, points_not_allotted, points_deducted_wrongly
    description = Column(String)
    reported_at = Column(DateTime, default=datetime.datetime.now)
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(String, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

# --- Database Initialization ---

def create_db_and_tables():
    Base.metadata.create_all(engine)
    print("Database tables created or already exist.")

if __name__ == "__main__":
    create_db_and_tables()
