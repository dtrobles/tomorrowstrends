# db.py
from sqlalchemy import create_engine, Column, Integer, String, Date, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
# Update DATABASE_URL with your PostgreSQL credentials and database name
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CountryData(Base):
    __tablename__ = "country_data"
    id = Column(Integer, primary_key=True, index=True)
    country_code = Column(String(10), index=True)
    date = Column(Date)
    term = Column(String)
    value = Column(Float)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    country_code = Column(String(10), index=True)
    horizon = Column(String)  # e.g. 'tomorrow', '3_days', '5_days'
    term = Column(String)

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    country_code = Column(String(10), unique=True, index=True)
    model_blob = Column(LargeBinary)  # Stores the pickled model

def init_db():
    Base.metadata.create_all(bind=engine)
