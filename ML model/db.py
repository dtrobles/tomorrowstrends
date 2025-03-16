from sqlalchemy import create_engine, Column, Integer, String, Date, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from the .env file
load_dotenv()

# Get DATABASE_URL from the environment
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

def update_country_data(session, data):
    """
    Update an existing CountryData record or insert a new one.
    
    data: dict with keys 'country_code', 'date', 'term', and 'value'.
          The 'date' can be a datetime.date object or a string in 'YYYY-MM-DD' format.
    """
    # Convert date string to date object if necessary
    if isinstance(data.get("date"), str):
        data["date"] = datetime.strptime(data["date"], "%Y-%m-%d").date()
    
    # Check for an existing record
    instance = session.query(CountryData).filter_by(
        country_code=data.get("country_code"),
        date=data.get("date"),
        term=data.get("term")
    ).first()

    if instance:
        # Update the record if found
        instance.value = data.get("value")
    else:
        # Create a new record if not found
        instance = CountryData(
            country_code=data.get("country_code"),
            date=data.get("date"),
            term=data.get("term"),
            value=data.get("value")
        )
        session.add(instance)
    
    session.commit()

# Example usage:
if __name__ == "__main__":
    # Initialize the database and create tables if they don't exist
    init_db()

    # Create a session
    session = SessionLocal()
    try:
        # Example data to update or insert
        sample_data = {
            "country_code": "US",
            "date": "2023-10-15",  # You can also provide a datetime.date object
            "term": "GDP",
            "value": 21.43
        }
        update_country_data(session, sample_data)
        print("Country data updated successfully!")
    finally:
        session.close()
