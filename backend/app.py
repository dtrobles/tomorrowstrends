import os
import sys
from flask import Flask, request, jsonify
from flask_graphql import GraphQLView
import graphene
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import date
import subprocess

# Add the "ML model" directory to sys.path so we can import db.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ML model')))

from db import Prediction, CountryData  # Import the Prediction and CountryData models

# Load environment variables from .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Setup SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def run_model_update(country_code: str):
    """
    Run the ML model update for the specified country.
    """
    model_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ML model', 'model.py'))
    try:
        subprocess.run(["python", model_script_path, "--country", country_code, "--update"], check=True)
        print(f"[DEBUG] Data for {country_code} updated via ML model.")
    except Exception as e:
        print(f"[DEBUG] Error updating data for {country_code}: {e}")

class TrendPrediction(graphene.ObjectType):
    tomorrow = graphene.List(graphene.String)
    threeDays = graphene.List(graphene.String)
    fiveDays = graphene.List(graphene.String)

class Query(graphene.ObjectType):
    predictions = graphene.Field(TrendPrediction, country=graphene.String())

    def resolve_predictions(self, info, country=None):
        # Mapping from frontend country names to standardized country codes
        mapping = {
            "United States of America": "US",
            "United States": "US",
            "US": "US",
            "Japan": "JP",
            "JP": "JP",
            "China": "CN",
            "CN": "CN",
            # Add more mappings as needed.
        }
        country_code = mapping.get(country, None)
        if not country_code:
            print(f"[DEBUG] No predictions available for country: {country}")
            return TrendPrediction(tomorrow=[], threeDays=[], fiveDays=[])

        session = SessionLocal()
        try:
            # Check if today's data exists; if not, trigger update.
            today = date.today()
            update_needed = session.query(CountryData).filter(
                CountryData.country_code == country_code,
                CountryData.date == today
            ).first() is None

            if update_needed:
                run_model_update(country_code)

            predictions = session.query(Prediction).filter(
                Prediction.country_code == country_code
            ).all()

            # If no predictions are available, run the ML model update and re-query.
            if not predictions:
                run_model_update(country_code)
                predictions = session.query(Prediction).filter(
                    Prediction.country_code == country_code
                ).all()

            data = {
                "tomorrow": [],
                "threeDays": [],
                "fiveDays": []
            }

            for pred in predictions:
                horizon = pred.horizon.strip().lower()
                term = pred.term.strip()
                if horizon == "tomorrow":
                    data["tomorrow"].append(term)
                elif horizon in ["3_days", "3 days"]:
                    data["threeDays"].append(term)
                elif horizon in ["5_days", "5 days"]:
                    data["fiveDays"].append(term)

            print(f"[DEBUG] Database predictions for {country} loaded:", data)
            return TrendPrediction(
                tomorrow=data["tomorrow"],
                threeDays=data["threeDays"],
                fiveDays=data["fiveDays"]
            )
        except Exception as e:
            print(f"[DEBUG] Error querying predictions for {country}: {e}")
            return TrendPrediction(tomorrow=[], threeDays=[], fiveDays=[])
        finally:
            session.close()

schema = graphene.Schema(query=Query)

app = Flask(__name__)
CORS(app)
app.debug = True

# GraphQL endpoint remains at /graphql
app.add_url_rule(
    '/graphql',
    view_func=GraphQLView.as_view(
        'graphql',
        schema=schema,
        graphiql=True  # Enables the GraphiQL UI for testing queries
    )
)

# Add a RESTful endpoint at /api/predictions using the same logic
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    country = request.args.get('country')
    mapping = {
        "United States of America": "US",
        "United States": "US",
        "US": "US",
        "Japan": "JP",
        "JP": "JP",
        "China": "CN",
        "CN": "CN",
        # Add more mappings as needed.
    }
    country_code = mapping.get(country, None)
    if not country_code:
        print(f"[DEBUG] No predictions available for country: {country}")
        return jsonify({"tomorrow": [], "threeDays": [], "fiveDays": []}), 404

    session = SessionLocal()
    try:
        # Check if today's data exists; if not, trigger update.
        today = date.today()
        update_needed = session.query(CountryData).filter(
            CountryData.country_code == country_code,
            CountryData.date == today
        ).first() is None

        if update_needed:
            run_model_update(country_code)

        predictions = session.query(Prediction).filter(
            Prediction.country_code == country_code
        ).all()

        # If no predictions are available, run the ML model update and re-query.
        if not predictions:
            run_model_update(country_code)
            predictions = session.query(Prediction).filter(
                Prediction.country_code == country_code
            ).all()

        data = {"tomorrow": [], "threeDays": [], "fiveDays": []}
        for pred in predictions:
            horizon = pred.horizon.strip().lower()
            term = pred.term.strip()
            if horizon == "tomorrow":
                data["tomorrow"].append(term)
            elif horizon in ["3_days", "3 days"]:
                data["threeDays"].append(term)
            elif horizon in ["5_days", "5 days"]:
                data["fiveDays"].append(term)
                
        print(f"[DEBUG] Database predictions for {country} loaded:", data)
        return jsonify(data)
    except Exception as e:
        print(f"[DEBUG] Error querying predictions for {country}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

if __name__ == '__main__':
    app.run(port=5000)
