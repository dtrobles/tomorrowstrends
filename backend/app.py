import os
import csv
from flask import Flask
from flask_graphql import GraphQLView
import graphene
from flask_cors import CORS

class TrendPrediction(graphene.ObjectType):
    tomorrow = graphene.List(graphene.String)
    threeDays = graphene.List(graphene.String)
    fiveDays = graphene.List(graphene.String)

class Query(graphene.ObjectType):
    predictions = graphene.Field(TrendPrediction, country=graphene.String())

    def resolve_predictions(self, info, country=None):
        # Mapping from frontend country names to folder codes
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
        folder = mapping.get(country, None)
        if not folder:
            print(f"[DEBUG] No predictions available for country: {country}")
            return TrendPrediction(tomorrow=[], threeDays=[], fiveDays=[])
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        predictions_file = os.path.join(base_dir, 'ML model', 'data', folder, 'predictions.csv')
        data = {
            "tomorrow": [],
            "threeDays": [],
            "fiveDays": []
        }
        try:
            print(f"[DEBUG] Reading predictions for {country} from: {predictions_file}")
            with open(predictions_file, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    horizon = row['horizon'].strip().lower()
                    term = row['term'].strip()
                    if horizon == "tomorrow":
                        data["tomorrow"].append(term)
                    elif horizon in ["3_days", "3 days"]:
                        data["threeDays"].append(term)
                    elif horizon in ["5_days", "5 days"]:
                        data["fiveDays"].append(term)
            print(f"[DEBUG] CSV predictions for {country} loaded:", data)
            return TrendPrediction(
                tomorrow=data["tomorrow"],
                threeDays=data["threeDays"],
                fiveDays=data["fiveDays"]
            )
        except Exception as e:
            print(f"[DEBUG] Error reading predictions for {country}: {e}")
            return TrendPrediction(tomorrow=[], threeDays=[], fiveDays=[])

schema = graphene.Schema(query=Query)

app = Flask(__name__)
CORS(app)  # Enable CORS
app.debug = True

app.add_url_rule(
    '/graphql',
    view_func=GraphQLView.as_view(
        'graphql',
        schema=schema,
        graphiql=True  # GraphiQL for testing queries
    )
)

if __name__ == '__main__':
    app.run(port=5000)
