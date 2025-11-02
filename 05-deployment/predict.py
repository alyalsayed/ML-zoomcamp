from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the model (same one you trained earlier)
with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

# Define the input format
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Subscription Prediction API is running."}

@app.post("/predict")
def predict_subscription(client: Client):
    # Convert client to dict
    client_data = client.dict()
    X = dv.transform([client_data])
    y_pred = model.predict_proba(X)[0, 1]  # probability of class 1
    return {"subscription_probability": float(y_pred)}
