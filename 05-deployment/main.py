import pickle

# Step 1: Load the pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# Step 2: Input record
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Step 3: Make prediction (probability)
X = [record]
pred = model.predict_proba(X)[0, 1]

print("Conversion probability:", pred)
