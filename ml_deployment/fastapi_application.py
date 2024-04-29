import pandas as pd

from fastapi import FastAPI
from mongoengine import connect

from models.pydantic_models import InputFeatures
from config.v1.database_config import mongo_config
from models.mongo_data import InputFeaturesDocument
from utils.v1.file_loader import model, scaler, pca_ds, K_elbow
from utils.v1.preprocessing_features import helper_scale_input_features

# Define FastAPI app
app = FastAPI()

# Endpoint to handle predictions
@app.post("/predict/")
async def predict(features: InputFeatures):

    scaled_inputs = helper_scale_input_features([
        features.Education, features.Income, features.Kidhome, features.Teenhome, 
        features.Recency, features.Wines, features.Fruits, features.Meat, features.Fish, 
        features.Sweets, features.Gold, features.NumDealsPurchases, features.NumWebPurchases, 
        features.NumCatalogPurchases, features.NumStorePurchases, features.NumWebVisitsMonth, 
        features.Customer_Days, features.Age, features.Money_Spent, features.Living_With, 
        features.Children, features.Family_Size, features.Is_Parent
    ], scaler)

    scaled_inputs_pca = pd.DataFrame(pca_ds.transform(scaled_inputs), columns=(["col1","col2", "col3"]))

    scaled_inputs_pca = K_elbow.fit(scaled_inputs_pca)

    clusters = model.fit_predict(scaled_inputs_pca)

    cluster_labels = ["cl1", "cl2", "cl3", "cl4"]

    prediction = cluster_labels[clusters[0]]

    # Save input features to MongoDB
    input_features_document = InputFeaturesDocument(
        Education=features.Education,
        Income=features.Income,
        Kidhome=features.Kidhome,
        Teenhome=features.Teenhome,
        Recency=features.Recency,
        Wines=features.Wines,
        Fruits=features.Fruits,
        Meat=features.Meat,
        Fish=features.Fish,
        Sweets=features.Sweets,
        Gold=features.Gold,
        NumDealsPurchases=features.NumDealsPurchases,
        NumWebPurchases=features.NumWebPurchases,
        NumCatalogPurchases=features.NumCatalogPurchases,
        NumStorePurchases=features.NumStorePurchases,
        NumWebVisitsMonth=features.NumWebVisitsMonth,
        Customer_Days=features.Customer_Days,
        Age=features.Age,
        Money_Spent=features.Money_Spent,
        Living_With=features.Living_With,
        Children=features.Children,
        Family_Size=features.Family_Size,
        Is_Parent=features.Is_Parent,
        prediction=prediction,
    )
    input_features_document.save()

    return {'prediction': prediction}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Define MongoDB connection settings
port = 27017
host = "localhost"
db_name = "Customer_personality_prediction_database"

# Connect to MongoDB
connect(db=db_name, host=f'mongodb://{mongo_config.mongo_host}:{port}')