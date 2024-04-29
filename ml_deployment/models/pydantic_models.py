from pydantic import BaseModel

# Define request body model
class InputFeatures(BaseModel):
    Education: float
    Income: float
    Kidhome: float
    Teenhome: float
    Recency: float
    Wines: float
    Fruits: float
    Meat: float
    Fish: float
    Sweets: float
    Gold: float
    NumDealsPurchases: float
    NumWebPurchases: float
    NumCatalogPurchases: float
    NumStorePurchases: float
    NumWebVisitsMonth: float
    Customer_Days: float
    Age: float
    Money_Spent: float
    Living_With: float
    Children: float
    Family_Size: float
    Is_Parent: float