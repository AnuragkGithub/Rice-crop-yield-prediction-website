
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("rice_data.csv")
df = df.dropna(subset=['Production', 'Area'])

le_state = LabelEncoder()
le_district = LabelEncoder()
le_season = LabelEncoder()

df['State_Name'] = le_state.fit_transform(df['State_Name'])
df['District_Name'] = le_district.fit_transform(df['District_Name'])
df['Season'] = le_season.fit_transform(df['Season'])

features = ['State_Name', 'District_Name', 'Crop_Year', 'Season',
            'Temperature', 'Humidity', 'Soil_Moisture', 'Area']
X = df[features]
y = df['Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("rice_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
# ... (previous code)

# Save model and encoders
with open("rice_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump({
        "state": le_state,
        "district": le_district,
        "season": le_season
    }, f)

print("✅ Model and encoders saved")

