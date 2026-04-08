import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

print("Loading data.csv...")
df = pd.read_csv('data.csv').dropna()

# Extract features
FEATURES = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 
            'waterfront', 'view', 'condition', 'sqft_lot', 'yr_built']

X = df[FEATURES]
y = df['price']

print("Normalizing Data and Training Advanced Model (Gradient Boosting)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a robust pipeline mimicking an advanced model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(n_estimators=250, learning_rate=0.08, max_depth=5, random_state=42))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Training Complete!")
print(f"Model MAE: ${mae:,.0f}")
print(f"Model R2:  {r2:.4f}")

# Save
pickle.dump(pipeline, open('model.pkl', 'wb'))
print("Advanced pipeline saved to model.pkl")
