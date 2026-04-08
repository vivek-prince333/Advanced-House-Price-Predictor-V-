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

# Save Analytics Json
print("Generating Analytics Dashboard Data...")

# 1. Feature Importance
importance = pipeline.named_steps['regressor'].feature_importances_
feat_imp = pd.DataFrame({'feature': FEATURES, 'importance': importance}).sort_values('importance', ascending=False)
feature_importance = feat_imp.to_dict('records')

# 2. Prediction vs Actual (sample 50 random)
sample_idx = range(min(50, len(y_test)))
pred_actual = pd.DataFrame({'actual': y_test.iloc[sample_idx].values, 'predicted': y_pred[sample_idx]}).to_dict('records')

# 3. Price vs Area (sample 100 random)
sample_df = df.sample(min(100, len(df)))
price_vs_area = [{'sqft_living': float(row['sqft_living']), 'price': float(row['price'])} for _, row in sample_df.iterrows()]

# 4. Top 15 Cities by Average Price
city_price = df.groupby('city')['price'].mean().sort_values(ascending=False).head(15)
city_vs_price = [{'city': str(city), 'avg_price': float(price)} for city, price in city_price.items()]

# 5. Price Trend over Years Built (Average price per year built, sampled every 2 years or so to smooth it)
yr_price = df.groupby('yr_built')['price'].mean().sort_index()
# smooth it slightly using rolling mean to look nice, then drop na
yr_price = yr_price.rolling(3, min_periods=1).mean()
year_vs_price = [{'yr_built': int(yr), 'avg_price': float(price)} for yr, price in yr_price.items()]

analytics_payload = {
    'feature_importance': feature_importance,
    'pred_actual': pred_actual,
    'price_vs_area': price_vs_area,
    'city_vs_price': city_vs_price,
    'year_vs_price': year_vs_price
}

import json
with open('analytics.json', 'w') as f:
    json.dump(analytics_payload, f)

print("Advanced pipeline saved to model.pkl and analytics.json saved!")
