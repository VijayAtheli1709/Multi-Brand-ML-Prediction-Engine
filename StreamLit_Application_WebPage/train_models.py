import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# Dictionary of brands and their best model type based on your team's analysis
BRAND_MODEL_MAP = {
    'Mercedes': 'DecisionTree',    # Based on previous analysis
    'Audi': 'DecisionTree',        # Based on previous analysis
    'Vauxhall': 'DecisionTree',    # Based on previous analysis
    'BMW': 'RandomForest',         # Based on previous analysis
    'Ford': 'DecisionTree',        # Based on previous analysis
    'Toyota': 'RandomForest',      # Based on previous analysis
    'Hyundai': 'LinearRegression', # RMSE: 2041.13 vs 2698.22 for DT
    'Skoda': 'DecisionTree',       # RMSE: 1808.78 vs 1828.48 for LR
    'VW': 'DecisionTree'           # RMSE: 2133.65 vs 2551.21 for LR
}

# Create models directory if it doesn't exist
print("Creating models directory...")
os.makedirs('models', exist_ok=True)

try:
    # Load dataset
    print("Loading dataset...")
    df = pd.read_excel('Final_dataset.xlsx')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Brands to process (you can modify this list)
    brands_to_process = list(BRAND_MODEL_MAP.keys())
    
    for brand in brands_to_process:
        print(f"\n{'='*50}")
        print(f"Processing {brand} data...")
        
        # Filter for brand cars
        brand_df = df[df['Brand'] == brand].copy()
        print(f"Found {brand_df.shape[0]} records for {brand}")
        
        # Skip if too few records
        if brand_df.shape[0] < 50:  # Adjust this threshold as needed
            print(f"Skipping {brand} due to insufficient data (less than 50 records)")
            continue
        
        # Define features/target
        print("Preparing features and target...")
        y = brand_df['price']
        X = brand_df[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']]
        
        # Check for missing values
        print("Checking for missing values...")
        missing = X.isnull().sum()
        if missing.sum() > 0:
            print("Warning: Missing values found:")
            print(missing[missing > 0])
            print("Filling missing values...")
            # Fill missing categorical values with 'Unknown'
            for col in ['model', 'transmission', 'fuelType']:
                if col in X.columns:
                    X[col].fillna('Unknown', inplace=True)
            # Fill missing numerical values with median
            for col in ['year', 'mileage', 'tax', 'mpg', 'engineSize']:
                if col in X.columns:
                    X[col].fillna(X[col].median(), inplace=True)
        
        # Create preprocessing pipeline
        print("Creating preprocessing pipeline...")
        categorical_cols = [col for col in ['model', 'transmission', 'fuelType'] if col in X.columns]
        numerical_cols = [col for col in ['year', 'mileage', 'tax', 'mpg', 'engineSize'] if col in X.columns]
        
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')
        
        # Select regressor based on brand's best model
        model_type = BRAND_MODEL_MAP.get(brand, 'DecisionTree')
        
        if model_type == 'LinearRegression':
            regressor = LinearRegression()
            print(f"Using Linear Regression for {brand}")
        elif model_type == 'RandomForest':
            regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            print(f"Using Random Forest Regressor for {brand}")
        else:  # Default to Decision Tree
            regressor = DecisionTreeRegressor(random_state=42)
            print(f"Using Decision Tree Regressor for {brand}")
        
        # Create full pipeline
        print("Creating full pipeline...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        # Train/test split
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train
        print(f"Training model for {brand}...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        print(f"R² score on training set: {train_score:.4f}")
        print(f"R² score on test set: {test_score:.4f}")
        
        # Save entire pipeline
        output_path = f'models/{brand}_pipeline.pkl'
        print(f"Saving pipeline to {output_path}...")
        joblib.dump(pipeline, output_path)
        print(f"✓ Successfully saved pipeline to {output_path}")
        
        # Verify the file was saved correctly
        if os.path.exists(output_path):
            print(f"✓ Verified: File exists at {output_path}")
            file_size = os.path.getsize(output_path)
            print(f"  File size: {file_size} bytes")
            
            if file_size < 2000:
                print("Warning: File size is suspiciously small! Check the file content.")
            
            # Try to reload the model to verify
            print("Verifying model can be loaded correctly...")
            try:
                loaded_pipeline = joblib.load(output_path)
                print("✓ Model loaded successfully!")
                
                # Make a test prediction
                test_data = X_test.iloc[:1].copy()
                test_price = y_test.iloc[0]
                pred_price = loaded_pipeline.predict(test_data)[0]
                print(f"Test prediction: ${pred_price:.2f} (Actual: ${test_price:.2f})")
                
            except Exception as e:
                print(f"× Error loading model: {str(e)}")
                
        else:
            print(f"× Error: File was not saved at {output_path}")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())

print("Done!")