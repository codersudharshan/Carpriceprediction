import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_clean_data(filepath):
    """Load and preprocess the dataset"""
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Basic data validation
        if df.empty:
            raise ValueError("Empty dataframe loaded")
        if 'Price' not in df.columns:
            raise ValueError("Target column 'Price' not found in data")
            
        # Check for missing values
        missing_values = df.isna().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in the dataset")
            
        # Drop rows where target is missing
        df = df.dropna(subset=['Price'])
        
        logger.info(f"Data loaded successfully with {df.shape[0]} samples and {df.shape[1]} features")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(X_train, y_train):
    """Train Random Forest model with proper NaN handling"""
    try:
        logger.info("Training Random Forest model...")
        
        # Create pipeline with imputer and model
        model = make_pipeline(
            SimpleImputer(strategy='median'),  # Handle missing values
            RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        )
        
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Model Evaluation Metrics:")
        logger.info(f"MAE: ₹{metrics['mae']:,.2f}")
        logger.info(f"RMSE: ₹{metrics['rmse']:,.2f}")
        logger.info(f"R² Score: {metrics['r2']:.4f}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_model(model, filepath):
    """Save trained model to disk"""
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    try:
        # Configuration
        DATA_FILE = "cleaned_car_data.csv"
        MODEL_FILE = "car_price_model.pkl"
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        
        # Load and clean data
        df = load_and_clean_data(DATA_FILE)
        
        # Prepare features and target
        X = df.drop(columns=['Price'])
        y = df['Price']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        save_model(model, MODEL_FILE)
        
        logger.info("✅ Model training pipeline completed successfully!")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return None, None

if __name__ == "__main__":
    model, metrics = main()