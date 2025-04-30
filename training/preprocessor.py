from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logger = logging.getLogger(__name__)

def build_preprocessor(config: dict) -> ColumnTransformer:
    """Create preprocessing pipeline with logging."""
    logger.info("Building preprocessing pipeline")
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=config["preprocessing"]["numeric_strategy"])),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=config["preprocessing"]["categorical_strategy"])),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config["preprocessing"]["numeric_features"]),
            ("cat", categorical_transformer, config["preprocessing"]["categorical_features"])
        ])
    
    logger.info("Preprocessor created successfully")
    return preprocessor