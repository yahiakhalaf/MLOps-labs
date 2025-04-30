import joblib
import pandas as pd
import logging
import argparse
from typing import Union
from config.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TitanicPredictor:
    def __init__(self, model_path: str = None):
        config = load_config()
        self.model_path = model_path or config["paths"]["model_output"]
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def predict(self, input_data: Union[dict, pd.DataFrame]) -> Union[int, pd.Series]:
        try:
            if isinstance(input_data, dict):
                input_data = self._prepare_input(input_data)
                prediction = int(self.model.predict(input_data)[0])
                logger.info(f"Prediction: {prediction}")
                return prediction
            elif isinstance(input_data, pd.DataFrame):
                predictions = self.model.predict(input_data)
                logger.info("Batch predictions completed.")
                return pd.Series(predictions)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _prepare_input(self, raw_data: dict) -> pd.DataFrame:
        config = load_config()
        expected_features = (
            config["preprocessing"]["numeric_features"] +
            config["preprocessing"]["categorical_features"]
        )
        df = pd.DataFrame([raw_data])
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = None
        return df


def main():
    parser = argparse.ArgumentParser(description="Titanic Predictor")
    parser.add_argument('--csv', type=str, help='Path to test CSV file')
    parser.add_argument('--instance', nargs='*', help='Single instance key=value pairs')

    args = parser.parse_args()
    predictor = TitanicPredictor()

    if args.csv:
        try:
            df = pd.read_csv(args.csv)
            preds = predictor.predict(df)
            for idx, pred in enumerate(preds):
                print(f"Row {idx}: {pred}")
        except Exception as e:
            print(f"Failed to predict from CSV: {str(e)}")

    elif args.instance:
        try:
            instance_dict = dict(kv.split('=') for kv in args.instance)
            # Convert numerical fields to float or int
            for key in instance_dict:
                try:
                    instance_dict[key] = float(instance_dict[key])
                except ValueError:
                    pass
            pred = predictor.predict(instance_dict)
            print(f"Prediction: {pred}")
        except Exception as e:
            print(f"Failed to predict from instance: {str(e)}")
    else:
        print("Please provide --csv path or --instance key=value pairs")

if __name__ == "__main__":
    main()
