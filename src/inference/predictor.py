import joblib
import pandas as pd
import logging
import argparse
import yaml
from pathlib import Path
import joblib
logger = logging.getLogger(__name__)

class TitanicPredictor:
    def __init__(self, model_path: str = None):
        # Load DVC config
        with open("params.yaml") as f:
            self.cfg = yaml.safe_load(f)
        
        self.model_path = model_path or self.cfg['paths']['model_output']
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def predict(self, input_data):
        try:
            if isinstance(input_data, dict):
                input_df = self._prepare_input(input_data)
                return int(self.model.predict(input_df)[0])
            elif isinstance(input_data, pd.DataFrame):
                return self.model.predict(input_data)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _prepare_input(self, raw_data: dict) -> pd.DataFrame:
        expected_features = (
            self.cfg['data']['features']['numerical'] +
            self.cfg['data']['features']['categorical']
        )
        return pd.DataFrame({k: [raw_data.get(k)] for k in expected_features})

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
    print(f"model Saved")
    joblib.dump(predictor,"/teamspace/studios/this_studio/mlops_project/MLOps-labs/models/predictor.joblib")
if __name__ == "__main__":
    main()