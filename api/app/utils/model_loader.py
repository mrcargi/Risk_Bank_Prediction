from joblib import load
import os 


def load_model_and_preprocessor(model_name):
    model_path = f'app/models/{model_name}/model.pkl'
    preprocessor_path =f'app/models/{model_name}/preprocessor.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise ValueError(f"Model or preprocessor not found for {model_name}")
    
    
    model = load(model_path)
    preprocessor = load(preprocessor_path)
    return model , preprocessor