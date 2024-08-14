from fastapi import FastAPI , HTTPException
from app.utils.model_loader import load_model_and_preprocessor
import pandas as pd 











app = FastAPI()





@app.get("/hello")
async def hello():
    return {"message ": "hello world"}



@app.post('/predict/{model_name}')
async def predict(model_name: str , request :PredictionRequest):
    try:  
         model , preprocessor = load_model_and_preprocessor(model_name)
    except ValueError as e:
        raise HTTPException(status_code=404 , detail=str(e))
    
    input_data = pd.DataFrame([request.dict()])
    input_data = preprocessor.transform(input_data)
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0,1] if hasattr(model, 'predict_proba')
    return {prediction : int(prediction[0]), "probability": probability}

