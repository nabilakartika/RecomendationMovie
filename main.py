from fastapi import FastAPI
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load the trained model
model = load_model('NetflixRecommendation.h5')

class RecommendationRequest(BaseModel):
    user_id: int
    movie_ids: list[int]

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    user_id = np.array([request.user_id])
    movie_ids = np.array(request.movie_ids)
    predictions = model.predict([user_id, movie_ids])
    recommended_movie_ids = movie_ids[np.argsort(predictions)[::-1]]
    return {"recommended_movie_ids": recommended_movie_ids.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
