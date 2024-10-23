import numpy as np
import pandas as pd
from keras.models import load_model

model =load_model('StockPrediction.keras')

model_json = model.to_json()
print("=================================================== model architecture ==========================================")
print(model_json)