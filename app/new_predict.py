from tensorflow.keras.models import load_model
import pandas as pd 
from io import StringIO

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

# Accessing Google Storage
from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket('toxic_detection')
blob = bucket.get_blob('train.csv')

model = load_model('kernel.h5')
train_data = pd.read_csv(StringIO(blob.download_as_string().decode()))["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_data))
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
maxlen = 200

def predict_text(text):
    print(text)
    test = tokenizer.texts_to_sequences([text])
    X_t = pad_sequences(test, maxlen=maxlen)
    result = model.predict(X_t)
    confidence = result.max()*100
    predicted_class = list_classes[np.argmax(result)]

    return predicted_class, confidence