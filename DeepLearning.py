import pandas as pd
import gzip
import json
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

from sklearn.metrics import mean_absolute_error, accuracy_score

from tensorflow.keras.utils import plot_model

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        if i > 100000:
            break
    return pd.DataFrame.from_dict(df, orient='index')

#--------------------------------------------------
#Itt kell kicserélni
df = getDF('Video_Games_5.json.gz')

#df = getDF('Movies_and_TV_5.json.gz') 
#----------------------------------------------

# Remove rows with missing review text
df = df[df['reviewText'].notna()]      

# Convert all reviewText values to string explicitly 
df['reviewText'] = df['reviewText'].astype(str)

# Now safe to drop remaining columns
df.drop(columns=[
    "verified", "reviewTime", "reviewerID", "asin", "reviewerName", 
    "unixReviewTime", "vote", "style", "summary"
], axis=1, inplace=True)


X = df["reviewText"]
Y = df["overall"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42)

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=42)

# Hyperparams
VOCAB_SIZE = 50000        # you can change to 50k if you have GPU
MAX_LEN = 300             # truncate/pad all reviews to 200 tokens

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_validation)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
X_val_pad   = pad_sequences(X_val_seq,   maxlen=MAX_LEN, padding='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding='post')

def to_coral_labels(y, num_classes=5):
    y = np.array(y).astype(int)
    new_y = []

    for rating in y:
        binary_targets = [1 if rating > k else 0 for k in range(1, num_classes)]
        new_y.append(binary_targets)

    return np.array(new_y)

Y_train_coral = to_coral_labels(Y_train)
Y_val_coral   = to_coral_labels(Y_validation)
Y_test_coral  = to_coral_labels(Y_test)

EMBED_DIM = 128
LSTM_UNITS = 128

model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.3),
    LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),

    # 4 outputs for the 5 ordinal classes
    Dense(4, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train_pad, Y_train_coral,
    validation_data=(X_val_pad, Y_val_coral),
    epochs=5,
    batch_size=128
)

def coral_predict(model, X):
    preds = model.predict(X)
    
    ratings = []
    for p in preds:
        # Count how many predictions > 0.5
        num = np.sum(p > 0.5)
        rating = num + 1
        ratings.append(rating)

    return np.array(ratings)

pred_test = coral_predict(model, X_test_pad)

#plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)

print("Ordinal Accuracy:", accuracy_score(Y_test, pred_test))
print("MAE:", mean_absolute_error(Y_test, pred_test))

