from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
max_words = 10000
max_len = 100
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words)
word_index = reuters.get_word_index()
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
y_integers = np.array(y_train)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weights_dict = dict(enumerate(class_weights))
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(SimpleRNN(128, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(46, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training the model... this may take a few minutes.")
model.fit(x_train, y_train_cat, epochs=20, batch_size=128, validation_split=0.2, class_weight=class_weights_dict, verbose=1)

loss, acc = model.evaluate(x_test, y_test_cat)
print(f"\n Test Accuracy: {acc:.2f}")
categories = [
    "cocoa", "grain", "crude oil", "earnings", "acquisitions", "grain/oilseed", "trade",
    "money-fx", "interest rates", "money supply", "inflation", "jobs", "reserves", "ship",
    "wheat", "corn", "oilseed", "veg-oil", "soybean", "livestock", "cpi", "gdp",
    "retail", "housing", "inventories", "aluminium", "gold", "tin", "strategic metals",
    "livestock/meat", "coffee", "sugar", "tea", "rubber", "cotton", "potato", "copper",
    "zinc", "lead", "nickel", "tin", "orange", "petrochemicals", "coal", "iron-steel",
    "housing", "other"
]

def encode_text(text):
       tokens = text_to_word_sequence(text)
    seq = []
    for word in tokens:
        if word in word_index and word_index[word] < max_words:
            seq.append(word_index[word])
        else:
            seq.append(2)  # unknown token
    return pad_sequences([seq], maxlen=max_len)

user_input = input("\n Enter a short news article text:\n> ")
encoded_input = encode_text(user_input)

prediction = model.predict(encoded_input)
predicted_class = prediction.argmax()
print("\n Predicted Category Index:", predicted_class)
print(" Predicted Category Name:", categories[predicted_class])
