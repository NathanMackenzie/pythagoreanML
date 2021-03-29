import datagenerator as dg 
import tensorflow as tf
import analyzefit as af
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

#Retrieve training data
file_path = "data.csv"
features = ["a", "b"]
labels = ["c"]

data = dg.DataGenerator(file_path=file_path, features=features, labels=labels)

train_features = data.features
train_labels = data.labels

normalize = preprocessing.Normalization()
normalize.adapt(train_features)

model = tf.keras.Sequential([
    normalize,
    layers.Dense(64, activation="tanh"),
    layers.Dense(1)
])

model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam(0.02))

history = model.fit(train_features, train_labels, validation_split=0.3, epochs=500)

print(model.predict([[3.0, 4.0]]))

graph = af.AnalyzeFit("Fit Data", "Epochs", "Loss")
graph.add_data([history.history['loss'], history.history['val_loss']])
graph.show()