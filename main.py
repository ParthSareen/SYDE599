import pickle

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier

with open('mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)

train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(mnist.data, mnist.target, train_size=60000)
model = MLPClassifier(random_state=1, max_iter=1, hidden_layer_sizes=100)
model.fit(train_x, train_y)

probs = model.predict(test_x)
print("model score:", model.score(test_x, test_y))

plt.figure(figsize=(16, 12))
sklearn.metrics.ConfusionMatrixDisplay.from_predictions(test_y, probs)

print(test_y, probs)
plt.show()
