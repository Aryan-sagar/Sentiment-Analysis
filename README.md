

# 🧠 Sentiment Analysis with RNN

A deep learning project using Recurrent Neural Networks (RNN) to classify movie reviews into positive or negative sentiments. Built using TensorFlow and trained on the IMDB movie review dataset.

---

## 🔍 Overview

This project demonstrates binary text classification using a Bidirectional LSTM-based RNN. The model processes raw text reviews, encodes them using `TextVectorization`, and learns to predict sentiment with high accuracy. It includes visualizations, metrics, and examples.

---

## ✨ Features

- Binary sentiment classification (positive/negative)
- Real-world dataset: IMDB Movie Reviews
- RNN architecture using Bidirectional LSTM
- TensorFlow TextVectorization for preprocessing
- Training visualization and sample predictions

---

## 🧰 Tech Stack

- **Language:** Python
- **Frameworks/Libraries:** TensorFlow, Keras, TensorFlow Datasets (TFDS), Numpy, Matplotlib
- **Dataset:** IMDB Movie Reviews (from TFDS)

---

## 🧠 Model Architecture

```python
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

- `TextVectorization` to tokenize and vectorize input text
- `Embedding` for word embeddings
- `Bidirectional LSTM` for sequential context
- `Dense` layers for binary classification

---

## 🧪 Performance

- **Accuracy:** ~85–90% on validation set
- **Test Accuracy:** Printed at end of training

```python
Test Loss: 0.28
Test Accuracy: 0.88
```

---

## 🔧 Installation & Usage

```bash
git clone https://github.com/yourusername/sentiment-analysis-rnn.git
cd sentiment-analysis-rnn
pip install -r requirements.txt
```

To run the notebook or script:
```bash
python sentiment_rnn.py
# or use Jupyter Notebook
jupyter notebook sentiment_rnn.ipynb
```

---

## 📈 Visualizations

- Training vs Validation Accuracy
- Training vs Validation Loss

```python
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
```

---

## 📝 Sample Prediction

```python
sample_text = "The movie was cool. The animation was amazing. I recommend it."
pred = model.predict(np.array([sample_text]))
print("Prediction:", pred)
```

---

## 🚀 Future Improvements

- Extend to multiclass sentiment (neutral, mixed)
- Train using LSTM + Attention or GRU
- Deploy as a REST API or Streamlit web app
- Explore transfer learning with pre-trained embeddings

---

## 📜 License

Apache 2.0 License — based on [TensorFlow official tutorial](https://www.tensorflow.org/text/tutorials/text_classification_rnn)

---
