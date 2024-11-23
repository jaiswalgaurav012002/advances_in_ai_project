import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import gradio as gr

# === Paths to data ===
preprocessed_data_path = '/storage/research/data/Few_shot/Ai_project/glove_preprocessed/'
padded_sequences_file = os.path.join(preprocessed_data_path, 'padded_sequences.npy')
labels_file = os.path.join(preprocessed_data_path, 'labels.npy')
glove_embeddings_file = '/storage/research/data/Few_shot/Ai_project/glove_preprocessed/glove.6B.300d.txt'
model_save_path = '/storage/research/data/Few_shot/Ai_project/Models/simple_rnn_model.h5'

# === Step 1: Load Preprocessed Data ===
X = np.load(padded_sequences_file)
y = np.load(labels_file)

# Debug: Check the shape of data
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# If X has 2 dimensions, add a third dimension for features
if len(X.shape) == 2:
    X = np.expand_dims(X, axis=-1)
    print(f"Shape of X after reshaping: {X.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# === Load GloVe embeddings ===
embedding_index = {}
with open(glove_embeddings_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embedding_index[word] = vector

# Create embedding matrix
embedding_dim = 300
vocab_size = X.shape[2]  # Assuming vocabulary size matches GloVe word count
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for i in range(vocab_size):
    embedding_vector = embedding_index.get(str(i))
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# === Step 2: Build the Simple RNN Model ===
model = Sequential()
model.add(Embedding(input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    input_length=X.shape[1],
                    trainable=False))  # Use GloVe embeddings as non-trainable weights
model.add(SimpleRNN(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))  # Simple RNN with 64 units
model.add(Dropout(0.2))
model.add(SimpleRNN(32))  # Simple RNN with 32 units
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model with a fixed learning rate
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary to verify parameter count
model.summary()

# === Step 3: Train the Model ===
batch_size = 128
epochs = 100
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Save the trained model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# === Step 4: Evaluate the Model ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")
y_pred_proba = model.predict(X_test).ravel()
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Save Classification Report
output_directory = '/storage/research/data/Few_shot/Ai_project/Models/'
os.makedirs(output_directory, exist_ok=True)
classification_report_text = classification_report(y_test, y_pred)
with open(os.path.join(output_directory, 'classification_report_simple_rnn.txt'), 'w') as f:
    f.write(classification_report_text)

# Save Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.savefig(os.path.join(output_directory, 'confusion_matrix_simple_rnn.png'))

# Save ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC-ROC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(output_directory, 'roc_curve_simple_rnn.png'))

# Save Accuracy and Loss Curves
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(output_directory, 'accuracy_curve_simple_rnn.png'))

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_directory, 'loss_curve_simple_rnn.png'))

# === Step 5: Deploy the Model with Gradio ===
# Function to preprocess input text for prediction
def preprocess_text(text, max_sequence_length=X.shape[1]):
    words = text.lower().split()  # Tokenization and lowercasing
    sequence = np.zeros((max_sequence_length, X.shape[2]))  # Match dimensions of training data
    for i, word in enumerate(words[:max_sequence_length]):
        if word in embedding_index:
            sequence[i] = embedding_index[word]
    return np.expand_dims(sequence, axis=0)  # Add batch dimension

# Prediction function for Gradio
def predict_speech_type(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)[0][0]
    label = "Hope Speech" if prediction >= 0.5 else "Non-Hope Speech"
    confidence = f"{prediction * 100:.2f}%" if prediction >= 0.5 else f"{(1 - prediction) * 100:.2f}%"
    return f"Prediction: {label} (Confidence: {confidence})"

# Gradio Interface
interface = gr.Interface(
    fn=predict_speech_type,
    inputs=gr.Textbox(lines=3, placeholder="Enter text to classify as Hope or Non-Hope Speech..."),
    outputs="text",
    title="Hope Speech Classifier (Simple RNN)",
    description="Enter a sentence or paragraph, and the model will classify it as either Hope Speech or Non-Hope Speech."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
