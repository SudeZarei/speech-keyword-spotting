import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import keras
import os
import librosa

# Load trained model
model = keras.models.load_model('speech_commands_model.keras')

# Define command labels
commands = np.array(['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'])
print(f"Commands: {commands}")

# Decode WAV file to waveform
def decode_audio(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        print(f"Original sample rate: {sample_rate} Hz (resampled to 16000 Hz)")
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio
    except Exception as e:
        raise ValueError(f"Invalid WAV file: {e}")

# Convert waveform to spectrogram
def get_spectrogram(waveform):
    waveform = waveform[:16000]
    print(f"Waveform shape after truncation: {waveform.shape}")
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

# Preprocess WAV file for prediction
def preprocess_wav(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"WAV file not found: {file_path}")
    waveform = decode_audio(file_path)
    print(f"Waveform shape before truncation: {waveform.shape}")
    spectrogram = get_spectrogram(waveform)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.expand_dims(spectrogram, 0)
    return spectrogram

# Process user-provided WAV file
sample_name = input("Sample name: ")
sample_file = f'../voices/{sample_name}.wav'
try:
    spectrogram = preprocess_wav(sample_file)
    prediction = model.predict(spectrogram)
    print(f"Prediction shape: {prediction.shape}")
    if prediction.shape[1] != len(commands):
        raise ValueError(f"Model output shape {prediction.shape[1]} does not match number of commands {len(commands)}")
    
    # Get predicted label
    predicted_label = commands[np.argmax(prediction[0])]
    
    # Plot prediction probabilities
    plt.bar(commands, tf.nn.softmax(prediction[0]))
    plt.title(f'Prediction: "{predicted_label}", Voice: "{sample_name}.wav"')
    plt.show()
    
    print(f'Predicted command: {predicted_label}')
except Exception as e:
    print(f"Error processing WAV file: {e}")