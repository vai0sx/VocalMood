import os
import librosa
import numpy as np
from keras.models import load_model
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr

# Cargar un modelo preentrenado para la detección de emociones
MODEL_PATH = "Emotion_Voice_Detection_Model.h5"
model = load_model(MODEL_PATH)

# Categorías de emociones
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def process_audio_file(file_path):
    data, sampling_rate = librosa.load(file_path, sr=22050)  # Asegurarse de que la tasa de muestreo sea 22050

    # Cambiamos la longitud del audio a 4 segundos
    desired_length = 4 * sampling_rate
    audio_length = len(data)

    if audio_length < desired_length:
        padding = desired_length - audio_length
        data = np.pad(data, (0, padding), 'constant')
    elif audio_length > desired_length:
        data = data[:desired_length]

    # Calcular el tamaño de la ventana y el solapamiento en función de la tasa de muestreo
    window_size = int(sampling_rate * 4 / 216)  # 4 segundos divididos por 216 columnas de MFCC
    overlap = int(window_size / 2)

    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_fft=window_size, hop_length=overlap)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=13)

    # Expandir las dimensiones de los datos para que sean compatibles con el modelo
    print("MFCC Shape:", mfcc.shape)  # Añadir esta línea
    processed_data = np.expand_dims(mfcc, axis=0)
    processed_data = np.expand_dims(processed_data, axis=3)

    return processed_data

def predict_emotion(audio_data):
    # Ajusta estas dimensiones según sea necesario
    new_input_shape = (audio_data.shape[0], 216, 1)

    # Redimensiona la entrada a la forma esperada
    audio_data_reshaped = np.reshape(audio_data, new_input_shape)

    # Alimenta la entrada redimensionada a la capa "sequential"
    prediction = model.predict(audio_data_reshaped)

def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        text = r.recognize_google(audio)

    return text

def analyze_voice(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_format = file_extension.lower()[1:]  # Eliminar el punto (.) y convertir a minúsculas

    if file_format not in ['mp3', 'wav']:
        raise ValueError(f"Formato de archivo no soportado: {file_format}")

    audio = AudioSegment.from_file(file_path, file_format)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    results = []

    for i, chunk in enumerate(chunks):
        chunk_path = f"chunk{i}.wav"
        chunk.export(chunk_path, format="wav")
        audio_data = process_audio_file(chunk_path)
        emotion = predict_emotion(audio_data)
        transcription = transcribe_audio(chunk_path)
        results.append({
            'transcription': transcription,
            'emotion': emotion
        })
        os.remove(chunk_path)

    return results
