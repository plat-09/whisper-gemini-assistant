# Импорт необходимых библиотек
import sounddevice as sd               # Для записи звука с микрофона
import numpy as np                     # Для работы с массивами (используется внутри sounddevice)
import whisper                         # Модель для распознавания речи от OpenAI
import tempfile                        # Для создания временных файлов
from scipy.io.wavfile import write    # Для записи WAV-файлов из numpy-массива
import google.generativeai as genai   # Библиотека для работы с Gemini API от Google
import os                             # Для работы с переменными окружения
from dotenv import load_dotenv        # Для загрузки переменных из .env файла
from config import AUDIO_INPUT_DEVICE, AUDIO_DURATION, WHISPER_LANGUAGE # Импортируем настройки из config.py


# Загружаем API-ключ из файла .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Конфигурируем доступ к Gemini API
genai.configure(api_key=api_key)

# Инициализируем модель Gemini (можно указать любую подходящую из list_models)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Создаём чат-сессию (может хранить контекст предыдущих сообщений, если нужно)
chat = model.start_chat()

# Загружаем модель Whisper для распознавания речи
# Вместо "medium" можно использовать "base", "small", "large" (в зависимости от качества и скорости)
whisper_model = whisper.load_model("medium")

def record_audio(duration=AUDIO_DURATION, fs=44100):
    """
    Записывает аудио с микрофона в течение указанного количества секунд.
    duration — длительность записи (в секундах)
    fs — частота дискретизации (обычно 44100 Гц)
    """
    print("🎙️ Запись...")
    sd.default.device = (AUDIO_INPUT_DEVICE, None)  # Устанавливаем устройство для записи (номер можно уточнить через sd.query_devices())
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)  # Записываем звук (моно)
    sd.wait()  # Ждём окончания записи
    print("✅ Запись завершена")
    return recording, fs

def transcribe_audio(audio_data, fs):
    """
    Преобразует аудиоданные в текст с помощью Whisper.
    Сохраняет данные во временный WAV-файл, отправляет в модель и удаляет файл.
    """
    print("🔎 Распознавание...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, fs, audio_data)  # Сохраняем запись во временный файл
        result = whisper_model.transcribe(tmpfile.name, language=WHISPER_LANGUAGE)  # Распознаём речь на русском
        os.unlink(tmpfile.name)  # Удаляем временный файл после использования
    print("🗣️ Текст:", result["text"])
    return result["text"]

def ask_gemini(prompt):
    """
    Отправляет распознанный текст в Gemini и выводит ответ.
    """
    response = chat.send_message(prompt)
    print("🤖 Gemini:\n", response.text)
    return response.text

# Главная точка входа
if __name__ == "__main__":
    audio, fs = record_audio()             # 1. Записываем голос
    question = transcribe_audio(audio, fs) # 2. Распознаём текст из аудио
    if question.strip():                   # 3. Если текст не пустой, отправляем в Gemini
        ask_gemini(question)
    else:
        print("⚠️ Не удалось распознать речь.")