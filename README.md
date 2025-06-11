Конечно! Вот пример аккуратного и полезного README.md для твоего проекта whisper-gemini-assistant — с описанием, установкой и использованием:

⸻


# 🎙️ Whisper + Gemini Voice Assistant

Ассистент с распознаванием речи через [OpenAI Whisper](https://github.com/openai/whisper) и генерацией ответов с помощью [Gemini API](https://ai.google.dev/).  
Работает локально, записывает звук, распознаёт текст и задаёт вопросы Gemini.

---

## 📦 Установка

1. Клонируй репозиторий:

```bash
git clone https://github.com/plat-09/whisper-gemini-assistant.git
cd whisper-gemini-assistant

	2.	Создай и активируй виртуальное окружение:

python3 -m venv venv
source venv/bin/activate

	3.	Установи зависимости:

pip install -r requirements.txt


⸻

🔐 Настройки

Создай файл .env и добавь туда:

GEMINI_API_KEY=ваш_API_ключ_от_Google_AI

Также можно настроить параметры в config.py:

# config.py
duration = 10              # длительность записи в секундах
language = "ru"            # язык распознавания речи
input_device = (2, None)   # индекс микрофона (см. sd.query_devices())


⸻

🚀 Запуск

python main.py

Скрипт:
	•	Запишет звук с микрофона
	•	Распознает речь с помощью Whisper
	•	Отправит вопрос в Gemini API
	•	Выведет текст ответа

⸻

⚙️ Зависимости
	•	openai-whisper
	•	google-generativeai
	•	sounddevice
	•	numpy, scipy
	•	python-dotenv

⸻

📌 Возможности
	•	Поддержка русского языка
	•	Локальное распознавание речи (без отправки аудио в облако)
	•	Работа с любой моделью Gemini (например, gemini-1.5-flash)
