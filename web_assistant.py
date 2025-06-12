import gradio as gr
from main import record_audio, transcribe_audio, ask_gemini

def voice_to_gemini():
    audio, fs = record_audio()
    text = transcribe_audio(audio, fs)

    if not text.strip():
        return "⚠️ Речь не распознана", ""

    reply = ask_gemini(text)
    return text, reply

with gr.Blocks(title="🎙️ Whisper + Gemini") as demo:
    gr.Markdown("## 🎤 Whisper + Gemini Ассистент")
    btn = gr.Button("🎙️ Говорить 10 сек")
    text_output = gr.Textbox(label="🗣️ Распознанный текст")
    gemini_output = gr.Textbox(label="🤖 Ответ от Gemini")

    btn.click(fn=voice_to_gemini, outputs=[text_output, gemini_output])

demo.launch()