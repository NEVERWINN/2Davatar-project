import os
import time
import torch
import cv2
from PIL import Image
from gradio_client import Client
import streamlit as st

st.set_page_config(page_title="SadTalker", layout="centered")

if "video_ready" not in st.session_state:
    st.session_state.video_ready = False
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "processing" not in st.session_state:
    st.session_state.processing = False

text = st.text_input("Введите сообщение для озвучки:")

def validate_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fixed_path = os.path.join(os.path.dirname(image_path), "fixed_" + os.path.basename(image_path))
        Image.fromarray(img).save(fixed_path)
        return fixed_path
    except Exception as e:
        raise RuntimeError(f"Ошибка обработки изображения: {str(e)}")

def text_to_speech(text, output_path='output.wav'):
    language = 'ru'
    model_id = 'v3_1_ru'
    speaker = 'baya'

    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=model_id
    )

    model.save_wav(
        text=text,
        speaker=speaker,
        sample_rate=48000,
        audio_path=output_path
    )
    return os.path.abspath(output_path)

def animate_with_sadtalker(image_path, audio_path, server_url="http://127.0.0.1:7860/"):
    client = Client(server_url)
    time.sleep(10)
    try:
        result = client.predict(
            image_path,
            audio_path,
            "full",
            False,
            True,
            1,
            256,
            0,
            fn_index=0
        )
        return result
    except Exception as e:
        print(f"Ошибка SadTalker: {e}")
        return None

def is_valid_video(path):
    try:
        cap = cv2.VideoCapture(path)
        valid = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 and cap.get(cv2.CAP_PROP_FPS) > 0
        cap.release()
        return valid
    except:
        return False

def wait_for_final_video(results_dir="results", after_time=0, timeout=180):
    start_time = time.time()

    while time.time() - start_time < timeout:
        candidate = None
        newest_time = 0

        for root, _, files in os.walk(results_dir):
            for f in files:
                if f.lower().endswith(".mp4") and "enhanced" in f.lower():
                    full_path = os.path.join(root, f)
                    mod_time = os.path.getmtime(full_path)
                    if mod_time > after_time and mod_time > newest_time:
                        candidate = full_path
                        newest_time = mod_time

        if candidate and os.path.exists(candidate):
            if is_valid_video(candidate):
                return candidate

        time.sleep(3)

    return None
if text and not st.session_state.video_ready and not st.session_state.processing:
    st.session_state.processing = True
    status = st.empty()

    try:
        status.info("Генерация аудио...")
        audio_file = text_to_speech(text)

        status.info("Обработка изображения...")
        image_file = os.path.abspath('../../../AI_Projects/SadTalker/test.jpg')
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Изображение не найдено: {image_file}")
        fixed_image = validate_image(image_file)

        gen_start_time = time.time()
        status.info("Генерация видео...")
        animate_with_sadtalker(fixed_image, audio_file)

        status.info("Улучшение качества видео...")
        final_video = wait_for_final_video("../../../AI_Projects/SadTalker/results", after_time=gen_start_time, timeout=180)

        if final_video and os.path.exists(final_video):
            st.session_state.video_ready = True
            st.session_state.video_path = final_video
            status.success("Видео готово!")
        else:
            status.warning("Видео не найдено.")
            st.session_state.processing = False

    except Exception as e:
        status.error(f"Ошибка: {str(e)}")
        st.session_state.processing = False

if st.session_state.video_ready and st.session_state.video_path:
    st.video(st.session_state.video_path)
    with open(st.session_state.video_path, "rb") as f:
        st.download_button(
            label="Скачать видео",
            data=f,
            file_name=os.path.basename(st.session_state.video_path),
            mime="video/mp4"
        )

