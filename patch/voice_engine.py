import json
import logging
import os
import re
import threading
import time
import urllib.request
import zipfile
from typing import Callable, Optional, Tuple

import pyttsx3
import speech_recognition as sr

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import vosk
    from vosk import KaldiRecognizer, Model
except ImportError:
    vosk = None
    KaldiRecognizer = None
    Model = None

from memory import get_memories, get_memory

logger = logging.getLogger("horizon_voice")


class HorizonWakeWordDetector:
    """Lightweight local wake-word detection backed by Vosk."""

    MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    MODEL_DIR_NAME = "vosk-model-small-en-us"

    def __init__(self, model_dir: Optional[str] = None, wake_phrases: Optional[Tuple[str, ...]] = None):
        base_dir = model_dir or os.path.join(os.path.expanduser("~"), ".horizon", "vosk")
        self.model_dir = os.path.join(base_dir, self.MODEL_DIR_NAME)
        self.wake_phrases = tuple((wake_phrases or ("hey horizon", "horizon")))
        self.model = None
        self.enabled = True
        self._callback: Optional[Callable[[], None]] = None
        self._stop_event = threading.Event()
        self._listener_thread = None
        self._download_model_if_missing()

    @property
    def available(self) -> bool:
        return self.model is not None and vosk is not None and KaldiRecognizer is not None

    def _download_model_if_missing(self):
        if vosk is None or Model is None:
            return False

        if os.path.isdir(self.model_dir):
            try:
                self.model = Model(self.model_dir)
                return True
            except Exception as exc:
                logger.warning("Unable to load cached Vosk model: %s", exc)

        try:
            os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
            archive_path = os.path.join(os.path.dirname(self.model_dir), "vosk-model-small-en-us.zip")
            logger.info("Downloading Vosk wake-word model to %s", self.model_dir)
            urllib.request.urlretrieve(self.MODEL_URL, archive_path)
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(os.path.dirname(self.model_dir))
            if os.path.isdir(self.model_dir):
                self.model = Model(self.model_dir)
                return True
        except Exception as exc:
            logger.warning("Unable to download Vosk model automatically: %s", exc)
            self.model = None

        return False

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def start(self, callback: Callable[[], None]):
        if self._listener_thread and self._listener_thread.is_alive():
            self._callback = callback
            return

        if not self.available or pyaudio is None:
            logger.warning("Wake-word listener unavailable. Install vosk and pyaudio or ensure the model exists.")
            self._callback = callback
            return

        self._callback = callback
        self._stop_event.clear()
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2)

    def _listen_loop(self):
        if not self.available or pyaudio is None:
            return

        stream = None
        audio = None
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000,
            )
            recognizer = KaldiRecognizer(self.model, 16000)
            stream.start_stream()

            while not self._stop_event.is_set():
                if not self.enabled:
                    time.sleep(0.1)
                    continue

                try:
                    chunk = stream.read(4000, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.1)
                    continue

                if not chunk:
                    continue

                if recognizer.AcceptWaveform(chunk):
                    payload = json.loads(recognizer.Result())
                    text = (payload.get("text") or "").lower()
                    if any(phrase in text for phrase in self.wake_phrases):
                        if self._callback is not None:
                            self._callback()
                        time.sleep(0.75)
        except Exception as exc:
            logger.warning("Wake-word listener error: %s", exc)
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                if audio is not None:
                    audio.terminate()
            except Exception:
                pass


class HorizonTTS:
    """Crisp, natural Text-To-Speech engine using pyttsx3."""

    def __init__(self, rate: int = 185, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
            voices = self.engine.getProperty("voices")
            if voices:
                chosen = voices[0]
                for voice in voices:
                    voice_name = (voice.name or "").lower()
                    voice_id = (voice.id or "").lower()
                    if "zira" in voice_name or "david" in voice_name or "en" in voice_id:
                        chosen = voice
                        break
                self.engine.setProperty("voice", chosen.id)
        except Exception as exc:
            logger.warning("Failed to initialize pyttsx3: %s", exc)
            self.engine = None

    def clean_text_for_speech(self, text: str) -> str:
        if not text:
            return ""

        clean = text
        clean = re.sub(r"```[\s\S]*?```", " Code snippet omitted. ", clean)
        clean = re.sub(r"`([^`]+)`", r"", clean)
        clean = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"", clean)
        clean = re.sub(r"^#+\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"", clean)
        clean = re.sub(r"https?://\S+", "", clean)
        clean = re.sub(r"^\s*[-*+]\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"^\s*>\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\|", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def speak(self, text: str, max_chars: int = 400) -> bool:
        clean_text = self.clean_text_for_speech(text)
        if not clean_text:
            return False

        if len(clean_text) > max_chars:
            truncated = clean_text[:max_chars]
            last_period = truncated.rfind(".")
            if last_period > 100:
                speech_payload = truncated[: last_period + 1]
            else:
                speech_payload = truncated + "..."
        else:
            speech_payload = clean_text

        try:
            if not self.engine:
                self._init_engine()
            if self.engine:
                self.engine.say(speech_payload)
                self.engine.runAndWait()
                return True
        except Exception as exc:
            logger.warning("Error during TTS playback: %s", exc)
            self._init_engine()

        return False


class HorizonSTT:
    """Speech recognition engine with ambient noise adjustment and fallback resilience."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.has_microphone = self._check_mic()

    def _check_mic(self) -> bool:
        try:
            return len(sr.Microphone.list_microphone_names()) > 0
        except Exception:
            return False

    def listen(self, timeout: int = 7, phrase_time_limit: int = 15) -> Tuple[bool, str]:
        if not self.has_microphone:
            return False, "No microphone input device detected."

        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            text = self.recognizer.recognize_google(audio)
            return True, text.strip()
        except sr.WaitTimeoutError:
            return False, "Listening timed out. No speech detected."
        except sr.UnknownValueError:
            return False, "Could not understand the audio."
        except sr.RequestError as exc:
            return False, f"Speech recognition service error: {exc}"
        except Exception as exc:
            return False, f"Microphone error: {str(exc)}"


def generate_daily_briefing(user_id: str = "default") -> Tuple[str, str]:
    clean_user = (user_id or "default").strip() or "default"
    memories = get_memories(clean_user, minimum_importance=1)

    user_name = "there"
    name_mem = get_memory(clean_user, "identity", "name")
    if name_mem and name_mem.get("value"):
        user_name = name_mem["value"].split()[0]

    now = __import__("datetime").datetime.now()
    hour = now.hour
    if hour < 12:
        greeting_time = "Good morning"
    elif hour < 17:
        greeting_time = "Good afternoon"
    else:
        greeting_time = "Good evening"

    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    projects = [m["value"] for m in memories if m.get("category") == "project"]
    skills = [m["value"] for m in memories if m.get("category") == "skill"]
    preferences = [m["value"] for m in memories if m.get("category") == "preference"]

    spoken_parts = [
        f"{greeting_time}, {user_name}.",
        f"Today is {date_str}, and the current time is {time_str}.",
    ]

    if projects:
        spoken_parts.append(f"Your active project is {projects[0]}.")

    spoken_parts.append("Horizon systems are online and ready for your command.")
    spoken_text = " ".join(spoken_parts)

    visual_lines = [
        f"**{greeting_time.upper()}, {user_name.upper()}**",
        f"📅 **Date:** {date_str}  |  ⏰ **Time:** {time_str}",
        f"👤 **User Identity:** {user_name} (ID: `{clean_user}`)",
    ]

    if projects:
        visual_lines.append(f"🚀 **Active Project:** {', '.join(projects[:2])}")
    if skills:
        visual_lines.append(f"⚡ **Specialization:** {', '.join(skills[:2])}")
    if preferences:
        visual_lines.append(f"⚙️ **Preferences:** {', '.join(preferences[:2])}")

    visual_lines.append("*Horizon desktop voice interface is fully armed and listening.*")
    visual_text = "".join(visual_lines)

    return spoken_text, visual_text
