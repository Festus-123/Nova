from pathlib import Path
from textwrap import dedent

BASE_DIR = Path(r"c:\Users\hp\Desktop\Practises-folder\Python\my_agent\agent")


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


voice_engine = dedent(
    '''\
    import json
    import logging
    import os
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
            self.wake_phrases = tuple(wake_phrases or ("hey horizon", "horizon"))
            self.model = None
            self.enabled = True
            self._callback = None
            self._stop_event = threading.Event()
            self._listener_thread = None
            self._download_model_if_missing()

        @property
        def available(self):
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

        def start(self, callback):
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
            clean = clean.replace("```", " Code snippet omitted. ")
            clean = clean.replace("`", "")
            clean = clean.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
            clean = clean.replace("#", "")
            clean = clean.replace("[", "").replace("]", "")
            clean = clean.replace("(", "").replace(")", "")
            clean = clean.replace("http://", "").replace("https://", "")
            clean = clean.replace("- ", "").replace("* ", "")
            clean = clean.replace("|", " ")
            while "  " in clean:
                clean = clean.replace("  ", " ")
            return clean.strip()

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

        visual_lines.append("\n*Horizon desktop voice interface is fully armed and listening.*")
        visual_text = "\n".join(visual_lines)

        return spoken_text, visual_text
    '''
)


desktop_agent = dedent(
    '''\
    import logging
    import sys
    import time
    import uuid

    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from main import run_research
    from memory import get_memory, get_memories, set_current_user_id
    from voice_engine import HorizonSTT, HorizonTTS, HorizonWakeWordDetector, generate_daily_briefing

    logging.basicConfig(level=logging.WARNING)
    console = Console()

    GOLD = "#D4AF37"
    AMBER = "#FFBF00"
    OBSIDIAN = "#121212"
    PLATINUM = "#E5E4E2"
    CHARCOAL = "#2A2A2A"
    DARK_GOLD = "#8C7853"
    MUTED_SLATE = "#5C677D"
    ACCENT_CYAN = "#00D2FF"


    class HorizonDesktopAgent:
        def __init__(self, user_id: str = "default"):
            self.user_id = user_id
            self.session_id = f"desktop_{uuid.uuid4().hex[:8]}"
            self.tts = HorizonTTS()
            self.stt = HorizonSTT()
            self.wake_word_detector = HorizonWakeWordDetector()
            self.wake_word_enabled = True
            self.current_state = "IDLE"
            self.trajectory_logs = []
            self.last_response = "Awaiting your next command."
            self.live = None
            set_current_user_id(self.user_id)
            self._start_wake_word_listener()

        def set_state(self, state: str):
            self.current_state = state

        def log_trajectory(self, action: str):
            timestamp = time.strftime("%H:%M:%S")
            self.trajectory_logs.append(f"[{timestamp}] {action}")
            if len(self.trajectory_logs) > 8:
                self.trajectory_logs.pop(0)

        def _animation_frame(self):
            state_frames = {
                "IDLE": ["[ • ]", "[ ◦ ]", "[   ]"],
                "LISTENING": ["───░▒▓▒░───", "───▒▓█▓▒───", "───░▒▓▒░───"],
                "THINKING": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
                "SPEAKING": [" ▃ ", " ▅ ", " ▇ ", " ▆ ", " ▅ ", " ▃ "],
            }
            frames = state_frames.get(self.current_state, state_frames["IDLE"])
            idx = int(time.time() * 5) % len(frames)
            return frames[idx]

        def render_header(self):
            header = Text()
            header.append("✦ ", style=f"bold {GOLD}")
            header.append("HORIZON DESKTOP", style=f"bold {GOLD}")
            header.append(" // ", style=f"bold {MUTED_SLATE}")
            header.append("VOICE INTERFACE", style=f"bold {PLATINUM}")
            header.append(" [ v1.2.0 ]", style=f"dim {MUTED_SLATE}")
            sub = Text("USER TUNED • LOCAL MEMORY • RESEARCH ENGINE • VOICE READY", style=f"italic {DARK_GOLD}")
            return Panel(Text.assemble(header, "\n", sub), border_style=GOLD, padding=(0, 1), style="on #0a0a0a")

        def render_status_bar(self):
            user_name = "User"
            name_mem = get_memory(self.user_id, "identity", "name")
            if name_mem and name_mem.get("value"):
                user_name = name_mem["value"]

            mic_status = "[bold green]MIC DETECTED[/bold green]" if self.stt.has_microphone else "[bold red]NO MIC[/bold red]"
            wake_label = (
                "[bold green]🎙️ WAKE-WORD ACTIVE (\"Hey Horizon\")[/bold green]"
                if self.wake_word_enabled
                else "[bold red]🔇 WAKE-WORD MUTED[/bold red]"
            )
            state_label = {
                "IDLE": f"[bold {MUTED_SLATE}]● STANDBY[/bold {MUTED_SLATE}]",
                "LISTENING": f"[bold {AMBER}]🎙️ LISTENING[/bold {AMBER}]",
                "THINKING": f"[bold {ACCENT_CYAN}]⚡ THINKING[/bold {ACCENT_CYAN}]",
                "SPEAKING": f"[bold {GOLD}]🔊 SPEAKING[/bold {GOLD}]",
            }.get(self.current_state, f"[bold {MUTED_SLATE}]{self.current_state}[/bold {MUTED_SLATE}]")

            status_text = (
                f"  {state_label}   |   {wake_label}   |   👤 {user_name} (ID: {self.user_id})   "
                f"|   💬 Session: {self.session_id}   |   {mic_status}   |   {self._animation_frame()}"
            )
            return Panel(status_text, border_style=GOLD, padding=(0, 1), style="on #101010")

        def render_trajectory_panel(self):
            if not self.trajectory_logs:
                body = Text("No active tool executions yet. Waiting for input.", style=f"italic {MUTED_SLATE}")
            else:
                lines = [f"> {log}" for log in self.trajectory_logs]
                body = Text.from_markup("\n".join(lines))
            return Panel(body, title=f"[bold {GOLD}]TRAJECTORY & LOGS[/bold {GOLD}]", border_style=DARK_GOLD, padding=(0, 1), style="on #0d0d0d")

        def render_response_panel(self):
            response_text = Text(self.last_response, style="white")
            return Panel(
                Align.left(response_text),
                title=f"[bold {GOLD}]HORIZON // LAST RESPONSE[/bold {GOLD}]",
                border_style=GOLD,
                padding=(1, 2),
                style="on #111111",
            )

        def render_controls_menu(self):
            controls = (
                f" [bold {GOLD}][V][/bold {GOLD}] Voice Query   "
                f" [bold {GOLD}][T][/bold {GOLD}] Text Query   "
                f" [bold {GOLD}][B][/bold {GOLD}] Executive Briefing   "
                f" [bold {GOLD}][M][/bold {GOLD}] Wake-Word Toggle   "
                f" [bold {GOLD}][Q][/bold {GOLD}] Exit"
            )
            return Panel(Align.center(controls), border_style=MUTED_SLATE, padding=(0, 1), style="on #080808")

        def render_dashboard(self):
            return Group(
                self.render_header(),
                self.render_status_bar(),
                self.render_trajectory_panel(),
                self.render_response_panel(),
                self.render_controls_menu(),
            )

        def display_dashboard(self):
            if self.live is not None:
                self.live.update(self.render_dashboard())
            else:
                console.print(self.render_dashboard())

        def _start_wake_word_listener(self):
            if self.wake_word_detector.available:
                self.wake_word_detector.start(self.handle_wake_word_detected)
            else:
                self.log_trajectory("Wake-word listener unavailable. Install Vosk and the model to enable 'Hey Horizon'.")

        def handle_wake_word_detected(self):
            if not self.wake_word_enabled:
                return
            self.log_trajectory("Wake word detected: 'Hey Horizon'")
            self.set_state("LISTENING")
            self.last_response = "Wake word detected. Listening for your command..."
            self.display_dashboard()
            self.handle_voice_input()

        def toggle_wake_word_listener(self):
            self.wake_word_enabled = not self.wake_word_enabled
            self.wake_word_detector.set_enabled(self.wake_word_enabled)
            if self.wake_word_enabled:
                self.log_trajectory("Wake-word listener re-enabled.")
                self._start_wake_word_listener()
            else:
                self.log_trajectory("Wake-word listener muted.")
                self.wake_word_detector.stop()
            self.display_dashboard()

        def perform_daily_briefing(self):
            self.set_state("THINKING")
            self.log_trajectory("Generating executive briefing from stored memory...")
            self.display_dashboard()
            spoken_text, visual_text = generate_daily_briefing(self.user_id)
            self.last_response = visual_text
            self.set_state("SPEAKING")
            self.log_trajectory("Speaking executive briefing via TTS...")
            self.display_dashboard()
            self.tts.speak(spoken_text)
            self.set_state("IDLE")
            self.log_trajectory("Executive briefing complete.")
            self.display_dashboard()

        def process_query(self, query: str, from_voice: bool = False):
            if not query or not query.strip():
                return

            clean_query = query.strip()
            source_label = "VOICE INPUT" if from_voice else "KEYBOARD INPUT"
            self.last_response = f"User query ({source_label}): {clean_query}"
            self.set_state("THINKING")
            self.log_trajectory(f"Query received: '{clean_query[:40]}...'")
            self.log_trajectory("Retrieving memory and invoking Horizon research agent...")
            self.display_dashboard()

            try:
                response = run_research(
                    query=clean_query,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                self.last_response = response
                self.log_trajectory("Horizon response generated successfully.")
                self.set_state("SPEAKING")
                self.log_trajectory("Synthesizing audio response via TTS...")
                self.display_dashboard()
                self.tts.speak(response)
            except Exception as exc:
                self.last_response = f"Error: {exc}"
                self.log_trajectory(f"Execution error: {exc}")
                self.display_dashboard()

            self.set_state("IDLE")
            self.log_trajectory("Ready for next command.")
            self.display_dashboard()

        def handle_voice_input(self):
            if not self.stt.has_microphone:
                self.last_response = "No microphone detected. Use text mode instead."
                self.display_dashboard()
                return

            self.set_state("LISTENING")
            self.display_dashboard()
            success, text = self.stt.listen(timeout=7, phrase_time_limit=15)

            if not success:
                self.set_state("IDLE")
                self.last_response = text
                self.display_dashboard()
                return

            self.process_query(text, from_voice=True)

        def handle_text_input(self):
            console.print()
            prompt = Text.from_markup(f"[bold {GOLD}]HORIZON PROMPT[/bold {GOLD}] [dim]>[/dim] ")
            try:
                query = console.input(prompt)
                if query and query.strip():
                    self.process_query(query.strip(), from_voice=False)
            except (KeyboardInterrupt, EOFError):
                pass

        def show_memory_vault(self):
            memories = get_memories(self.user_id, minimum_importance=1)
            table = Table(
                title=f"HORIZON PERSISTENT MEMORY VAULT // USER: {self.user_id}",
                border_style=GOLD,
                header_style=f"bold {GOLD}",
                show_lines=True,
                style="on #0f0f0f",
            )
            table.add_column("ID", style="dim", width=4)
            table.add_column("Category", style=f"bold {AMBER}", width=14)
            table.add_column("Key", style=f"bold {PLATINUM}", width=20)
            table.add_column("Stored Value / Memory", style="white")
            table.add_column("Priority", justify="center", width=8)
            table.add_column("Last Updated", style=f"dim {MUTED_SLATE}", width=20)

            if not memories:
                table.add_row("-", "Empty", "No memories", "No persistent memories stored yet.", "-", "-")
            else:
                for memory in memories:
                    table.add_row(
                        str(memory["id"]),
                        memory["category"],
                        memory["key"],
                        memory["value"],
                        f"{memory['importance']}/10",
                        memory["updated_at"][:19],
                    )

            self.last_response = "Persistent memory vault shown."
            console.print(table)

        def run(self):
            console.clear()
            self.display_dashboard()
            self.perform_daily_briefing()

            with Live(self.render_dashboard(), console=console, refresh_per_second=6, transient=False) as live:
                self.live = live
                while True:
                    try:
                        self.display_dashboard()
                        menu_prompt = Text.from_markup(
                            f"[bold {GOLD}]HORIZON[/bold {GOLD}] [dim](V=Voice, T=Type, B=Briefing, M=Wake, Q=Quit)[/dim] [bold {GOLD}]>[/bold {GOLD}] "
                        )
                        choice = console.input(menu_prompt).strip().upper()
                        if not choice:
                            continue

                        if choice == "V":
                            self.handle_voice_input()
                        elif choice == "T":
                            self.handle_text_input()
                        elif choice == "B":
                            self.perform_daily_briefing()
                        elif choice == "M":
                            self.toggle_wake_word_listener()
                        elif choice == "Q" or choice == "EXIT":
                            self.last_response = "Horizon powering down. Have a productive day."
                            self.display_dashboard()
                            self.tts.speak("Horizon powering down. Have a productive day.")
                            break
                        else:
                            self.process_query(choice, from_voice=False)
                    except (KeyboardInterrupt, EOFError):
                        self.last_response = "Horizon desktop agent shutting down."
                        self.display_dashboard()
                        break

                self.live = None
                self.wake_word_detector.stop()


    def main():
        user_id = "default"
        if len(sys.argv) > 1:
            user_id = sys.argv[1]

        agent = HorizonDesktopAgent(user_id=user_id)
        agent.run()


    if __name__ == "__main__":
        main()
    '''
)


requirement_text = """\
langchain
wikipedia
langchain-community
langchain-openai
langchain-anthropic
python-dotenv
pydantic
ddgs
uvicorn
fastapi
pytesseract
pillow
rich
SpeechRecognition
pyttsx3
pyaudio
vosk
"""


if __name__ == "__main__":
    write_file(BASE_DIR / "voice_engine.py", voice_engine)
    write_file(BASE_DIR / "desktop_agent.py", desktop_agent)
    write_file(BASE_DIR / "requirement.txt", requirement_text)

    requirements_path = BASE_DIR / "requirements.txt"
    if not requirements_path.exists():
        write_file(requirements_path, requirement_text)
