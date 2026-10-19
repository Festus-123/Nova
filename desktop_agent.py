import sys
import time
import uuid
import logging

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

from main import run_research
from memory import get_memory, get_memories, set_current_user_id
from patch.voice_engine import NebulaSTT, NebulaTTS, NebulaWakeWordDetector, generate_daily_briefing

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


class NebulaDesktopAgent:
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.session_id = f"desktop_{uuid.uuid4().hex[:8]}"
        self.tts = NebulaTTS()
        self.stt = NebulaSTT()
        self.wake_word_detector = NebulaWakeWordDetector()
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
            "[bold green]🎙️ WAKE-WORD ACTIVE (\"Hey Nebula\")[/bold green]"
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
            self.log_trajectory("Wake-word listener unavailable. Install Vosk and the model to enable 'Hey Nebula'.")

    def handle_wake_word_detected(self):
        if not self.wake_word_enabled:
            return
        self.log_trajectory("Wake word detected: 'Hey Nebula'")
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
        self.log_trajectory("Retrieving memory and invoking Nebula research agent...")
        self.display_dashboard()

        try:
            response = run_research(
                query=clean_query,
                user_id=self.user_id,
                session_id=self.session_id,
            )
            self.last_response = response
            self.log_trajectory("Nebula response generated successfully.")
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
                        self.last_response = "Nebula powering down. Have a productive day."
                        self.display_dashboard()
                        self.tts.speak("Nebula powering down. Have a productive day.")
                        break
                    else:
                        self.process_query(choice, from_voice=False)
                except (KeyboardInterrupt, EOFError):
                    self.last_response = "Nebula desktop agent shutting down."
                    self.display_dashboard()
                    break

            self.live = None
            self.wake_word_detector.stop()


def main():
    user_id = "default"
    if len(sys.argv) > 1:
        user_id = sys.argv[1]

    agent = NebulaDesktopAgent(user_id=user_id)
    agent.run()


# Backwards-compatible name from the pre-Nebula (Horizon) release.
HorizonDesktopAgent = NebulaDesktopAgent


if __name__ == "__main__":
    main()
