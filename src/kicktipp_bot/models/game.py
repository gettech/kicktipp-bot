"""Game model for representing football matches and calculating betting tips."""
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Union

# --- NEU: Das neue offizielle Google GenAI SDK ---
from google import genai
from google.genai import types

from ..config import Config

# API-Client einmalig vorbereiten, falls ein Key hinterlegt ist
gemini_client = None
if Config.GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)


@dataclass
class Game:
    """Represents a football game with teams, betting quotes, and tip calculation logic."""

    home_team: str
    away_team: str
    quotes: List[str]
    game_time: datetime
    _validated_quotes: List[float] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate and process data after initialization."""
        self.home_team = self.home_team.strip()
        self.away_team = self.away_team.strip()
        self._validated_quotes = self._validate_quotes(self.quotes)

    def _validate_quotes(self, quotes: List[str]) -> List[float]:
        """
        Validate and convert quotes to float values.

        Args:
            quotes: List of quote strings

        Returns:
            List of float quotes

        Raises:
            ValueError: If quotes are invalid
        """
        if len(quotes) != 3:
            raise ValueError(f"Expected 3 quotes, got {len(quotes)}")

        try:
            return [float(quote) for quote in quotes]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid quote values: {quotes}") from e

    def calculate_tip(self, home_quote: Union[float, None] = None, away_quote: Union[float, None] = None) -> Tuple[int, int]:
        """
        Calculate betting tip using Gemini API (if enabled), 
        falling back to logic/quotes.
        """
        if home_quote is None:
            home_quote = self._validated_quotes[0]
        if away_quote is None:
            away_quote = self._validated_quotes[2]

        quote_difference = home_quote - away_quote

        # --- GEMINI MODE MIT WEBSUCHE ---
        if Config.USE_GEMINI and gemini_client:
            try:
                home_name = getattr(self, "home_team", "Heimteam")
                away_name = getattr(self, "away_team", "Auswärtsteam")
                
                prompt = (
                    f"Du bist ein professioneller Fußball-Analyst. "
                    f"Recherchiere jetzt sofort mit der Google Suche die aktuelle Form, "
                    f"letzte News, Verletzungen, Sperren und Head-to-Head-Statistiken "
                    f"für das bevorstehende Spiel: {home_name} gegen {away_name}. "
                    f"Die aktuellen Wettquoten sind Heim: {home_quote}, Auswärts: {away_quote}. "
                    f"Nutze diese Quoten nur als groben Indikator, verlasse dich aber für deinen Tipp "
                    f"auf deine aktuelle Recherche der echten Umstände. "
                    f"Tippe basierend auf diesen Fakten das wahrscheinlichste genaue Endergebnis. "
                    f"WICHTIG: Deine finale Antwort darf AUSSCHLIESSLICH das Ergebnis im Format X:Y enthalten (z.B. 2:1). "
                    f"Schreibe absolut keinen anderen Text, keine Begründung und keine Quelle dazu."
                )
                
                # Aufruf der API mit aktiviertem Google-Search Tool
                response = gemini_client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}]
                    )
                )
                
                match = re.search(r'(\d+)\s*:\s*(\d+)', response.text)
                if match:
                    return int(match.group(1)), int(match.group(2))
                else:
                    print(f"Konnte Gemini-Antwort nicht parsen: {response.text}")
            except Exception as e:
                print(f"Gemini API Fehler: {e}. Nutze Fallback-Logik.")

        # --- DISCOURAGE_MODE ---
        if Config.DISCOURAGE_MODE:
            if abs(quote_difference) < 0.25:
                return 1, 1
            elif quote_difference < 0:
                return 2, 1
            else:
                return 1, 2

        # --- ORIGINAL LOGIK ---
        random_goal = random.randint(0, 1)
        coefficient = 0.3 if abs(quote_difference) > 7 else 0.75

        if abs(quote_difference) < 0.25:
            return random_goal, random_goal
        elif quote_difference < 0:
            home_goals = max(0, round(-quote_difference * coefficient)) + random_goal
            away_goals = random_goal
            return home_goals, away_goals
        else:
            home_goals = random_goal
            away_goals = max(0, round(quote_difference * coefficient)) + random_goal
            return home_goals, away_goals
            
    def __str__(self) -> str:
        """String representation of the game."""
        return f"{self.home_team} vs {self.away_team} at {self.game_time.strftime('%d.%m.%y %H:%M')}"
