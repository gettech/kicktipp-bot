"""Game model for representing football matches and calculating betting tips."""
import random
import re
from typing import Tuple, Union
import google.generativeai as genai
from ..config import Config

# API einmalig konfigurieren, falls ein Key hinterlegt ist
if Config.GEMINI_API_KEY:
    genai.configure(api_key=Config.GEMINI_API_KEY)

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

        # --- GEMINI MODE  ---
        if Config.USE_GEMINI and Config.GEMINI_API_KEY:
            try:
                # Passe self.home_team / self.away_team ggf. an deine echten Variablen an!
                home_name = getattr(self, "home_team", "Heimteam")
                away_name = getattr(self, "away_team", "Auswärtsteam")
                
                # Wir nutzen gemini-1.5-flash, da es extrem schnell und perfekt für Text ist
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    f"Du bist ein Fußballexperte. Tippe das genaue Endergebnis für das Spiel "
                    f"{home_name} gegen {away_name}. Die Quoten sind Heim: {home_quote}, Auswärts: {away_quote}. "
                    f"Antworte AUSSCHLIESSLICH mit dem Ergebnis im Format X:Y (z.B. 2:1). Keine Erklärungen."
                )
                
                response = model.generate_content(prompt)
                
                # Sichere Extraktion des Ergebnisses via Regex (sucht nach Zahl:Zahl)
                match = re.search(r'(\d+)\s*:\s*(\d+)', response.text)
                if match:
                    return int(match.group(1)), int(match.group(2))
                else:
                    print(f"Konnte Gemini-Antwort nicht parsen: {response.text}")
            except Exception as e:
                print(f"Gemini API Fehler: {e}. Nutze Fallback-Logik.")

        # ---  DISCOURAGE_MODE ---
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
