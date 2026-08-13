"""
app.py — ValueBet AI PL (wersja jednoplikowa)
===============================================
Wszystko w jednym pliku: pobieranie prawdziwych meczów/kursów (The Odds API),
podatek PL 12%, wyliczenie +EV, analiza AI (Claude/OpenAI) i UI Streamlit.

Brak jakichkolwiek danych mockowych. Jeśli dana liga/sport nie jest aktywna
w tej chwili w The Odds API, aplikacja automatycznie próbuje pozostałych lig
z tej samej dyscypliny (wciąż 100% realne dane) i informuje w UI, które ligi
faktycznie odpowiedziały — nic nie jest zmyślane ani podstawiane na sztywno.

Uruchomienie:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

import requests
import streamlit as st

# ============================================================================
# SEKCJA 1: api_tracker — pobieranie prawdziwych meczów i kursów
# ============================================================================

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Kandydaci sport_key wg dyscypliny. Nie wszystkie muszą być aktywne w danym
# momencie (np. tenisowe turnieje istnieją tylko gdy trwają) — kod filtruje
# to dynamicznie przez realny wykaz /v4/sports, więc zawsze pyta tylko o to,
# co faktycznie istnieje teraz.
DISCIPLINE_SPORT_KEYS: dict[str, list[str]] = {
    "Piłka nożna": [
        "soccer_uefa_champs_league",
        "soccer_epl",
        "soccer_poland_ekstraklasa",
        "soccer_uefa_europa_league",
        "soccer_germany_bundesliga",
        "soccer_spain_la_liga",
        "soccer_italy_serie_a",
        "soccer_france_ligue_one",
    ],
    "Tenis": [
        "tennis_atp_singles",
        "tennis_wta_singles",
    ],
    "MMA": [
        "mma_mixed_martial_arts",
    ],
}

PREFERRED_BOOKMAKER_KEY = "betclic"
REQUEST_TIMEOUT_S = 15


class OddsApiError(Exception):
    """Błąd komunikacji z The Odds API (zły klucz, limit, sieć, brak danych)."""


def _get(url: str, params: dict[str, Any]) -> Any:
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        raise OddsApiError(f"Błąd sieci przy zapytaniu do The Odds API: {exc}") from exc

    if resp.status_code == 401:
        raise OddsApiError("Nieprawidłowy klucz API do The Odds API (401 Unauthorized).")
    if resp.status_code == 422:
        raise OddsApiError(f"Nieprawidłowe parametry zapytania (422): {resp.text}")
    if resp.status_code == 429:
        raise OddsApiError("Przekroczono limit zapytań do The Odds API (429).")
    if not resp.ok:
        raise OddsApiError(f"The Odds API zwróciło błąd {resp.status_code}: {resp.text}")

    return resp.json()


def get_active_sport_keys(api_key: str, discipline: str) -> list[str]:
    """
    Zwraca listę sport_key faktycznie aktywnych TERAZ (wg /v4/sports),
    ograniczoną do kandydatów danej dyscypliny. Jeśli konkretna liga akurat
    nie jest aktywna (np. przerwa międzysezonowa), po prostu nie znajdzie
    się na liście — nie ma tu żadnego fallbacku na dane zmyślone, tylko
    zawężenie do tego, co realnie istnieje.
    """
    candidates = set(DISCIPLINE_SPORT_KEYS.get(discipline, []))
    if not candidates:
        return []

    all_sports = _get(f"{ODDS_API_BASE}/sports", {"apiKey": api_key})
    active = [s["key"] for s in all_sports if s.get("key") in candidates and s.get("active")]
    return active


def fetch_upcoming_odds(
    api_key: str,
    discipline: str,
    hours_ahead: int = 48,
    regions: str = "eu",
    markets: str = "h2h,spreads",
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Pobiera prawdziwe, nadchodzące mecze wraz z kursami dla wybranej
    dyscypliny. Zwraca (lista_meczów, lista_lig_ktore_faktycznie_odpowiedzialy).

    Jeśli jedna liga zwróci błąd/brak danych, próba przechodzi do kolejnej
    ligi z tej samej dyscypliny — cała pula kandydatów jest realna, więc
    to wciąż wyłącznie prawdziwe dane, tylko szersze pokrycie.
    """
    if not api_key:
        raise OddsApiError("Brak klucza API do The Odds API.")

    sport_keys = get_active_sport_keys(api_key, discipline)
    if not sport_keys:
        return [], []

    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(hours=hours_ahead)

    results: list[dict[str, Any]] = []
    leagues_used: list[str] = []

    for sport_key in sport_keys:
        params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        try:
            events = _get(f"{ODDS_API_BASE}/sports/{sport_key}/odds", params)
        except OddsApiError:
            continue  # ta konkretna liga bez danych -> próbuj kolejnej, realnej

        got_any = False
        for event in events:
            commence_raw = event.get("commence_time")
            if not commence_raw:
                continue
            commence_time = dt.datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
            if not (now <= commence_time <= horizon):
                continue

            home_team = event.get("home_team")
            away_team = event.get("away_team")
            bookmakers = event.get("bookmakers", [])
            if not home_team or not away_team or not bookmakers:
                continue

            odds_home, odds_draw, odds_away, used_bk = _extract_h2h_odds(
                bookmakers, home_team, away_team
            )
            if odds_home is None and odds_away is None:
                continue

            results.append(
                {
                    "sport_key": sport_key,
                    "sport_title": sport_key,
                    "event_id": event.get("id", f"{home_team}-{away_team}-{commence_raw}"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence_time,
                    "bookmaker_used": used_bk,
                    "odds_home": odds_home,
                    "odds_draw": odds_draw,
                    "odds_away": odds_away,
                }
            )
            got_any = True

        if got_any:
            leagues_used.append(sport_key)

    results.sort(key=lambda e: e["commence_time"])
    return results, leagues_used


def _extract_h2h_odds(
    bookmakers: list[dict[str, Any]], home_team: str, away_team: str
) -> tuple[float | None, float | None, float | None, str]:
    """
    Priorytet: kurs bezpośrednio od Betclic. Jeśli Betclic nie wystawił
    jeszcze oferty h2h na dane zdarzenie -> realna średnia kursów h2h ze
    WSZYSTKICH bukmacherów EU zwróconych przez API (nadal 100% realne dane).
    """
    for bk in bookmakers:
        if bk.get("key") == PREFERRED_BOOKMAKER_KEY:
            h2h = _find_market(bk, "h2h")
            if h2h:
                home, draw, away = _parse_h2h_outcomes(h2h, home_team, away_team)
                if home is not None or away is not None:
                    return home, draw, away, "betclic"

    home_vals, draw_vals, away_vals = [], [], []
    for bk in bookmakers:
        h2h = _find_market(bk, "h2h")
        if not h2h:
            continue
        h, d, a = _parse_h2h_outcomes(h2h, home_team, away_team)
        if h is not None:
            home_vals.append(h)
        if d is not None:
            draw_vals.append(d)
        if a is not None:
            away_vals.append(a)

    avg = lambda vals: (sum(vals) / len(vals)) if vals else None  # noqa: E731
    return avg(home_vals), avg(draw_vals), avg(away_vals), "eu_average"


def _find_market(bookmaker: dict[str, Any], key: str) -> dict[str, Any] | None:
    for m in bookmaker.get("markets", []):
        if m.get("key") == key:
            return m
    return None


def _parse_h2h_outcomes(
    market: dict[str, Any], home_team: str, away_team: str
) -> tuple[float | None, float | None, float | None]:
    home = draw = away = None
    for outcome in market.get("outcomes", []):
        name = outcome.get("name")
        price = outcome.get("price")
        if name == home_team:
            home = price
        elif name == away_team:
            away = price
        elif name and name.lower() == "draw":
            draw = price
    return home, draw, away


# ============================================================================
# SEKCJA 2: math_engine — podatek PL 12% i wyliczenie +EV
# ============================================================================

POLISH_TAX_RATE = 0.12
TAX_MULTIPLIER = 1 - POLISH_TAX_RATE  # 0.88


def effective_odds(bookmaker_odds: float) -> float:
    """Efektywny_Kurs = Kurs_Betclic * 0.88 (podatek 12%)."""
    if bookmaker_odds is None or bookmaker_odds <= 1.0:
        raise ValueError("Kurs bukmacherski musi być liczbą > 1.0")
    return round(bookmaker_odds * TAX_MULTIPLIER, 4)


def expected_value(ai_probability_pct: float, effective_odd: float) -> float:
    """EV = (Prawdopodobienstwo_AI_PCT / 100 * Efektywny_Kurs) - 1"""
    if ai_probability_pct is None:
        raise ValueError("Brak prawdopodobieństwa AI.")
    prob = max(0.0, min(100.0, ai_probability_pct)) / 100.0
    return round((prob * effective_odd) - 1, 4)


def evaluate_selection(bookmaker_odds: float, ai_probability_pct: float) -> dict:
    eff_odds = effective_odds(bookmaker_odds)
    ev = expected_value(ai_probability_pct, eff_odds)
    return {
        "bookmaker_odds": round(bookmaker_odds, 3),
        "effective_odds": eff_odds,
        "ai_probability_pct": ai_probability_pct,
        "ev": ev,
        "ev_pct": round(ev * 100, 2),
        "is_value_bet": ev >= 0.02,
    }


def filter_and_rank_value_bets(
    evaluated_selections: list[dict], min_ev_pct: float = 2.0, top_n: int = 5
) -> list[dict]:
    min_ev = min_ev_pct / 100.0
    filtered = [s for s in evaluated_selections if s["ev"] >= min_ev]
    filtered.sort(key=lambda s: s["ev"], reverse=True)
    return filtered[:top_n]


# ============================================================================
# SEKCJA 2.5: learning_store — pamięć wyników i "uczenie się" AI na historii
# ============================================================================
# AI (Claude/GPT) nie ma trwałej pamięci między zapytaniami — nie da się go
# "douczyć" wagami tak jak sieć neuronową. Zamiast tego robimy coś, co realnie
# działa i jest uczciwe: zapisujemy KAŻDĄ predykcję do lokalnej bazy SQLite,
# Ty wpisujesz prawdziwy wynik po zakończeniu meczu, a przy KOLEJNYCH analizach
# wstrzykujemy do promptu Twoją dotychczasową, realną skuteczność (per
# dyscyplina i poziom ryzyka) — model dostaje kontekst "jak dotąd trafiałem"
# i kalibruje swoją pewność. To jest 100% oparte na Twoich prawdziwych danych,
# zero zmyślania.

DB_PATH = Path(__file__).parent / "valuebet_history.db"
MIN_SAMPLES_FOR_CALIBRATION = 5


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            event_id TEXT PRIMARY KEY,
            sport_title TEXT,
            home_team TEXT,
            away_team TEXT,
            commence_time TEXT,
            primary_side TEXT,
            rekomendowany_typ TEXT,
            ai_probability_pct INTEGER,
            poziom_ryzyka TEXT,
            ev_pct REAL,
            actual_result TEXT,
            was_correct INTEGER,
            logged_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(record: dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT OR IGNORE INTO predictions
        (event_id, sport_title, home_team, away_team, commence_time, primary_side,
         rekomendowany_typ, ai_probability_pct, poziom_ryzyka, ev_pct, actual_result,
         was_correct, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
        """,
        (
            record["event_id"],
            record["sport_title"],
            record["home_team"],
            record["away_team"],
            record["commence_time"].isoformat(),
            record["primary_side"],
            record["rekomendowany_typ"],
            record["ai_probability_pct"],
            record["poziom_ryzyka"],
            record["ev_pct"],
            dt.datetime.now(dt.timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_pending_predictions() -> list[dict[str, Any]]:
    """Mecze zapisane w historii, które już się rozpoczęły, ale użytkownik
    nie wpisał jeszcze prawdziwego wyniku — czekają na "nauczenie" AI."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT * FROM predictions
        WHERE actual_result IS NULL AND commence_time <= ?
        ORDER BY commence_time DESC
        LIMIT 30
        """,
        (now,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_actual_result(event_id: str, actual_result: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT primary_side FROM predictions WHERE event_id = ?", (event_id,)
    ).fetchone()
    was_correct = None
    if row is not None:
        was_correct = 1 if row[0] == actual_result else 0
    conn.execute(
        "UPDATE predictions SET actual_result = ?, was_correct = ? WHERE event_id = ?",
        (actual_result, was_correct, event_id),
    )
    conn.commit()
    conn.close()


def get_accuracy_stats() -> list[dict[str, Any]]:
    """Skuteczność AI pogrupowana wg dyscypliny — tylko z rozstrzygniętych
    meczów, gdzie użytkownik wpisał prawdziwy wynik."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT sport_title,
               COUNT(*) AS total,
               SUM(was_correct) AS correct
        FROM predictions
        WHERE actual_result IS NOT NULL
        GROUP BY sport_title
        """
    ).fetchall()
    conn.close()
    stats = []
    for r in rows:
        total = r["total"] or 0
        correct = r["correct"] or 0
        accuracy = round((correct / total) * 100, 1) if total else 0.0
        stats.append({"sport_title": r["sport_title"], "total": total, "correct": correct, "accuracy": accuracy})
    return stats


def get_calibration_note(sport_title: str) -> str:
    """
    Buduje krótką notatkę o dotychczasowej, realnej skuteczności AI dla danej
    dyscypliny — wstrzykiwaną do promptu LLM, żeby model "uczył się" na
    Twoich prawdziwych, potwierdzonych wynikach. Zwraca pusty string, jeśli
    próbka jest za mała (żeby nie sugerować fałszywej pewności).
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """
        SELECT COUNT(*) AS total, SUM(was_correct) AS correct
        FROM predictions
        WHERE actual_result IS NOT NULL AND sport_title = ?
        """,
        (sport_title,),
    ).fetchone()
    conn.close()

    total = row[0] or 0
    correct = row[1] or 0
    if total < MIN_SAMPLES_FOR_CALIBRATION:
        return ""

    accuracy = round((correct / total) * 100, 1)
    return (
        f"KONTEKST KALIBRACYJNY: w dyscyplinie '{sport_title}' Twoje dotychczasowe "
        f"rekomendacje (na podstawie {total} rozegranych i potwierdzonych przez "
        f"użytkownika meczów) miały realną skuteczność {accuracy}%. Jeśli ta wartość "
        f"jest wyraźnie niższa niż Twoje typowe prawdopodobieństwa, bądź bardziej "
        f"konserwatywny w ocenie pewności; jeśli wyższa — możesz ufać swojej ocenie "
        f"nieco bardziej. Nie zmieniaj tego mechanicznie, potraktuj jako dodatkowy sygnał."
    )


# ============================================================================
# SEKCJA 3: ai_analyzer — integracja z LLM (Anthropic / OpenAI)
# ============================================================================

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_VERSION = "2023-06-01"

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

Provider = Literal["anthropic", "openai"]


class AiAnalyzerError(Exception):
    """Błąd komunikacji z LLM albo niepoprawny format odpowiedzi."""


SYSTEM_PROMPT = (
    "Jesteś analitykiem sportowym specjalizującym się w typowaniu wyników "
    "meczów na podstawie formy zespołów, składów, historii H2H i kontekstu "
    "sytuacyjnego. Odpowiadasz WYŁĄCZNIE czystym obiektem JSON, bez żadnego "
    "tekstu przed ani po, bez bloków markdown ```. Format odpowiedzi:\n"
    '{"skorygowana_szansa_wygranej_pct": int, "poziom_ryzyka": '
    '"Niski" | "Średni" | "Wysoki", "uzasadnienie_analityczne": string, '
    '"rekomendowany_typ": string}'
)


def _build_user_prompt(
    home_team: str,
    away_team: str,
    commence_time_str: str,
    odds_home: float | None,
    odds_draw: float | None,
    odds_away: float | None,
    sport_title: str,
    calibration_note: str = "",
) -> str:
    odds_lines = [f"Kurs na wygraną {home_team}: {odds_home}"]
    if odds_draw is not None:
        odds_lines.append(f"Kurs na remis: {odds_draw}")
    if odds_away is not None:
        odds_lines.append(f"Kurs na wygraną {away_team}: {odds_away}")

    prompt = (
        f"Dyscyplina: {sport_title}\n"
        f"Mecz: {home_team} vs {away_team}\n"
        f"Data i godzina rozpoczęcia (UTC): {commence_time_str}\n"
        + "\n".join(odds_lines)
        + "\n\nNa podstawie swojej wiedzy o formie, składach i historii "
        "bezpośrednich spotkań tych drużyn/zawodników, oceń realne "
        "prawdopodobieństwo wygranej najbardziej prawdopodobnego wyniku "
        "i zwróć wynik w wymaganym formacie JSON."
    )
    if calibration_note:
        prompt += f"\n\n{calibration_note}"
    return prompt


def _extract_json(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise AiAnalyzerError(f"Nie udało się sparsować JSON z odpowiedzi LLM: {exc}") from exc
    raise AiAnalyzerError("Odpowiedź LLM nie zawierała poprawnego JSON.")


def _validate_analysis(data: dict[str, Any]) -> dict[str, Any]:
    required = {
        "skorygowana_szansa_wygranej_pct",
        "poziom_ryzyka",
        "uzasadnienie_analityczne",
        "rekomendowany_typ",
    }
    missing = required - data.keys()
    if missing:
        raise AiAnalyzerError(f"Odpowiedź LLM nie zawiera pól: {missing}")

    pct = data["skorygowana_szansa_wygranej_pct"]
    try:
        pct = int(round(float(pct)))
    except (TypeError, ValueError) as exc:
        raise AiAnalyzerError("skorygowana_szansa_wygranej_pct nie jest liczbą.") from exc
    data["skorygowana_szansa_wygranej_pct"] = max(1, min(99, pct))

    if data["poziom_ryzyka"] not in ("Niski", "Średni", "Wysoki"):
        data["poziom_ryzyka"] = "Średni"

    return data


def _call_anthropic(api_key: str, user_prompt: str) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    try:
        resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise AiAnalyzerError(f"Błąd sieci przy zapytaniu do Anthropic API: {exc}") from exc

    if not resp.ok:
        raise AiAnalyzerError(f"Anthropic API zwróciło błąd {resp.status_code}: {resp.text}")

    data = resp.json(
