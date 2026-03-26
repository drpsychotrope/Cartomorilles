"""weather.py — Alertes météo pour prospection morilles (Meteoblue)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType

import numpy as np
import requests

from config import MAP_CENTER

logger = logging.getLogger("cartomorilles.weather")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_METEOBLUE_BASE_URL = "https://my.meteoblue.com/packages/basic-day"
_API_KEY_ENV = "METEOBLUE_API_KEY"
_API_KEY_DEFAULT = "Rgyb8cKoA9MNOLHv"

_CACHE_DIR = Path("data")
_CACHE_MAX_AGE_S = 3600  # 1 h

# Seuils prospection morilles
_TEMP_OPTIMAL: tuple[float, float] = (10.0, 20.0)
_TEMP_RANGE: tuple[float, float] = (5.0, 25.0)
_TEMP_NIGHT_MIN: float = 2.0
_PRECIP_RECENT_OPTIMAL: tuple[float, float] = (2.0, 15.0)
_PRECIP_DAY_MAX: float = 5.0
_HUMIDITY_OPTIMAL: tuple[float, float] = (60.0, 90.0)
_WIND_OPTIMAL: float = 15.0
_WIND_MAX: float = 25.0

# Choc hydrique → explosion de fructification
_BURST_RAIN_WINDOW: int = 3          # jours de pluie à observer
_BURST_RAIN_MIN: float = 10.0        # mm cumulés min sur la fenêtre
_BURST_RAIN_INTENSE: float = 20.0    # mm cumulés = choc marqué
_BURST_DRY_THRESH: float = 2.0       # mm max pour considérer jour sec
_BURST_TEMP_MIN: float = 8.0         # °C moyenne min jour de rebond
_BURST_TEMP_DELTA_BONUS: float = 3.0 # °C hausse vs moyenne pluie → bonus
_BURST_BONUS: float = 0.12           # bonus score jour standard
_BURST_BONUS_INTENSE: float = 0.20   # bonus score choc marqué
_BURST_BONUS_THERMAL: float = 0.06   # bonus supplémentaire si réchauffement

_PROSPECTING_LABELS = MappingProxyType({
    4: "🟢 Excellent",
    3: "🟡 Bon",
    2: "🟠 Moyen",
    1: "🔴 Défavorable",
    0: "⛔ Très défavorable",
})

_JOURS_FR = (
    "lundi", "mardi", "mercredi", "jeudi",
    "vendredi", "samedi", "dimanche",
)
_MOIS_FR = (
    "", "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_date_fr(iso_date: str) -> str:
    """'2026-03-27' → 'vendredi 27 mars'."""
    d = date.fromisoformat(iso_date)
    return f"{_JOURS_FR[d.weekday()]} {d.day} {_MOIS_FR[d.month]}"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DayForecast:
    """Prévision journalière Meteoblue."""

    date: str
    temperature_max: float
    temperature_min: float
    temperature_mean: float
    precipitation: float
    precipitation_probability: float
    relativehumidity_mean: float
    windspeed_mean: float
    predictability: float


@dataclass(frozen=True, slots=True)
class BurstSignal:
    """Signal d'explosion de fructification détecté."""

    rain_cumul: float
    rain_days: int
    temp_delta: float
    intensity: str       # "modéré" | "fort"
    bonus: float


@dataclass(frozen=True, slots=True)
class ProspectingDay:
    """Évaluation d'un jour pour la prospection morilles."""

    date: str
    date_fr: str
    score: float
    level: int
    label: str
    details: tuple[str, ...]
    forecast: DayForecast
    burst: BurstSignal | None


# ---------------------------------------------------------------------------
# WeatherChecker
# ---------------------------------------------------------------------------


class WeatherChecker:
    """Check météo locale pour prospection morilles via Meteoblue."""

    def __init__(
        self,
        api_key: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get(_API_KEY_ENV)
            or _API_KEY_DEFAULT
        )
        self._lat = lat if lat is not None else float(MAP_CENTER["lat"])
        self._lon = lon if lon is not None else float(MAP_CENTER["lon"])
        self._cache_file = (cache_dir or _CACHE_DIR) / "weather_cache.json"
        self._forecasts: list[DayForecast] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch(self, use_cache: bool = True) -> list[DayForecast]:
        """Récupère les prévisions 7 jours."""
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                self._forecasts = cached
                logger.info(
                    "✅ Météo : cache valide (%d jours)", len(cached)
                )
                return cached

        raw = self._fetch_api()
        self._forecasts = self._parse_response(raw)
        self._save_cache(raw)
        logger.info(
            "✅ Météo : %d jours récupérés via API", len(self._forecasts)
        )
        return self._forecasts

    def evaluate(self) -> list[ProspectingDay]:
        """Évalue chaque jour pour la prospection morilles."""
        if not self._forecasts:
            self.fetch()

        results: list[ProspectingDay] = []

        for i, fc in enumerate(self._forecasts):
            score, details = self._score_day(fc, i, self._forecasts)

            # Détection choc hydrique → fructification
            burst = self._detect_burst(fc, i, self._forecasts)
            if burst is not None:
                score = float(np.clip(score + burst.bonus, 0.0, 1.0))
                details.append(
                    f"🍄 Choc hydrique {burst.intensity} "
                    f"({burst.rain_cumul:.0f}mm/{burst.rain_days}j "
                    f"→ +{burst.bonus:.0%})"
                )
                if burst.temp_delta >= _BURST_TEMP_DELTA_BONUS:
                    details.append(
                        f"🌡️ Réchauffement +{burst.temp_delta:.0f}°C"
                    )

            level = self._score_to_level(score)
            label = _PROSPECTING_LABELS[level]
            date_fr = _format_date_fr(fc.date)
            results.append(
                ProspectingDay(
                    date=fc.date,
                    date_fr=date_fr,
                    score=round(score, 3),
                    level=level,
                    label=label,
                    details=tuple(details),
                    forecast=fc,
                    burst=burst,
                )
            )
            logger.debug(
                "  %s : %.2f (%s)%s — %s",
                date_fr,
                score,
                label,
                " 🍄BURST" if burst else "",
                "; ".join(details),
            )

        excellent = sum(1 for r in results if r.level >= 3)
        bursts = sum(1 for r in results if r.burst is not None)
        logger.info(
            "📊 Prospection : %d/%d jours favorables (bon+), "
            "%d signaux fructification",
            excellent,
            len(results),
            bursts,
        )
        return results

    @staticmethod
    def format_report(days: list[ProspectingDay]) -> str:
        """Rapport texte des alertes prospection."""
        lines: list[str] = [
            "",
            "═══ ALERTES PROSPECTION MORILLES ═══",
            "",
        ]
        for d in days:
            fc = d.forecast
            burst_tag = " 🍄" if d.burst else ""
            lines.append(
                f"  {d.date_fr:<22s}  {d.label}{burst_tag:<22s}  "
                f"score={d.score:.2f}  "
                f"T={fc.temperature_min:.0f}/{fc.temperature_max:.0f}°C  "
                f"P={fc.precipitation:.0f}mm  "
                f"H={fc.relativehumidity_mean:.0f}%  "
                f"V={fc.windspeed_mean:.0f}km/h"
            )
            if d.details:
                lines.append(
                    f"  {'':<22s}  └─ {' | '.join(d.details)}"
                )

        lines.append("")

        # Alerte fructification
        burst_days = [d for d in days if d.burst is not None]
        if burst_days:
            lines.append(
                "  🍄🍄🍄 SIGNAL FRUCTIFICATION détecté — "
                "choc hydrique puis beau temps :"
            )
            for d in burst_days:
                assert d.burst is not None
                lines.append(
                    f"       → {d.date_fr} "
                    f"(intensité {d.burst.intensity}, "
                    f"+{d.burst.bonus:.0%} bonus)"
                )
            lines.append("")

        best = [d for d in days if d.level >= 3]
        if best:
            lines.append(
                f"  🍄 Meilleurs jours : "
                f"{', '.join(d.date_fr for d in best)}"
            )
        else:
            lines.append("  ⚠️ Aucun jour favorable cette semaine")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Détection explosion de fructification
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_burst(
        fc: DayForecast,
        idx: int,
        all_fc: list[DayForecast],
    ) -> BurstSignal | None:
        """
        Détecte le pattern choc hydrique → beau temps.

        Conditions :
        - Fenêtre J-3..J-1 : cumul précipitations >= 10 mm
        - Jour J : sec (< 2mm) + température moyenne >= 8°C
        - Bonus si réchauffement marqué vs jours de pluie
        """
        if idx < 1:
            return None

        # Jour J doit être sec et assez chaud
        if fc.precipitation > _BURST_DRY_THRESH:
            return None
        if fc.temperature_mean < _BURST_TEMP_MIN:
            return None

        # Cumul pluie fenêtre précédente
        win_start = max(0, idx - _BURST_RAIN_WINDOW)
        rain_days_fc = all_fc[win_start:idx]
        if not rain_days_fc:
            return None

        rain_cumul = sum(d.precipitation for d in rain_days_fc)
        if rain_cumul < _BURST_RAIN_MIN:
            return None

        # Au moins 1 jour significatif de pluie dans la fenêtre
        rain_days = sum(
            1 for d in rain_days_fc if d.precipitation > _BURST_DRY_THRESH
        )
        if rain_days < 1:
            return None

        # Delta thermique : moyenne jour J vs moyenne jours de pluie
        rain_temp_mean = np.mean(
            [d.temperature_mean for d in rain_days_fc]
        )
        temp_delta = fc.temperature_mean - float(rain_temp_mean)

        # Calcul bonus
        intense = rain_cumul >= _BURST_RAIN_INTENSE
        bonus = _BURST_BONUS_INTENSE if intense else _BURST_BONUS

        if temp_delta >= _BURST_TEMP_DELTA_BONUS:
            bonus += _BURST_BONUS_THERMAL

        # Atténuation J+2, J+3 après le choc (premier jour sec = max)
        # Cherche le premier jour sec consécutif
        first_dry = idx
        for j in range(idx - 1, win_start - 1, -1):
            if all_fc[j].precipitation <= _BURST_DRY_THRESH:
                first_dry = j
            else:
                break
        days_since_rain = idx - first_dry
        if days_since_rain > 0:
            bonus *= max(0.4, 1.0 - days_since_rain * 0.25)

        return BurstSignal(
            rain_cumul=round(rain_cumul, 1),
            rain_days=rain_days,
            temp_delta=round(temp_delta, 1),
            intensity="fort" if intense else "modéré",
            bonus=round(bonus, 3),
        )

    # ------------------------------------------------------------------
    # Scoring interne
    # ------------------------------------------------------------------

    def _score_day(
        self,
        fc: DayForecast,
        idx: int,
        all_fc: list[DayForecast],
    ) -> tuple[float, list[str]]:
        """Score composite [0, 1] pour un jour de prospection."""
        components: list[tuple[float, float]] = []
        details: list[str] = []

        # 1. Température jour (poids 0.25)
        t_score = self._score_temperature(fc.temperature_mean)
        components.append((t_score, 0.25))
        if t_score < 0.3:
            details.append(f"T° défavorable ({fc.temperature_mean:.0f}°C)")
        elif t_score >= 0.8:
            details.append(f"T° idéale ({fc.temperature_mean:.0f}°C)")

        # 2. Gel nocturne (poids 0.20)
        frost_score = self._score_frost(fc.temperature_min)
        components.append((frost_score, 0.20))
        if frost_score < 0.5:
            details.append(
                f"⚠️ Gel nocturne ({fc.temperature_min:.0f}°C)"
            )

        # 3. Précipitations jour — sec = mieux (poids 0.15)
        precip_score = self._score_precip_day(fc.precipitation)
        components.append((precip_score, 0.15))
        if fc.precipitation > _PRECIP_DAY_MAX:
            details.append(f"Pluie forte ({fc.precipitation:.0f}mm)")

        # 4. Précipitations récentes J-3..J-1 — humidité sol (poids 0.20)
        recent_precip = self._recent_precipitation(idx, all_fc)
        recent_score = self._score_recent_precip(recent_precip)
        components.append((recent_score, 0.20))
        if recent_score >= 0.7:
            details.append(f"Sol humide ({recent_precip:.0f}mm J-3)")
        elif recent_score < 0.3:
            details.append(f"Sol sec ({recent_precip:.0f}mm J-3)")

        # 5. Humidité relative (poids 0.10)
        hum_score = self._score_humidity(fc.relativehumidity_mean)
        components.append((hum_score, 0.10))

        # 6. Vent (poids 0.10)
        wind_score = self._score_wind(fc.windspeed_mean)
        components.append((wind_score, 0.10))
        if fc.windspeed_mean > _WIND_MAX:
            details.append(f"Vent fort ({fc.windspeed_mean:.0f}km/h)")

        # Composite pondéré
        total_w = sum(w for _, w in components)
        composite = (
            sum(s * w for s, w in components) / total_w
            if total_w > 0
            else 0.0
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        if not details:
            details.append("Conditions correctes")

        return composite, details

    @staticmethod
    def _score_temperature(t_mean: float) -> float:
        lo, hi = _TEMP_OPTIMAL
        rlo, rhi = _TEMP_RANGE
        if lo <= t_mean <= hi:
            return 1.0
        if t_mean < rlo or t_mean > rhi:
            return 0.0
        if t_mean < lo:
            return float(np.clip((t_mean - rlo) / (lo - rlo), 0.0, 1.0))
        return float(np.clip((rhi - t_mean) / (rhi - hi), 0.0, 1.0))

    @staticmethod
    def _score_frost(t_min: float) -> float:
        if t_min >= _TEMP_NIGHT_MIN:
            return 1.0
        if t_min <= -3.0:
            return 0.0
        return float(
            np.clip((t_min + 3.0) / (_TEMP_NIGHT_MIN + 3.0), 0.0, 1.0)
        )

    @staticmethod
    def _score_precip_day(precip: float) -> float:
        if precip <= 1.0:
            return 1.0
        if precip <= _PRECIP_DAY_MAX:
            t = (precip - 1.0) / (_PRECIP_DAY_MAX - 1.0)
            return float(np.clip(1.0 - t * 0.4, 0.6, 1.0))
        return float(
            np.clip(0.6 - (precip - _PRECIP_DAY_MAX) / 20.0, 0.0, 0.6)
        )

    @staticmethod
    def _recent_precipitation(
        idx: int, all_fc: list[DayForecast]
    ) -> float:
        start = max(0, idx - 3)
        return sum(all_fc[j].precipitation for j in range(start, idx))

    @staticmethod
    def _score_recent_precip(cumul: float) -> float:
        lo, hi = _PRECIP_RECENT_OPTIMAL
        if lo <= cumul <= hi:
            return 1.0
        if cumul < lo:
            return float(np.clip(cumul / lo, 0.2, 1.0))
        return float(np.clip(1.0 - (cumul - hi) / 30.0, 0.3, 1.0))

    @staticmethod
    def _score_humidity(rh: float) -> float:
        lo, hi = _HUMIDITY_OPTIMAL
        if lo <= rh <= hi:
            return 1.0
        if rh < lo:
            return float(np.clip(rh / lo, 0.2, 1.0))
        return float(np.clip(1.0 - (rh - hi) / 10.0, 0.5, 1.0))

    @staticmethod
    def _score_wind(ws: float) -> float:
        ceil = _WIND_MAX * 1.5
        if ws <= _WIND_OPTIMAL:
            return 1.0
        if ws >= ceil:
            return 0.0
        return float(
            np.clip(
                1.0 - (ws - _WIND_OPTIMAL) / (ceil - _WIND_OPTIMAL),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _score_to_level(score: float) -> int:
        if score >= 0.75:
            return 4
        if score >= 0.55:
            return 3
        if score >= 0.40:
            return 2
        if score >= 0.25:
            return 1
        return 0

    # ------------------------------------------------------------------
    # API / cache
    # ------------------------------------------------------------------

    def _fetch_api(self) -> dict[str, object]:
        """Appel Meteoblue basic-day."""
        params: dict[str, object] = {
            "apikey": self._api_key,
            "lat": self._lat,
            "lon": self._lon,
            "format": "json",
            "timeformat": "iso8601",
        }
        logger.info(
            "🌤️  Meteoblue API → lat=%.4f lon=%.4f",
            self._lat,
            self._lon,
        )
        resp = requests.get(
            _METEOBLUE_BASE_URL,
            params=params,  # type: ignore[arg-type]
            timeout=15,
        )
        resp.raise_for_status()
        data: dict[str, object] = resp.json()

        if "error" in data:
            msg = data.get("error_message", str(data["error"]))
            raise RuntimeError(f"Meteoblue API error: {msg}")

        return data

    @staticmethod
    def _parse_response(raw: dict[str, object]) -> list[DayForecast]:
        """Parse la réponse Meteoblue basic-day."""
        dd = raw.get("data_day")
        if not isinstance(dd, dict):
            raise KeyError("Réponse Meteoblue sans 'data_day'")

        times = dd["time"]
        assert isinstance(times, list)
        n = len(times)

        def _f(key: str, i: int, default: float = 80.0) -> float:
            seq = dd.get(key)
            if isinstance(seq, list) and i < len(seq):
                return float(seq[i])
            return default

        forecasts: list[DayForecast] = []
        for i in range(n):
            forecasts.append(
                DayForecast(
                    date=str(times[i])[:10],
                    temperature_max=_f("temperature_max", i, 15.0),
                    temperature_min=_f("temperature_min", i, 5.0),
                    temperature_mean=_f("temperature_mean", i, 10.0),
                    precipitation=_f("precipitation", i, 0.0),
                    precipitation_probability=_f(
                        "precipitation_probability", i, 50.0
                    ),
                    relativehumidity_mean=_f(
                        "relativehumidity_mean", i, 70.0
                    ),
                    windspeed_mean=_f("windspeed_mean", i, 10.0),
                    predictability=_f("predictability", i, 80.0),
                )
            )

        return forecasts

    def _load_cache(self) -> list[DayForecast] | None:
        if not self._cache_file.exists():
            return None
        try:
            raw = json.loads(
                self._cache_file.read_text(encoding="utf-8")
            )
            ts = raw.get("fetched_at", 0)
            age = datetime.now(tz=timezone.utc).timestamp() - ts
            if age > _CACHE_MAX_AGE_S:
                logger.debug("   Cache météo expiré (%.0fs)", age)
                return None
            return self._parse_response(raw["data"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("⚠️ Cache météo corrompu : %s", exc)
            return None

    def _save_cache(self, raw: dict[str, object]) -> None:
        payload = {
            "fetched_at": datetime.now(tz=timezone.utc).timestamp(),
            "lat": self._lat,
            "lon": self._lon,
            "data": raw,
        }
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(
                "   Cache météo sauvegardé → %s", self._cache_file
            )
        except OSError as exc:
            logger.warning(
                "⚠️ Impossible d'écrire le cache météo : %s", exc
            )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def check_weather(
    api_key: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> list[ProspectingDay]:
    """Fetch + évalue + log le rapport prospection."""
    checker = WeatherChecker(api_key=api_key, lat=lat, lon=lon)
    checker.fetch()
    days = checker.evaluate()
    report = WeatherChecker.format_report(days)
    logger.info(report)
    return days


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_weather()