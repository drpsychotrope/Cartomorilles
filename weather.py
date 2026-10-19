"""weather.py — Alertes météo pour prospection morilles (Open-Meteo).

Source : API Open-Meteo ``/v1/forecast`` (https://open-meteo.com), sans clé,
usage non commercial. Au point MAP_CENTER : 14 jours d'historique
(``past_days``) + 7 jours de prévision. Le sol (température vers 10 cm,
humidité 3–9 cm) vient des variables horaires, moyennées par jour ; son
réchauffement sur 7 jours multiplie le score du jour (×0,5–1), ce qui
sépare le printemps de l'automne (D42).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import requests

from config import ALTITUDE_OPTIMAL, MAP_CENTER

logger = logging.getLogger("cartomorilles.weather")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEZONE = "Europe/Paris"
# Altitude du point : milieu de l'optimum morille (config), pas celle du
# terrain à MAP_CENTER (~1 060 m, sol 3 à 6 °C plus froid) — D42
_ELEVATION_M: float = (ALTITUDE_OPTIMAL[0] + ALTITUDE_OPTIMAL[1]) / 2
_PAST_DAYS: int = 14       # historique : pluie J-3..J-1, choc hydrique, réchauffement du sol
_FORECAST_DAYS: int = 7    # aujourd'hui → J+6
_DAILY_VARS: tuple[str, ...] = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "precipitation_probability_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
)
_HOURLY_VARS: tuple[str, ...] = (
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_moisture_3_to_9cm",
)

_CACHE_DIR = Path("data")
_CACHE_FILE = "openmeteo_cache.json"
_CACHE_SOURCE = "open-meteo"
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

# Sol (moyennes journalières des valeurs horaires)
_SOIL_DEPTH_CM: float = 10.0                 # profondeur des relevés publiés (5–10 cm)
_SOIL_TEMP_OPTIMAL: tuple[float, float] = (8.0, 12.0)
_SOIL_TEMP_RANGE: tuple[float, float] = (4.0, 18.0)
_SOIL_WARMING_WINDOW: int = 7                # jours : J-6..J contre J-13..J-7
_SOIL_WARMING_MIN_DAYS: int = 5              # valeurs mini par fenêtre, sinon tendance inconnue
_SOIL_WARMING_FULL: float = 1.5              # °C d'écart : +1,5 → ×1 ; −1,5 → ×plancher
_SOIL_WARMING_FLOOR: float = 0.5             # facteur quand le sol refroidit (automne)
_SOIL_MOISTURE_OPTIMAL: tuple[float, float] = (0.20, 0.40)  # m³/m³, 3–9 cm
_SOIL_MOISTURE_DRY: float = 0.10             # m³/m³ → score plancher 0,1
_SOIL_MOISTURE_WET: float = 0.50             # m³/m³ → score plancher 0,5 (sol saturé)

# Poids du score du jour (somme = 1) ; le réchauffement du sol est un facteur
_WEIGHTS = MappingProxyType({
    "soil_temperature": 0.30,
    "soil_moisture": 0.15,
    "temperature": 0.15,
    "frost": 0.15,
    "precip_day": 0.10,
    "precip_recent": 0.05,
    "humidity": 0.05,
    "wind": 0.05,
})

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


def _local_today(utc_offset_s: float) -> str:
    """Date du jour (ISO) à Paris ; sans base de fuseaux, décalage de la réponse."""
    try:
        return datetime.now(ZoneInfo(_TIMEZONE)).date().isoformat()
    except ZoneInfoNotFoundError:
        now = datetime.now(tz=timezone.utc) + timedelta(seconds=utc_offset_s)
        return now.date().isoformat()


def _age_text(fetched_at: float) -> str:
    """Âge d'un horodatage UTC : 'il y a 3 h' / 'il y a 2 j'."""
    age_h = max(
        0.0, (datetime.now(tz=timezone.utc).timestamp() - fetched_at) / 3600
    )
    if age_h < 48:
        return f"il y a {age_h:.0f} h"
    return f"il y a {age_h / 24:.0f} j"


def _short_reason(exc: BaseException) -> str:
    """Cause d'un échec, sur une ligne courte."""
    return f"{type(exc).__name__} : {exc}"[:160]


def _num(value: float | None) -> str:
    """Nombre arrondi à l'unité, ou 'n/d' s'il manque."""
    return "n/d" if value is None else f"{value:.0f}"


def _complete(values: list[float | None]) -> list[float] | None:
    """Les valeurs si aucune ne manque, sinon None."""
    present = [v for v in values if v is not None]
    return present if len(present) == len(values) else None


def _trapezoid(x: float, rlo: float, lo: float, hi: float, rhi: float) -> float:
    """1 sur [lo, hi], 0 hors ]rlo, rhi[, linéaire entre les deux."""
    if lo <= x <= hi:
        return 1.0
    if x <= rlo or x >= rhi:
        return 0.0
    if x < lo:
        return (x - rlo) / (lo - rlo)
    return (rhi - x) / (rhi - hi)


def _soil_temperature_10cm(
    t6: float | None, t18: float | None
) -> float | None:
    """Température du sol à 10 cm, interpolée entre 6 et 18 cm."""
    if t6 is None:
        return t18
    if t18 is None:
        return t6
    return t6 + (t18 - t6) * (_SOIL_DEPTH_CM - 6.0) / (18.0 - 6.0)


def _daily_means(hourly: dict[str, object], key: str) -> dict[str, float]:
    """Moyenne journalière d'une variable horaire : {'AAAA-MM-JJ': valeur}."""
    times = hourly.get("time")
    values = hourly.get(key)
    if not isinstance(times, list) or not isinstance(values, list):
        return {}
    by_day: dict[str, list[float]] = {}
    for t, v in zip(times, values):
        if v is not None:
            by_day.setdefault(str(t)[:10], []).append(float(v))
    return {d: float(np.mean(vs)) for d, vs in by_day.items()}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DayForecast:
    """Journée Open-Meteo (passée ou prévue) au point MAP_CENTER.

    None = valeur absente de la réponse (jamais remplacée par un défaut).
    """

    date: str
    temperature_max: float | None
    temperature_min: float | None
    temperature_mean: float | None
    precipitation: float | None
    precipitation_probability: float | None
    relativehumidity_mean: float | None
    windspeed_mean: float | None
    soil_temperature: float | None   # °C vers 10 cm (interpolé 6/18 cm)
    soil_moisture: float | None      # m³/m³, 3–9 cm


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
    """Check météo locale pour prospection morilles via Open-Meteo (sans clé)."""

    def __init__(
        self,
        *,
        lat: float | None = None,
        lon: float | None = None,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> None:
        self._lat = lat if lat is not None else float(MAP_CENTER["lat"])
        self._lon = lon if lon is not None else float(MAP_CENTER["lon"])
        self._cache_file = (cache_dir or _CACHE_DIR) / _CACHE_FILE
        self._offline = offline
        self._series: list[DayForecast] = []     # historique + prévisions
        self._today_idx: int = 0
        self._forecasts: list[DayForecast] = []  # aujourd'hui → J+6

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch(self, use_cache: bool = True) -> list[DayForecast]:
        """
        Récupère l'historique et les prévisions ; renvoie les jours
        à partir d'aujourd'hui.

        Ordre : cache de moins d'une heure → API Open-Meteo → hors ligne,
        dernier cache tant qu'il couvre aujourd'hui. ``offline=True``
        n'appelle jamais le réseau.
        """
        cached = self._load_cache()

        if self._offline:
            if cached is None:
                raise RuntimeError("Météo hors ligne : aucun cache Open-Meteo")
            self._use(cached[1])
            logger.info(
                "✅ Météo hors ligne : cache %s (%d jours)",
                _age_text(cached[0]),
                len(self._forecasts),
            )
            return self._forecasts

        if use_cache and cached is not None:
            age = datetime.now(tz=timezone.utc).timestamp() - cached[0]
            if not 0.0 <= age <= _CACHE_MAX_AGE_S:
                logger.debug("   Cache météo expiré ou daté du futur (%.0fs)", age)
            else:
                try:
                    self._use(cached[1])
                except RuntimeError as exc:
                    logger.debug("   Cache météo écarté : %s", exc)
                else:
                    logger.info(
                        "✅ Météo : cache valide (%d jours)",
                        len(self._forecasts),
                    )
                    return self._forecasts

        try:
            raw = self._fetch_api()
            self._use(raw)
        except (requests.RequestException, RuntimeError) as exc:
            if cached is None:
                raise
            try:
                self._use(cached[1])
            except RuntimeError:
                raise exc from None
            logger.warning(
                "⚠️ Météo hors ligne (%s) : prévisions du cache %s",
                _short_reason(exc),
                _age_text(cached[0]),
            )
            return self._forecasts

        self._save_cache(raw)
        logger.info(
            "✅ Météo : %d jours récupérés via Open-Meteo "
            "(+%d jours d'historique, altitude %.0f m)",
            len(self._forecasts),
            self._today_idx,
            _ELEVATION_M,
        )
        return self._forecasts

    def evaluate(self) -> list[ProspectingDay]:
        """Évalue chaque jour (aujourd'hui → J+6) pour la prospection morilles."""
        if not self._forecasts:
            self.fetch()

        results: list[ProspectingDay] = []
        series = self._series

        for i in range(self._today_idx, len(series)):
            fc = series[i]
            score, details = self._score_day(fc, i, series)

            # Détection choc hydrique → fructification
            burst = self._detect_burst(fc, i, series)
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

            # Réchauffement du sol sur 7 jours : printemps ×1, automne ×0,5 ;
            # tendance inconnue = sol stable (jamais mieux qu'une mesure)
            warming = self._soil_warming(i, series)
            factor = self._soil_warming_factor(
                warming if warming is not None else 0.0
            )
            score *= factor
            mult = f"{factor:.2f}".replace(".", ",")
            if warming is None:
                details.append(f"Tendance du sol inconnue (→ ×{mult})")
            else:
                if warming >= 0.5:
                    trend = "Sol qui se réchauffe"
                elif warming <= -0.5:
                    trend = "Sol qui se refroidit"
                else:
                    trend = "Sol stable"
                delta = f"{warming:+.1f}".replace(".", ",")
                details.append(f"{trend} ({delta}°C en 7 j → ×{mult})")

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
            f"  Données météo : Open-Meteo.com (CC BY 4.0), "
            f"altitude {_ELEVATION_M:.0f} m",
            "",
        ]
        for d in days:
            fc = d.forecast
            burst_tag = " 🍄" if d.burst else ""
            lines.append(
                f"  {d.date_fr:<22s}  {d.label}{burst_tag:<22s}  "
                f"score={d.score:.2f}  "
                f"T={_num(fc.temperature_min)}/{_num(fc.temperature_max)}°C  "
                f"P={_num(fc.precipitation)}mm  "
                f"H={_num(fc.relativehumidity_mean)}%  "
                f"V={_num(fc.windspeed_mean)}km/h  "
                f"{WeatherChecker._soil_text(fc)}"
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

    @staticmethod
    def _soil_text(fc: DayForecast) -> str:
        """'Sol=9°C/36%' (température vers 10 cm / eau 3–9 cm)."""
        temp = (
            f"{fc.soil_temperature:.0f}°C"
            if fc.soil_temperature is not None
            else "n/d"
        )
        water = (
            f"{fc.soil_moisture * 100:.0f}%"
            if fc.soil_moisture is not None
            else "n/d"
        )
        return f"Sol={temp}/{water}"

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
        - Une valeur manquante (jour J ou fenêtre) → pas de signal
        """
        if idx < 1:
            return None

        # Jour J doit être sec et assez chaud
        if fc.precipitation is None or fc.temperature_mean is None:
            return None
        if fc.precipitation > _BURST_DRY_THRESH:
            return None
        if fc.temperature_mean < _BURST_TEMP_MIN:
            return None

        # Cumul pluie fenêtre précédente
        win_start = max(0, idx - _BURST_RAIN_WINDOW)
        rain_days_fc = all_fc[win_start:idx]
        if not rain_days_fc:
            return None
        rains = _complete([d.precipitation for d in rain_days_fc])
        temps = _complete([d.temperature_mean for d in rain_days_fc])
        if rains is None or temps is None:
            return None

        rain_cumul = sum(rains)
        if rain_cumul < _BURST_RAIN_MIN:
            return None

        # Au moins 1 jour significatif de pluie dans la fenêtre
        rain_days = sum(1 for p in rains if p > _BURST_DRY_THRESH)
        if rain_days < 1:
            return None

        # Delta thermique : moyenne jour J vs moyenne jours de pluie
        temp_delta = fc.temperature_mean - float(np.mean(temps))

        # Calcul bonus
        intense = rain_cumul >= _BURST_RAIN_INTENSE
        bonus = _BURST_BONUS_INTENSE if intense else _BURST_BONUS

        if temp_delta >= _BURST_TEMP_DELTA_BONUS:
            bonus += _BURST_BONUS_THERMAL

        # Atténuation J+2, J+3 après le choc (premier jour sec = max)
        # Cherche le premier jour sec consécutif
        first_dry = idx
        for j in range(idx - 1, win_start - 1, -1):
            if rains[j - win_start] <= _BURST_DRY_THRESH:
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

        # 1. Température du sol vers 10 cm — pousse vers 8–12 °C
        if fc.soil_temperature is not None:
            st_score = self._score_soil_temperature(fc.soil_temperature)
            components.append((st_score, _WEIGHTS["soil_temperature"]))
            if st_score >= 0.8:
                details.append(
                    f"Sol idéal ({fc.soil_temperature:.0f}°C à 10 cm)"
                )
            elif st_score < 0.3:
                word = (
                    "froid"
                    if fc.soil_temperature < _SOIL_TEMP_OPTIMAL[0]
                    else "chaud"
                )
                details.append(
                    f"Sol {word} ({fc.soil_temperature:.0f}°C à 10 cm)"
                )

        # 2. Humidité du sol 3–9 cm
        if fc.soil_moisture is not None:
            components.append(
                (self._score_soil_moisture(fc.soil_moisture),
                 _WEIGHTS["soil_moisture"])
            )
            lo, hi = _SOIL_MOISTURE_OPTIMAL
            if fc.soil_moisture < lo:
                details.append(
                    f"Sol sec ({fc.soil_moisture * 100:.0f}% d'eau)"
                )
            elif fc.soil_moisture > hi:
                details.append(
                    f"Sol détrempé ({fc.soil_moisture * 100:.0f}% d'eau)"
                )

        # 3. Température de l'air
        if fc.temperature_mean is not None:
            t_score = self._score_temperature(fc.temperature_mean)
            components.append((t_score, _WEIGHTS["temperature"]))
            if t_score < 0.3:
                details.append(
                    f"T° défavorable ({fc.temperature_mean:.0f}°C)"
                )
            elif t_score >= 0.8:
                details.append(f"T° idéale ({fc.temperature_mean:.0f}°C)")

        # 4. Gel nocturne
        if fc.temperature_min is not None:
            frost_score = self._score_frost(fc.temperature_min)
            components.append((frost_score, _WEIGHTS["frost"]))
            if frost_score < 0.5:
                details.append(
                    f"⚠️ Gel nocturne ({fc.temperature_min:.0f}°C)"
                )

        # 5. Précipitations jour — sec = mieux
        if fc.precipitation is not None:
            precip_score = self._score_precip_day(fc.precipitation)
            components.append((precip_score, _WEIGHTS["precip_day"]))
            if fc.precipitation > _PRECIP_DAY_MAX:
                details.append(f"Pluie forte ({fc.precipitation:.0f}mm)")

        # 6. Précipitations récentes J-3..J-1 (historique inclus)
        recent_precip = self._recent_precipitation(idx, all_fc)
        if recent_precip is not None:
            recent_score = self._score_recent_precip(recent_precip)
            components.append((recent_score, _WEIGHTS["precip_recent"]))
            if recent_score >= 0.7:
                details.append(f"Pluie récente ({recent_precip:.0f}mm J-3)")
            elif recent_score < 0.3:
                details.append(f"Peu de pluie ({recent_precip:.0f}mm J-3)")

        # 7. Humidité relative
        if fc.relativehumidity_mean is not None:
            hum_score = self._score_humidity(fc.relativehumidity_mean)
            components.append((hum_score, _WEIGHTS["humidity"]))

        # 8. Vent
        if fc.windspeed_mean is not None:
            wind_score = self._score_wind(fc.windspeed_mean)
            components.append((wind_score, _WEIGHTS["wind"]))
            if fc.windspeed_mean > _WIND_MAX:
                details.append(f"Vent fort ({fc.windspeed_mean:.0f}km/h)")

        # Composite pondéré (valeur absente → poids renormalisés)
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
        rlo, rhi = _TEMP_RANGE
        lo, hi = _TEMP_OPTIMAL
        return _trapezoid(t_mean, rlo, lo, hi, rhi)

    @staticmethod
    def _score_soil_temperature(t_soil: float) -> float:
        rlo, rhi = _SOIL_TEMP_RANGE
        lo, hi = _SOIL_TEMP_OPTIMAL
        return _trapezoid(t_soil, rlo, lo, hi, rhi)

    @staticmethod
    def _soil_warming(idx: int, all_fc: list[DayForecast]) -> float | None:
        """
        Écart (°C) entre le sol moyen de J-6..J et celui de J-13..J-7.

        None si une fenêtre compte moins de _SOIL_WARMING_MIN_DAYS valeurs.
        """
        n = _SOIL_WARMING_WINDOW
        if idx < 2 * n - 1:
            return None
        recent = [
            d.soil_temperature for d in all_fc[idx - n + 1:idx + 1]
            if d.soil_temperature is not None
        ]
        before = [
            d.soil_temperature for d in all_fc[idx - 2 * n + 1:idx - n + 1]
            if d.soil_temperature is not None
        ]
        if min(len(recent), len(before)) < _SOIL_WARMING_MIN_DAYS:
            return None
        return float(np.mean(recent) - np.mean(before))

    @staticmethod
    def _soil_warming_factor(delta: float) -> float:
        """Facteur [plancher, 1] : +1,5 °C en 7 j → ×1 ; −1,5 °C → ×0,5."""
        s = float(
            np.clip(0.5 + delta / (2.0 * _SOIL_WARMING_FULL), 0.0, 1.0)
        )
        return _SOIL_WARMING_FLOOR + (1.0 - _SOIL_WARMING_FLOOR) * s

    @staticmethod
    def _score_soil_moisture(theta: float) -> float:
        lo, hi = _SOIL_MOISTURE_OPTIMAL
        if lo <= theta <= hi:
            return 1.0
        if theta < lo:
            t = (theta - _SOIL_MOISTURE_DRY) / (lo - _SOIL_MOISTURE_DRY)
            return float(np.clip(0.1 + 0.9 * t, 0.1, 1.0))
        t = (theta - hi) / (_SOIL_MOISTURE_WET - hi)
        return float(np.clip(1.0 - 0.5 * t, 0.5, 1.0))

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
    ) -> float | None:
        """Cumul J-3..J-1 (None si une valeur manque)."""
        start = max(0, idx - 3)
        rains = _complete([all_fc[j].precipitation for j in range(start, idx)])
        return sum(rains) if rains is not None else None

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

    def _params(self) -> dict[str, str | float | int]:
        """Paramètres de la requête (aucune clé : usage non commercial)."""
        return {
            "latitude": self._lat,
            "longitude": self._lon,
            "timezone": _TIMEZONE,
            "past_days": _PAST_DAYS,
            "forecast_days": _FORECAST_DAYS,
            "elevation": _ELEVATION_M,
            "daily": ",".join(_DAILY_VARS),
            "hourly": ",".join(_HOURLY_VARS),
        }

    def _fetch_api(self) -> dict[str, object]:
        """Appel Open-Meteo /v1/forecast (réponse validée)."""
        logger.info(
            "🌤️  Open-Meteo → lat=%.4f lon=%.4f alt=%.0f m",
            self._lat,
            self._lon,
            _ELEVATION_M,
        )
        resp = requests.get(
            _OPEN_METEO_URL,
            params=self._params(),
            timeout=15,
        )
        try:
            data = resp.json()
        except ValueError:
            data = None
        # Erreur de paramètre : HTTP 400 + {"error": true, "reason": "..."}
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Open-Meteo : {data.get('reason', 'erreur')}")
        resp.raise_for_status()
        if not isinstance(data, dict):
            raise RuntimeError("Open-Meteo : réponse illisible")
        try:
            self._parse_response(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Open-Meteo : réponse inattendue ({exc})"
            ) from exc
        return data

    def _use(self, raw: dict[str, object]) -> None:
        """Range la série et repère aujourd'hui (RuntimeError s'il manque)."""
        series = self._parse_response(raw)
        offset = raw.get("utc_offset_seconds")
        today = _local_today(
            float(offset) if isinstance(offset, (int, float)) else 0.0
        )
        dates = [d.date for d in series]
        if today not in dates:
            raise RuntimeError(
                f"Météo : pas de données pour aujourd'hui ({today})"
            )
        self._series = series
        self._today_idx = dates.index(today)
        self._forecasts = series[self._today_idx:]

    @staticmethod
    def _parse_response(raw: dict[str, object]) -> list[DayForecast]:
        """Parse la réponse Open-Meteo : journalier + sol horaire."""
        dd = raw.get("daily")
        if not isinstance(dd, dict):
            raise KeyError("Réponse Open-Meteo sans 'daily'")

        times = dd.get("time")
        if not isinstance(times, list):
            raise KeyError("Réponse Open-Meteo sans 'daily.time'")

        def _f(key: str, i: int) -> float | None:
            seq = dd.get(key)
            if isinstance(seq, list) and i < len(seq) and seq[i] is not None:
                return float(seq[i])
            return None

        hourly_raw = raw.get("hourly")
        hourly = hourly_raw if isinstance(hourly_raw, dict) else {}
        soil_t6 = _daily_means(hourly, "soil_temperature_6cm")
        soil_t18 = _daily_means(hourly, "soil_temperature_18cm")
        soil_water = _daily_means(hourly, "soil_moisture_3_to_9cm")

        forecasts: list[DayForecast] = []
        for i, t in enumerate(times):
            day = str(t)[:10]
            forecasts.append(
                DayForecast(
                    date=day,
                    temperature_max=_f("temperature_2m_max", i),
                    temperature_min=_f("temperature_2m_min", i),
                    temperature_mean=_f("temperature_2m_mean", i),
                    precipitation=_f("precipitation_sum", i),
                    precipitation_probability=_f(
                        "precipitation_probability_max", i
                    ),
                    relativehumidity_mean=_f("relative_humidity_2m_mean", i),
                    windspeed_mean=_f("wind_speed_10m_mean", i),
                    soil_temperature=_soil_temperature_10cm(
                        soil_t6.get(day), soil_t18.get(day)
                    ),
                    soil_moisture=soil_water.get(day),
                )
            )

        return forecasts

    def _load_cache(self) -> tuple[float, dict[str, object]] | None:
        """(horodatage, réponse) du cache s'il vaut pour ce point, sinon None."""
        if not self._cache_file.exists():
            return None
        try:
            payload = json.loads(
                self._cache_file.read_text(encoding="utf-8")
            )
            if (
                payload.get("source") != _CACHE_SOURCE
                or payload.get("params") != self._params()
            ):
                logger.debug(
                    "   Cache météo ignoré (autre point ou autres variables)"
                )
                return None
            data = payload["data"]
            if not isinstance(data, dict):
                raise TypeError("'data' n'est pas un objet")
            self._parse_response(data)  # lève si la réponse est mal formée
            return float(payload["fetched_at"]), data
        except (
            OSError, json.JSONDecodeError, KeyError, TypeError, ValueError,
            AttributeError,
        ) as exc:
            logger.warning("⚠️ Cache météo illisible : %s", exc)
            return None

    def _save_cache(self, raw: dict[str, object]) -> None:
        payload = {
            "source": _CACHE_SOURCE,
            "fetched_at": datetime.now(tz=timezone.utc).timestamp(),
            "params": self._params(),
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
    *,
    lat: float | None = None,
    lon: float | None = None,
    offline: bool = False,
) -> list[ProspectingDay]:
    """Fetch + évalue + log le rapport prospection."""
    checker = WeatherChecker(lat=lat, lon=lon, offline=offline)
    checker.fetch()
    days = checker.evaluate()
    report = WeatherChecker.format_report(days)
    logger.info(report)
    return days


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Alertes météo prospection morilles (Open-Meteo)."
    )
    _parser.add_argument(
        "--offline",
        action="store_true",
        help="Sans réseau : dernier cache, s'il couvre aujourd'hui.",
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_weather(offline=_parser.parse_args().offline)
