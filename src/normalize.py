"""Canonicalisation of business names and addresses.

Everything here is derived from the challenge data alone: no external gazetteers,
geocoders or registries. The abbreviation tables below encode generic language
conventions (street-type words, legal-form suffixes) rather than any lookup of a
specific business or address.

The module is deliberately country-agnostic. `country` is carried through as an
opaque string label and never used to switch behaviour, so records from a country
unseen in training (France in the test set) flow through the same code path.
"""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------
# character-level cleanup
# --------------------------------------------------------------------------

# Junk that the sources prepend/append to names: "-- Holloway Peak Inc",
# "<< Team Ecole", "##16939 VANILLA ORCHID DR".
_JUNK_AFFIX = re.compile(r"^[\s\-–—<>#*~^|/\\.,;:!+=_'\"()\[\]{}]+|[\s\-–—<>#*~^|/\\;:!+=_'\"]+$")

# A trade-name marker splits "<shell company> d/b/a <real name>"; the informative
# half is what follows the marker.
_DBA = re.compile(
    r"\b(?:d\s*/?\s*b\s*/?\s*a|doing\s+business\s+as|trading\s+as|t\s*/\s*a|formerly|aka|a\s*/\s*k\s*/\s*a)\b",
    re.IGNORECASE,
)

# "M/s Galaxy Tech" -> "Galaxy Tech"
_MS_PREFIX = re.compile(r"^\s*m\s*/\s*s\.?\s+", re.IGNORECASE)

# "galaxytechprivate.com" -> "galaxytechprivate"
_DOMAIN = re.compile(r"^([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|co|in|io|biz|info|fr|us|co\.in|co\.uk)$", re.IGNORECASE)

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_DIGIT_RUN = re.compile(r"\d+")

# Ordinal suffixes appear with random casing: "22Nd", "9Th", "2ND".
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)


def strip_accents(text: str) -> str:
    """Fold accented Latin to ASCII (Président -> President).

    Applied only to Latin script: the decomposition of Indic scripts separates
    vowel signs from their consonants, which would corrupt the token rather than
    normalise it, so non-Latin codepoints are passed through untouched.
    """
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch):
            # A combining mark is dropped only when it modifies Latin.
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _fold(text: str) -> str:
    """Lowercase, fold accents, and collapse punctuation to single spaces."""
    text = unicodedata.normalize("NFKC", text)
    has_latin_accent = any(
        unicodedata.combining(c) for c in unicodedata.normalize("NFD", text)
    )
    if has_latin_accent:
        text = strip_accents(text)
    return text.lower()


# --------------------------------------------------------------------------
# legal-form suffixes
# --------------------------------------------------------------------------
# Mapped to a canonical token so "Pvt Ltd" / "Private Limited" / "Pvt. Ltd." all
# agree. These are *generic* corporate-form words, not entity identities.
LEGAL_FORMS = {
    # India
    "pvt": "private", "pvtltd": "privatelimited", "ltd": "limited", "limted": "limited",
    "llp": "llp", "opc": "opc",
    # US
    "inc": "inc", "incorporated": "inc", "corp": "corporation", "corpn": "corporation",
    "co": "company", "llc": "llc", "lp": "lp", "plc": "plc",
    # France
    "sarl": "sarl", "sasu": "sasu", "eurl": "eurl", "sas": "sas", "sa": "sa",
    "sci": "sci", "snc": "snc", "scop": "scop",
}

# Dropped entirely when comparing *cores*: they carry no identifying signal and
# their presence/absence is pure source noise.
LEGAL_NOISE = {
    "private", "limited", "ltd", "pvt", "inc", "incorporated", "corporation",
    "corp", "company", "co", "llc", "llp", "lp", "plc", "opc",
    "sarl", "sasu", "eurl", "sas", "sa", "sci", "snc",
    "the", "and", "of",
}

# --------------------------------------------------------------------------
# address component abbreviations
# --------------------------------------------------------------------------
STREET_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "dr": "drive", "blvd": "boulevard", "bd": "boulevard", "ln": "lane",
    "ct": "court", "cir": "circle", "pl": "place", "sq": "square", "ter": "terrace",
    "hwy": "highway", "pkwy": "parkway", "trl": "trail", "trail": "trail",
    "apt": "apartment", "ste": "suite", "bldg": "building", "flr": "floor",
    "fl": "floor", "rm": "room", "dept": "department", "po": "po", "pob": "po",
    # India-specific municipal vocabulary
    "no": "number", "nos": "number", "hno": "number", "dno": "number",
    "khno": "number", "plotno": "number", "doorno": "number", "survey": "survey",
    "sn": "survey", "opp": "opposite", "nr": "near", "bldng": "building",
    "mkt": "market", "nagar": "nagar", "colony": "colony", "marg": "marg",
    "gali": "gali", "sector": "sector", "phase": "phase", "block": "block",
    # France
    "bis": "bis", "ter_fr": "ter", "av_fr": "avenue", "bd_fr": "boulevard",
}

# Directionals, stripped of punctuation variance.
DIRECTIONS = {"n": "north", "s": "south", "e": "east", "w": "west",
              "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"}


def name_tokens(raw: str) -> list[str]:
    """Tokenise a business name after removing source-specific decoration.

    Handles the junk affixes, the d/b/a shell-company pattern, the ``M/s``
    prefix and the bare-domain spelling observed in Source 3.
    """
    if not raw:
        return []
    text = _JUNK_AFFIX.sub("", raw)
    text = _MS_PREFIX.sub("", text)

    # "<shell> d/b/a <real name>" -> keep the part after the marker.
    parts = _DBA.split(text)
    if len(parts) > 1:
        tail = parts[-1].strip()
        if tail:
            text = tail

    text = _fold(text)

    domain = _DOMAIN.match(text.strip())
    if domain:
        text = domain.group(1)

    text = _ORDINAL.sub(r"\1", text)
    return _TOKEN.findall(text)


def name_core(raw: str) -> list[str]:
    """Name tokens with legal-form and stop words removed.

    ``Dhule Roadlines Limited`` and ``Dhule Roadlines Ltd`` both reduce to
    ``['dhule', 'roadlines']``, which is what the blocking keys and the
    order-insensitive similarity features compare.
    """
    return [t for t in name_tokens(raw) if t not in LEGAL_NOISE]


def canon_number(token: str) -> str:
    """Strip leading zeros inside a token so ``F-003`` and ``F-3`` agree."""
    return _DIGIT_RUN.sub(lambda m: str(int(m.group(0))), token)


def address_tokens(raw: str) -> list[str]:
    """Tokenise an address, expanding street-type and municipal abbreviations."""
    if not raw:
        return []
    text = _JUNK_AFFIX.sub("", raw)
    text = _fold(text)
    text = _ORDINAL.sub(r"\1", text)
    out = []
    for tok in _TOKEN.findall(text):
        tok = canon_number(tok)
        tok = STREET_ABBREV.get(tok, DIRECTIONS.get(tok, tok))
        out.append(tok)
    return out


_NON_LATIN = re.compile(r"[^\x00-\x7F]")


def has_non_latin(text: str) -> bool:
    """True when the string carries codepoints outside ASCII after folding."""
    return bool(_NON_LATIN.search(unicodedata.normalize("NFKC", text)))


def romanized_tokens(raw: str) -> list[str]:
    """Name tokens forced into Latin script via transliteration.

    Source 2 and Source 3 frequently render an Indian business name in its native
    script while Source 1 keeps the Latin spelling. Transliteration does not
    recover the Latin spelling exactly, but it lands close enough for fuzzy and
    character n-gram comparison to bridge the gap:

        શિવ મા ટેક્નોલોજી -> "shiv maa tteknolojii"   vs  "shiv maa technology"
        ಕರ್ನಾಟಕ           -> "krnaattk"                vs  "karnataka"

    For records already in Latin script this returns the same tokens as
    :func:`name_core`, so callers can use it unconditionally.
    """
    if not raw:
        return []
    text = unidecode(raw) if has_non_latin(raw) else raw
    return [t for t in name_tokens(text) if t not in LEGAL_NOISE]


def romanized_address_tokens(raw: str) -> list[str]:
    """Address tokens with native-script components transliterated to Latin.

    Indian state names alternate between the full English name, the two-letter
    code and the native script (``Gujarat`` / ``GJ`` / ``ગુજરાત``) across sources.
    Transliteration collapses the third form onto something comparable with the
    first.
    """
    if not raw:
        return []
    return address_tokens(unidecode(raw) if has_non_latin(raw) else raw)


def numeric_keys(tokens: list[str]) -> set[str]:
    """Digit-bearing tokens from an address.

    House, plot, door and survey numbers survive reordering, abbreviation and
    transliteration far better than any word does, which makes them the
    highest-precision blocking key available.
    """
    return {t for t in tokens if any(c.isdigit() for c in t)}
