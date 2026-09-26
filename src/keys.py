"""Token views used as blocking keys.

Each function turns one record into the set of strings that represent it inside
the blocking index. Tokens are namespaced by a short prefix (``a:`` address,
``n:`` name, ``d:`` digit, ``g:`` n-gram, ``c:`` composite) so several views can
share one vocabulary without an address word colliding with a name word that
happens to be spelled the same.

Everything is computed from the record's own text. Nothing consults an external
gazetteer, and no function branches on the country label.
"""

from __future__ import annotations

from .normalize import (
    numeric_keys,
    romanized_address_tokens,
    romanized_tokens,
)

# Address components are comma-separated; the tail of the list carries the
# locality, district and region. Postal codes are absent from almost all
# records, so these tail components are the only geographic anchor available.
LOCALITY_TAIL = 3


def _locality_tokens(address: str) -> set[str]:
    """Tokens drawn from the last few comma-separated address components."""
    parts = [p for p in address.split(",") if p.strip()]
    tail = parts[-LOCALITY_TAIL:] if parts else []
    out: set[str] = set()
    for part in tail:
        out.update(romanized_address_tokens(part))
    return out


def char_ngrams(text: str, sizes: tuple[int, ...] = (3, 4)) -> set[str]:
    """Character n-grams over a whitespace-collapsed string.

    These are what bridge spellings no token view can align: a name written as a
    bare domain (``galaxytechprivate.com``) shares no whole token with
    ``Galaxy Tech Private Limited`` but shares most of its trigrams.
    """
    squeezed = "".join(text.split())
    grams: set[str] = set()
    for size in sizes:
        if len(squeezed) >= size:
            grams.update(squeezed[i : i + size] for i in range(len(squeezed) - size + 1))
    return grams


def address_view(name: str, address: str) -> set[str]:
    """Address words and numbers only."""
    tokens = romanized_address_tokens(address)
    view = {f"a:{t}" for t in tokens}
    view.update(f"d:{t}" for t in numeric_keys(tokens))
    return view


def name_view(name: str, address: str) -> set[str]:
    """Name words only, legal forms stripped."""
    return {f"n:{t}" for t in romanized_tokens(name)}


def name_ngram_view(name: str, address: str) -> set[str]:
    """Name character n-grams, for typos, truncations and run-together spellings."""
    core = " ".join(romanized_tokens(name))
    return {f"g:{g}" for g in char_ngrams(core)}


def combined_view(name: str, address: str) -> set[str]:
    """Name and address words in one space.

    The default. Address carries most of the signal, but a record with an empty
    address still retrieves on its name, which is the only route for the ~3% of
    Source 2/3 records that have no address at all.
    """
    view = address_view(name, address)
    view.update(name_view(name, address))
    return view


def composite_view(name: str, address: str) -> set[str]:
    """Combined words plus digit-anchored composite keys.

    A house or plot number alone is common; a number paired with a locality word
    is close to unique, and both halves survive reordering, abbreviation and
    transliteration. Pairing every digit token with every locality token costs a
    handful of extra keys per record and sharpens the ranking considerably.
    """
    tokens = romanized_address_tokens(address)
    view = combined_view(name, address)
    digits = sorted(numeric_keys(tokens))
    locality = sorted(_locality_tokens(address))
    for digit in digits[:4]:
        for place in locality[:6]:
            view.add(f"c:{digit}|{place}")
    return view


def full_view(name: str, address: str) -> set[str]:
    """Composite keys plus name n-grams: the widest view we evaluate."""
    view = composite_view(name, address)
    view.update(name_ngram_view(name, address))
    return view


def addr_ngram_view(name: str, address: str) -> set[str]:
    """Address character n-grams, for transliterated locality and street names."""
    core = " ".join(romanized_address_tokens(address))
    return {f"h:{g}" for g in char_ngrams(core)}


VIEWS = {
    "address": address_view,
    "name": name_view,
    "ngram": name_ngram_view,
    "addr_ngram": addr_ngram_view,
    "combined": combined_view,
    "composite": composite_view,
    "full": full_view,
}
