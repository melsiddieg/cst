"""Syllogism generator (Category 2), bilingual.

Produces categorical syllogisms of the form

    Major premise: All M are P.
    Minor premise: All S are M.
    Conclusion:    All S are P.

Valid moods covered: Barbara, Celarent, Darii, Ferio (Figure 1).
Invalid controls are included so the downstream model must learn to
reject bad inferences, not memorize "syllogism → yes".

Run::

    python -m reasoning.data.generators.syllogisms --count 5000 --out out/syllogisms.jsonl
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..schema import Meta, Record, write_jsonl


# Content categories (S, M, P) — deliberately abstract so the logic,
# not world-knowledge, drives the inference.
CATEGORIES_EN = [
    ("birds", "animals", "living things"),
    ("squares", "rectangles", "quadrilaterals"),
    ("programmers", "engineers", "professionals"),
    ("poets", "writers", "artists"),
    ("apples", "fruits", "plants"),
]

CATEGORIES_AR = [
    ("الطيور", "الحيوانات", "الكائنات الحية"),
    ("المربعات", "المستطيلات", "الأشكال الرباعية"),
    ("المبرمجون", "المهندسون", "المحترفون"),
    ("الشعراء", "الكتّاب", "الفنانون"),
    ("التفاح", "الفواكه", "النباتات"),
]


@dataclass
class Mood:
    name: str
    premise_major: str   # template over (M, P)
    premise_minor: str   # template over (S, M)
    conclusion: str      # template over (S, P)
    valid: bool


# Figure-1 valid + undistributed-middle invalid controls (easy / medium)
MOODS_EN = [
    Mood("Barbara", "All {M} are {P}.", "All {S} are {M}.", "All {S} are {P}.", True),
    Mood("Celarent", "No {M} are {P}.", "All {S} are {M}.", "No {S} are {P}.", True),
    Mood("Darii",   "All {M} are {P}.", "Some {S} are {M}.", "Some {S} are {P}.", True),
    Mood("Ferio",   "No {M} are {P}.", "Some {S} are {M}.", "Some {S} are not {P}.", True),
    # Invalid controls (undistributed middle, illicit major, …)
    Mood("Invalid-1", "All {P} are {M}.", "All {S} are {M}.", "All {S} are {P}.", False),
    Mood("Invalid-2", "Some {M} are {P}.", "Some {S} are {M}.", "Some {S} are {P}.", False),
]

MOODS_AR = [
    Mood("Barbara",
         "كل {M} هي {P}.", "كل {S} هي {M}.", "كل {S} هي {P}.", True),
    Mood("Celarent",
         "لا شيء من {M} هو {P}.", "كل {S} هي {M}.", "لا شيء من {S} هو {P}.", True),
    Mood("Darii",
         "كل {M} هي {P}.", "بعض {S} هي {M}.", "بعض {S} هي {P}.", True),
    Mood("Ferio",
         "لا شيء من {M} هو {P}.", "بعض {S} هي {M}.", "بعض {S} ليست {P}.", True),
    Mood("Invalid-1",
         "كل {P} هي {M}.", "كل {S} هي {M}.", "كل {S} هي {P}.", False),
    Mood("Invalid-2",
         "بعض {M} هي {P}.", "بعض {S} هي {M}.", "بعض {S} هي {P}.", False),
]

# Figure-2 moods: middle term is predicate in both premises, which makes
# distribution analysis non-obvious. Plus an illicit-major invalid
# control. These are tagged ``hard``.
MOODS_HARD_EN = [
    Mood("Camestres", "All {P} are {M}.", "No {S} are {M}.", "No {S} are {P}.", True),
    Mood("Baroco",    "All {P} are {M}.", "Some {S} are not {M}.", "Some {S} are not {P}.", True),
    Mood("Cesare",    "No {P} are {M}.", "All {S} are {M}.", "No {S} are {P}.", True),
    Mood("Festino",   "No {P} are {M}.", "Some {S} are {M}.", "Some {S} are not {P}.", True),
    # Invalid: illicit major (P distributed in conclusion but not premise)
    Mood("Invalid-Hard-1",
         "All {M} are {P}.", "Some {S} are not {M}.", "Some {S} are not {P}.", False),
    # Invalid: exclusive premises (two negatives yield no conclusion)
    Mood("Invalid-Hard-2",
         "No {M} are {P}.", "No {S} are {M}.", "No {S} are {P}.", False),
]

MOODS_HARD_AR = [
    Mood("Camestres",
         "كل {P} هي {M}.", "لا شيء من {S} هو {M}.", "لا شيء من {S} هو {P}.", True),
    Mood("Baroco",
         "كل {P} هي {M}.", "بعض {S} ليست {M}.", "بعض {S} ليست {P}.", True),
    Mood("Cesare",
         "لا شيء من {P} هو {M}.", "كل {S} هي {M}.", "لا شيء من {S} هو {P}.", True),
    Mood("Festino",
         "لا شيء من {P} هو {M}.", "بعض {S} هي {M}.", "بعض {S} ليست {P}.", True),
    Mood("Invalid-Hard-1",
         "كل {M} هي {P}.", "بعض {S} ليست {M}.", "بعض {S} ليست {P}.", False),
    Mood("Invalid-Hard-2",
         "لا شيء من {M} هو {P}.", "لا شيء من {S} هو {M}.", "لا شيء من {S} هو {P}.", False),
]


def _fill(mood: Mood, cats: tuple[str, str, str]) -> tuple[str, str, str]:
    S, M, P = cats
    return (
        mood.premise_major.format(M=M, P=P),
        mood.premise_minor.format(S=S, M=M),
        mood.conclusion.format(S=S, P=P),
    )


def _cot(mood: Mood, cats: tuple[str, str, str], lang: str) -> list[str]:
    S, M, P = cats
    if lang == "en":
        if mood.valid:
            return [
                f"Identify terms: S={S}, M={M}, P={P}.",
                f"Major premise links M and P.",
                f"Minor premise links S and M.",
                f"Middle term M is distributed; inference is valid.",
            ]
        return [
            f"Identify terms: S={S}, M={M}, P={P}.",
            f"Middle term M is not distributed in the premises.",
            f"Classical rules forbid the conclusion.",
        ]
    # Arabic
    if mood.valid:
        return [
            f"حدد الحدود: S={S} ، M={M} ، P={P}.",
            "المقدمة الكبرى تربط M و P.",
            "المقدمة الصغرى تربط S و M.",
            "الحد الأوسط M موزّع، فالاستنتاج صحيح.",
        ]
    return [
        f"حدد الحدود: S={S} ، M={M} ، P={P}.",
        "الحد الأوسط M غير موزّع في المقدمات.",
        "القواعد الكلاسيكية تمنع هذا الاستنتاج.",
    ]


def _record(
    *, idx: int, lang: str, mood: Mood, cats: tuple[str, str, str],
    difficulty: str,
) -> Record:
    major, minor, conc = _fill(mood, cats)
    if lang == "en":
        question = f"{major} {minor} Does it follow that: {conc}"
        answer = "yes" if mood.valid else "no"
    else:
        question = f"{major} {minor} هل يلزم أن: {conc}"
        answer = "نعم" if mood.valid else "لا"
    return Record(
        id=f"syllog-{lang}-{idx:06d}",
        lang=lang,  # type: ignore[arg-type]
        category=2,
        question=question,
        answer=answer,
        cot=_cot(mood, cats, lang),
        meta=Meta(
            source="syllogisms",
            license="cc0-1.0",
            difficulty=difficulty,  # type: ignore[arg-type]
        ),
    )


def _sorites_cats(rng: random.Random) -> tuple[list[str], list[str]]:
    """Pick 4 distinct category chains (EN, AR) for a sorites."""
    idxs = rng.sample(range(len(CATEGORIES_EN)), 2)
    # Build a 4-term chain by concatenating two 3-term chains on their
    # shared super-category. Here we synthesise a fresh chain instead.
    en = list(CATEGORIES_EN[idxs[0]]) + [CATEGORIES_EN[idxs[1]][-1]]
    ar = list(CATEGORIES_AR[idxs[0]]) + [CATEGORIES_AR[idxs[1]][-1]]
    return en, ar


def _sorites_record(
    *, idx: int, lang: str, chain: list[str], valid: bool,
) -> Record:
    """4-term sorites: A⊂B, B⊂C, C⊂D ⊢ A⊂D (valid); swap one link for invalid."""
    A, B, C, D = chain
    if not valid:
        # Break transitivity by reversing the middle link.
        B, C = C, B
    if lang == "en":
        p1 = f"All {A} are {B}."
        p2 = f"All {B} are {C}."
        p3 = f"All {C} are {D}."
        conc = f"All {A} are {D}."
        question = f"{p1} {p2} {p3} Does it follow that: {conc}"
        answer = "yes" if valid else "no"
        cot = [
            f"Chain premises: {A} ⊆ {B}, {B} ⊆ {C}, {C} ⊆ {D}."
            if valid else
            f"Chain premises: {A} ⊆ {B}, {B} ⊆ {C}, {C} ⊆ {D} — broken order.",
            "Transitivity of universal inclusion carries the subject through each link."
            if valid else
            "Middle link runs the wrong way, so the chain does not compose.",
            f"Therefore {A} ⊆ {D}." if valid else "Conclusion does not follow.",
        ]
    else:
        p1 = f"كل {A} هي {B}."
        p2 = f"كل {B} هي {C}."
        p3 = f"كل {C} هي {D}."
        conc = f"كل {A} هي {D}."
        question = f"{p1} {p2} {p3} هل يلزم أن: {conc}"
        answer = "نعم" if valid else "لا"
        cot = [
            f"سلسلة المقدمات: {A} ⊆ {B} ، {B} ⊆ {C} ، {C} ⊆ {D}."
            if valid else
            f"سلسلة المقدمات: {A} ⊆ {B} ، {B} ⊆ {C} ، {C} ⊆ {D} — ترتيب مكسور.",
            "تعدّي الاحتواء الكلي ينقل الموضوع عبر كل حلقة."
            if valid else
            "الحلقة الوسطى معكوسة، فلا تتركّب السلسلة.",
            f"إذًا {A} ⊆ {D}." if valid else "الاستنتاج لا يلزم.",
        ]
    return Record(
        id=f"syllog-sorites-{lang}-{idx:06d}",
        lang=lang,  # type: ignore[arg-type]
        category=2,
        question=question,
        answer=answer,
        cot=cot,
        meta=Meta(
            source="syllogisms",
            license="cc0-1.0",
            difficulty="hard",  # type: ignore[arg-type]
        ),
    )


def generate(count: int, *, seed: int = 42) -> Iterable[Record]:
    """Yield ``2 * count`` records (one EN + one AR per sample).

    Difficulty mix roughly follows §11.2 of REASONING_DATA.md:
    70% easy/medium (Figure-1 + controls), 20% hard (Figure-2),
    10% hard (4-term sorites).
    """
    rng = random.Random(seed)
    for i in range(count):
        r = rng.random()
        if r < 0.70:
            # Figure-1: easy when valid, medium when invalid control.
            j = rng.randrange(len(MOODS_EN))
            cats_en = rng.choice(CATEGORIES_EN)
            cats_ar = CATEGORIES_AR[CATEGORIES_EN.index(cats_en)]
            mood_en, mood_ar = MOODS_EN[j], MOODS_AR[j]
            difficulty = "easy" if mood_en.valid else "medium"
            yield _record(idx=i, lang="en", mood=mood_en, cats=cats_en,
                          difficulty=difficulty)
            yield _record(idx=i, lang="ar", mood=mood_ar, cats=cats_ar,
                          difficulty=difficulty)
        elif r < 0.90:
            # Figure-2 and illicit-major controls — all hard.
            j = rng.randrange(len(MOODS_HARD_EN))
            cats_en = rng.choice(CATEGORIES_EN)
            cats_ar = CATEGORIES_AR[CATEGORIES_EN.index(cats_en)]
            mood_en, mood_ar = MOODS_HARD_EN[j], MOODS_HARD_AR[j]
            yield _record(idx=i, lang="en", mood=mood_en, cats=cats_en,
                          difficulty="hard")
            yield _record(idx=i, lang="ar", mood=mood_ar, cats=cats_ar,
                          difficulty="hard")
        else:
            # 4-term sorites — hard.
            chain_en, chain_ar = _sorites_cats(rng)
            valid = rng.random() < 0.5
            yield _sorites_record(idx=i, lang="en", chain=chain_en, valid=valid)
            yield _sorites_record(idx=i, lang="ar", chain=chain_ar, valid=valid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n = write_jsonl(args.out, generate(args.count, seed=args.seed))
    print(f"Wrote {n:,} records to {args.out}")


if __name__ == "__main__":
    main()
