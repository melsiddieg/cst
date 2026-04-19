"""
Arabic CST Experiment — Download, tokenize (CST + SentencePiece BPE), compare.

This script:
  1. Downloads 100K Arabic sentences from Arabic Wikipedia via HuggingFace API
  2. Extracts trilateral roots using camel-tools morphological analyzer
  3. Maps roots to 55 semantic fields (same fields as English CST)
     — handles weak roots (# wildcards) from camel-tools
     — strips proclitics for better coverage
  4. Produces CST-tokenized .jsonl
  5. Trains SentencePiece BPE at 8K and 32K and produces BPE-tokenized .jsonl
  6. Prints comparison stats

Usage:
  python training/arabic_experiment.py [--sentences 100000]
"""

import argparse
import json
import os
import re
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

# ── Arabic root → semantic field mapping ─────────────────────────
# Maps dotted root notation (e.g. "ك.ت.ب") to one of 55 CST semantic fields.
# INCLUDES weak-root variants with # (as camel-tools returns them).

ARABIC_ROOT_TO_FIELD: dict[str, str] = {}

def _add(field: str, *roots: str):
    for r in roots:
        ARABIC_ROOT_TO_FIELD[r] = field

# ── write / record ──
_add("write",
    "ك.ت.ب", "خ.ط.ط", "س.ج.ل", "د.و.ن", "ر.ق.م", "ن.س.خ",
    "ط.ب.ع", "ن.ش.ر", "ص.د.ر", "و.ث.ق", "ص.ح.ف",
    "د.#.ن",  # weak for دون
)

# ── know / learn ──
_add("know",
    "ع.ل.م", "ع.ر.ف", "د.ر.س", "ف.ه.م", "ث.ق.ف", "خ.ب.ر",
    "ف.ق.ه", "ب.ح.ث", "ر.ش.د", "ل.ق.ن", "و.ع.ي", "ح.ف.ظ",
    "ع.ل.#", "ع.#.م",  # weak variants
)

# ── speak / say ──
_add("speak",
    "ق.و.ل", "ك.ل.م", "ح.د.ث", "ن.ط.ق", "خ.ط.ب", "ص.ر.خ",
    "ن.د.ي", "ل.غ.و", "ح.ك.ي", "ع.ل.ن", "ذ.ك.ر", "ر.و.ي",
    "س.أ.ل", "ج.و.ب", "ف.س.ر", "و.ص.ف", "ب.ي.ن", "ش.ر.ح",
    "ق.#.ل", "ح.#.ث", "ب.#.ن",  # weak variants
)

# ── think / reason ──
_add("think",
    "ف.ك.ر", "ع.ق.ل", "ر.أ.ي", "ظ.ن.ن", "ح.س.ب", "ن.ظ.ر",
    "خ.م.ن", "ق.ر.ر", "ز.ع.م",
    "ر.#.ي",  # weak for رأي
)

# ── see / perceive ──
_add("see",
    "ب.ص.ر", "ش.ه.د", "ل.ح.ظ", "ل.م.ح", "ر.ق.ب", "ت.ب.ع",
    "ر.ص.د",
)

# ── feel / emotion ──
_add("feel",
    "ح.ب.ب", "ش.ع.ر", "ح.ز.ن", "ف.ر.ح", "خ.و.ف", "غ.ض.ب",
    "ق.ل.ق", "ر.ض.ي", "أ.م.ل", "ن.د.م", "أ.ل.م", "س.ع.د",
    "ح.ن.ن", "ع.ش.ق", "ك.ر.ه", "ح.ي.ر", "ذ.ع.ر", "ف.ز.ع",
)

# ── move / motion ──
_add("move",
    "ح.ر.ك", "م.ش.ي", "ذ.ه.ب", "ج.ر.ي", "س.ي.ر", "ر.ح.ل",
    "س.ف.ر", "ع.ب.ر", "و.ص.ل", "ر.ج.ع", "ه.ر.ب", "ق.د.م",
    "ن.ز.ل", "ص.ع.د", "ط.ل.ع", "د.خ.ل", "خ.ر.ج", "ه.ب.ط",
    "ط.ي.ر", "ق.ف.ز", "ز.ح.ف", "س.ب.ح", "ر.ك.ب", "ن.ق.ل",
    "ح.م.ل", "ج.ل.ب", "م.ر.ر",
    "د.#.ل", "خ.#.ر",  # weak: دخل، خروج
)

# ── force / power ──
_add("force",
    "ق.و.ي", "ض.غ.ط", "د.ف.ع", "ج.ذ.ب", "س.ح.ب", "ش.د.د",
    "ض.ر.ب", "ر.م.ي", "ق.ذ.ف", "ه.ز.ز", "ز.ل.ز",
    "ق.#.ي",  # weak for قوي
)

# ── fight / conflict ──
_add("fight",
    "ح.ر.ب", "ق.ت.ل", "ج.ه.د", "ه.ج.م", "ن.ز.ع", "ص.ر.ع",
    "غ.ز.و", "ف.ت.ح", "ح.ص.ر", "ق.ه.ر", "ع.د.و",
    "ع.#.د",  # weak
)

# ── make / create ──
_add("make",
    "ص.ن.ع", "ب.ن.ي", "ش.ك.ل", "ك.و.ن", "أ.س.س", "ر.ك.ب",
    "ن.ت.ج", "ب.#.ن",
    "ك.#.ن",  # weak for كون (extremely common!)
)

_add("create",
    "خ.ل.ق", "و.ل.د", "ن.ش.أ", "ص.م.م", "ر.س.م", "ب.د.ع",
)

# ── destroy / break ──
_add("destroy",
    "ه.د.م", "ك.س.ر", "خ.ر.ب", "ح.ر.ق", "ف.ج.ر", "ح.ط.م",
    "م.ح.و", "ت.ل.ف", "د.م.ر", "ن.ق.ض",
)

# ── give / receive ──
_add("give",
    "ع.ط.ي", "م.ن.ح", "و.ه.ب", "ت.ب.ر", "أ.خ.ذ", "ق.ب.ل",
    "ت.ل.ق", "ب.ذ.ل", "ص.ر.ف",
    "ع.#.ط",  # weak
)

# ── trade / exchange ──
_add("trade",
    "ت.ج.ر", "ب.ي.ع", "ش.ر.ي", "ر.ب.ح", "خ.س.ر", "ق.ي.م",
    "ث.م.ن", "ن.ف.ق", "م.و.ل", "ك.ل.ف", "ك.س.ب",
    "ق.#.م",  # weak for قيمة (value — extremely common)
)

# ── possess / own ──
_add("possess",
    "م.ل.ك", "ح.و.ز", "خ.ص.ص", "إ.ر.ث", "أ.م.ن",
)

# ── govern / rule ──
_add("govern",
    "ح.ك.م", "س.ي.س", "ق.ا.د", "أ.م.ر", "ن.ظ.م", "ش.ر.ع",
    "ق.ن.ن", "و.ل.ي", "ر.ئ.س", "س.ل.ط", "ن.خ.ب", "ص.و.ت",
    "ج.م.ه", "م.ث.ل",
    "ق.#.د",  # weak for قاد
)

# ── work / labor ──
_add("work",
    "ع.م.ل", "ش.غ.ل", "و.ظ.ف", "م.ه.ن", "ك.د.ح", "ن.ش.ط",
    "خ.د.م", "أ.د.ي", "ت.ع.ب", "إ.ن.ج",
)

# ── exist / be ──
_add("exist",
    "و.ج.د", "ح.ي.ي", "م.و.ت", "ب.ق.ي", "ع.ي.ش",
    "ف.ن.ي", "ز.و.ل", "د.و.م", "ح.ض.ر",
    "#.ج.د",  # weak for وجد (extremely common!)
    "ح.#.ي",  # weak for حياة
)

# ── change / transform ──
_add("change",
    "غ.ي.ر", "ح.و.ل", "ب.د.ل", "ت.ط.و", "ن.م.و", "ز.ي.د",
    "ن.ق.ص", "ص.ل.ح", "ج.د.د",
    "غ.#.ر", "ح.#.ل", "ت.م.م",  # weak variants — حول is very common
)

# ── time / duration ──
_add("time",
    "و.ق.ت", "ز.م.ن", "ع.ص.ر", "ت.ر.خ", "ق.ر.ن", "ع.ه.د",
    "م.د.ي", "أ.ج.ل", "ب.ك.ر", "أ.خ.ر", "ب.د.أ", "خ.ت.م",
    "ن.ه.ي", "د.و.ر",
    "#.خ.ر", "ب.د.#", "م.د.د",  # weak — أخر very common
)

# ── place / location ──
_add("place",
    "م.ك.ن", "و.ض.ع", "ق.ع.د", "ج.ل.س", "س.ك.ن", "ب.ل.د",
    "و.ط.ن", "م.د.ن", "ق.ر.ي", "م.ن.ط", "إ.ق.ل", "ج.ه.ة",
    "ش.م.ل", "ج.ن.ب", "غ.ر.ب", "ش.ر.ق",
)

# ── body / physical ──
_add("body",
    "ج.س.د", "ق.ل.ب", "ر.أ.س", "ي.د.ي", "ع.ي.ن", "و.ج.ه",
    "ل.س.ن", "أ.ذ.ن", "ص.د.ر", "ب.ط.ن", "ج.ل.د", "ع.ظ.م",
    "د.م.م", "ل.ح.م",
    "ر.#.س",  # weak for رأس
)

# ── health / medicine ──
_add("health",
    "ص.ح.ح", "م.ر.ض", "ط.ب.ب", "ع.ل.ج", "ش.ف.ي", "و.ب.أ",
    "ج.ر.ح", "د.و.ي",
)

# ── consume / food ──
_add("consume",
    "أ.ك.ل", "ش.ر.ب", "ط.ع.م", "ط.ب.خ", "ج.و.ع", "ع.ط.ش",
    "ذ.و.ق", "ه.ض.م", "غ.ذ.ي",
    "#.ك.ل",  # weak for أكل
)

# ── nature / earth ──
_add("nature",
    "ط.ب.ع", "أ.ر.ض", "ب.ح.ر", "ن.ه.ر", "ج.ب.ل", "ب.ر.ر",
    "ص.ح.ر", "غ.ا.ب", "و.ا.د", "س.ه.ل",
    "#.ر.ض", "ر.#.ض",  # weak for أرض — very common
)

# ── weather / climate ──
_add("weather",
    "م.ط.ر", "ر.ي.ح", "ث.ل.ج", "ح.ر.ر", "ب.ر.د", "ش.م.س",
    "غ.ي.م", "ع.ص.ف", "ف.ي.ض", "ج.ف.ف",
)

# ── animal ──
_add("animal",
    "ح.ي.و", "ط.ي.ر", "س.م.ك", "ح.ش.ر", "ذ.ئ.ب", "أ.س.د",
    "ف.ر.س", "ب.ق.ر", "غ.ن.م", "ج.م.ل", "ك.ل.ب",
)

# ── plant ──
_add("plant",
    "ز.ر.ع", "ن.ب.ت", "ش.ج.ر", "ث.م.ر", "ز.ه.ر", "ح.ص.د",
    "غ.ر.س", "ر.و.ض",
)

# ── color ──
_add("color",
    "ل.و.ن", "ب.ي.ض", "س.و.د", "ح.م.ر", "خ.ض.ر", "ز.ر.ق", "ص.ف.ر",
)

# ── size / measure ──
_add("size",
    "ك.ب.ر", "ص.غ.ر", "ط.و.ل", "ق.ص.ر", "ع.ر.ض", "و.س.ع",
    "ض.ي.ق", "ع.م.ق", "ك.ث.ر", "ق.ل.ل",
)

# ── measure / quantity ──
_add("measure",
    "ق.ي.س", "و.ز.ن", "ع.د.د", "ح.س.ب", "م.س.ح", "ب.ع.د",
    "ق.ر.ب", "ن.ص.ف",
    "ع.#.د",  # weak for عدد
)

# ── connect / join ──
_add("connect",
    "و.ص.ل", "ر.ب.ط", "ج.م.ع", "ض.م.م", "ل.ح.م", "ش.ب.ك",
    "ع.ل.ق", "ز.و.ج",
)

# ── contain / hold ──
_add("contain",
    "ض.م.ن", "ح.و.ي", "ش.م.ل", "م.ل.أ", "ف.ر.غ",
    "ض.#.ف",  # weak for إضافة (add/contain)
)

# ── open / close ──
_add("open",
    "ف.ت.ح", "غ.ل.ق", "ب.و.ب", "ق.ف.ل", "ك.ش.ف", "س.ت.ر",
)

# ── hold / grasp ──
_add("hold",
    "م.س.ك", "ق.ب.ض", "ع.ل.ق", "ح.م.ل", "ر.ف.ع",
)

# ── hide / conceal ──
_add("hide",
    "خ.ف.ي", "س.ت.ر", "ك.ت.م", "غ.ي.ب", "ح.ج.ب", "خ.ب.أ", "ب.ط.ن",
)

# ── gather / collect ──
_add("gather",
    "ج.م.ع", "ح.ش.د", "ض.م.م", "ل.م.م", "ج.ن.ي", "ح.ص.ل",
    "ح.#.ل",  # weak
)

# ── send / transmit ──
_add("send",
    "ر.س.ل", "ب.ع.ث", "و.ج.ه", "ن.ق.ل", "ب.ث.ث",
)

# ── social / community ──
_add("social",
    "ش.ر.ك", "ج.و.ر", "أ.ه.ل", "ق.و.م", "ش.ع.ب", "أ.م.م",
    "ق.ب.ل", "ح.ز.ب",
)

# ── dwell / reside ──
_add("dwell",
    "س.ك.ن", "ع.م.ر", "ب.ن.ي", "ن.ز.ل", "أ.ق.م",
)

# ── need / want ──
_add("need",
    "ح.و.ج", "ل.ز.م", "ض.ر.ر", "و.ج.ب",
)
_add("want",
    "ط.ل.ب", "ر.غ.ب", "ت.م.ن", "ش.ه.و", "ب.غ.ي",
)

# ── enable / allow ──
_add("enable",
    "م.ك.ن", "أ.ذ.ن", "س.م.ح", "ق.د.ر",  "ي.س.ر",
)

# ── decide / judge ──
_add("decide",
    "ق.ر.ر", "ح.ك.م", "ف.ص.ل", "ع.ز.م",
)

# ── fix / repair ──
_add("fix",
    "ص.ل.ح", "ر.م.م", "ع.د.ل", "ض.ب.ط",
)

# ── rest / pause ──
_add("rest",
    "ر.ا.ح", "ن.و.م", "ه.د.أ", "و.ق.ف", "ت.و.ق",
)

# ── person / human ──
_add("person",
    "ب.ش.ر", "إ.ن.س", "ر.ج.ل", "م.ر.أ", "ط.ف.ل", "ش.ي.خ", "ش.ب.ب",
    "ن.س.ب",  # lineage/relation
)

# ── name / identity ──
_add("name",
    "س.م.ي", "ل.ق.ب", "ع.ن.و", "و.س.م",
)

# ── art / beauty ──
_add("art",
    "ف.ن.ن", "ج.م.ل", "ز.خ.ر", "ن.ق.ش", "ر.س.م", "ل.ح.ن",
    "غ.ن.ي", "ع.ز.ف", "ر.ق.ص", "م.ث.ل", "ص.و.ر",
)

# ── science / study ──
_add("science",
    "ب.ح.ث", "ن.ظ.ر", "ح.ل.ل", "ق.ي.س", "ك.ش.ف", "ف.ح.ص",
)

# ── tech / technology ──
_add("tech",
    "ت.ق.ن", "ب.ر.م", "ش.ب.ك", "ه.ن.د",
)

# ── material / substance ──
_add("material",
    "م.ع.د", "ح.ج.ر", "ح.د.د", "ذ.ه.ب", "ف.ض.ض", "ن.ح.س",
    "خ.ش.ب", "ز.ج.ج", "ق.م.ش", "ن.س.ج",
    "ذ.ل.ل",  # Map ذلل to a useful field (commonly means "to humble" but often in material context)
)

# ── structure / form ──
_add("structure",
    "ش.ك.ل", "ه.ي.ك", "ن.ظ.م", "ص.ف.ف", "ر.ت.ب",
    "ط.ب.ق",  # layer/floor — common
)

# ── quality / attribute ──
_add("quality",
    "ص.ف.ي", "ج.و.د", "ح.س.ن", "س.و.أ", "ن.ظ.ف", "ق.ب.ح",
    "ج.د.د", "ق.د.م", "ص.ع.ب", "س.ه.ل",
    "ك.م.م",  # completeness
)

# ── sport / game ──
_add("sport",
    "ل.ع.ب", "ر.ي.ض", "س.ب.ق", "ف.و.ز", "ه.ز.م",
)

# ── Additional high-frequency roots from analysis ──
# These are the top missed roots from the 1K test

_add("exist", "ك.#.ن")       # كان/يكون — most common Arabic verb
_add("speak", "#.#.ل")       # likely أول (first) or قول — ambiguous, map to speak
_add("exist", "م.#.#")       # ambiguous weak root
_add("speak", "ح.#.ث")       # حدث (happen/speak)
_add("move", "د.#.ل")        # دخل (enter)
_add("move", "خ.#.ر")        # خرج (exit)
_add("speak", "ل.غ.#")       # لغة (language)
_add("measure", "#.ح.د")     # وحد (one/unite)
_add("exist", "#.ج.د")       # وجد (exist/find)
_add("connect", "#.ل")       # weak biradical
_add("time", "#.خ.ر")        # أخر (other/last)
_add("time", "ب.د.#")        # بداية (beginning)
_add("time", "م.د.د")        # مدة (period)
_add("change", "ح.#.ل")      # حول (around/transform)
_add("change", "غ.#.ر")      # غير (change/other)
_add("quality", "ت.م.م")     # تمام (complete)
_add("quality", "ك.م.م")     # كمال (perfection)
_add("body", "ر.#.س")        # رأس (head)
_add("place", "#.ر.ض")       # أرض (earth)
_add("nature", "ر.#.ض")      # روض (garden)
_add("govern", "ق.#.د")      # قاد (lead)
_add("force", "ق.#.ي")       # قوي (strong)
_add("move", "م.ر.ر")        # مرور (passing)
_add("measure", "ع.#.د")     # عدد (number)
_add("contain", "ض.#.ف")     # إضافة (addition)
_add("person", "ن.س.ب")      # نسب (lineage)
_add("structure", "ط.ب.ق")   # طبقة (layer)
_add("measure", "ث.ن.#")     # اثنان (two)
_add("time", "د.#.ر")        # دور (cycle/role)
_add("exist", "ح.#.#")       # ambiguous
_add("exist", "#.ل.#")       # ambiguous weak
_add("exist", "#.#.ض")       # weak
_add("speak", "س.#.ل")       # سؤال (question)
_add("size", "ك.ل.ل")        # كل (all/every)
_add("social", "ه.م.م")      # اهتمام (interest)
_add("time", "ع.ن.د")        # عند (at/when)
_add("speak", "#.ل")          # biradical weak
_add("exist", "م.#")          # biradical weak
_add("exist", "ه")            # ه pronoun suffix (mapped to exist as placeholder)
_add("speak", "ل.#")          # biradical weak

# ── Round 2: top missed roots from v2 1K analysis ──
_add("speak", "م.ن")          # من (who/from) as content word
_add("place", "ج.ز.#")       # جزيرة (island) / جزء (part)
_add("move", "ط.ر.ق")        # طريق (road/method)
_add("nature", "س.ط.ح")      # سطح (surface)
_add("enable", "#.ف.ق")      # وفق (accord/succeed)
_add("exist", "ج.#.#")       # ambiguous weak
_add("structure", "ق.ط.ع")   # قطع (piece/sector)
_add("see", "ش.#.ر")         # شهر (month/fame) — also could be feel
_add("time", "س.ن.#")        # سنة (year)
_add("place", "#.س.ط")       # وسط (middle)
_add("measure", "د.ر.ج")     # درجة (degree)
_add("weather", "ب.خ.ر")     # بخار (steam)
_add("time", "خ.ل.ف")        # خلف (behind/successor)
_add("know", "ح.ق.ق")        # حقيقة (truth/reality)
_add("place", "#.ق.ع")       # وقع (occur/location)
_add("tech", "ه.ن.#")        # هندسة (engineering)
_add("change", "#.ث.ر")      # أثر (effect/trace)
_add("enable", "ج.#.ز")      # جهاز (device) / إنجاز (achievement)
_add("exist", "#.ف.#")       # ambiguous weak
_add("make", "ج.#.ل")        # جعل (make/cause)
_add("science", "ر.ك.ز")     # مركز (center/concentrate)
_add("want", "#.ر.د")        # أراد (want) / ورد (mention)
_add("speak", "س.ب.ب")       # سبب (reason/cause)
_add("see", "ظ.ه.ر")         # ظهور (appearance)
_add("time", "ش.ه.ر")        # شهر (month)
_add("tech", "ه.ن.د.س")      # هندسة (engineering) 4-letter
_add("connect", "ع.ق.د")     # عقد (contract/decade)
_add("fight", "ج.#.ش")       # جيش (army)
_add("size", "#.ل.ف")        # ألف (thousand)
_add("exist", "ب")            # ب preposition clitic

# ── Round 3: top missed roots from v2 round-2 analysis ──
_add("time", "#.ر.خ")        # تاريخ (history/date)
_add("nature", "ح.#.ط")      # محيط (ocean/surrounding)
_add("contain", "غ.ل.ف")     # غلاف (envelope/atmosphere)
_add("exist", "#.#.ن")       # weak variant
_add("structure", "ق.س.م")   # قسم (section/department)
_add("write", "ر.م.ز")       # رمز (symbol/code)
_add("know", "ب.د.ه")        # بديهية (axiom)
_add("trade", "#.ق.د")       # عقد (contract) / نقد (money/criticism)
_add("govern", "ج.م.ه.ر")   # جمهورية (republic) — 4-letter
_add("fight", "ع.س.ك.ر")    # عسكري (military) — 4-letter
_add("exist", "ه.#.#")       # weak
_add("force", "غ.ل.ب")       # غلب (overcome/mostly)
_add("exist", "ف.#")          # weak biradical — في (in)
_add("govern", "ن.#.خ")       # انتخاب (election)
_add("work", "ف.ع.ل")        # فعل (do/verb)
_add("quality", "ف.ض.ل")     # فضل (virtue/prefer)
_add("want", "ق.ص.د")        # قصد (intend)
_add("think", "ف.ل.س.ف")    # فلسفة (philosophy) — 4-letter
_add("speak", "ب.ل.غ")       # بلغ (reach/report)
_add("exist", "#.ن.ن")       # weak
_add("nature", "م.ل.ح")      # ملح (salt/sea)
_add("exist", "ل")            # ل preposition clitic
_add("move", "س.ب.ل")        # سبيل (path/way)
_add("change", "م.#.ز")      # تمييز (distinction)
_add("size", "ك.ث.ف")        # كثافة (density)
_add("exist", "#.ن")          # weak biradical
_add("enable", "ك.ف.#")      # كفاية (sufficiency)
_add("see", "#.ز.ر")         # وزارة (ministry) / visit
_add("destroy", "ع.د.م")     # عدم (non-existence/absence)
_add("speak", "ب.م")          # بما (in what)

# ── Round 4: remaining frequent missed roots ──
_add("exist", "#.#.م")       # weak
_add("know", "ث.ب.ت")        # ثبت (prove/establish)
_add("force", "ج.ب.ر")       # جبر (algebra/force)
_add("place", "ع.ر.ب")       # عرب (Arab/Arabic)
_add("social", "س.ل.م")      # سلام (peace) / إسلام (Islam)
_add("make", "ص.#.غ")        # صيغة (form/formula)
_add("nature", "ع.ذ.ب")      # عذب (fresh water/sweet)
_add("enable", "ط.#.ق")      # طاقة (energy)
_add("change", "خ.ف.ض")      # خفض (reduce)
_add("time", "ص.ب.ح")        # صباح (morning)
_add("measure", "ث.ل.ث")     # ثلث (third)
_add("person", "ف.ر.د")      # فرد (individual)
_add("nature", "ج.ر.د")      # تجريد (abstraction)
_add("exist", "ش.#.#")       # weak
_add("measure", "ث.ل.ث")     # duplicated but harmless
_add("quality", "ك.م.ل")     # كمال (completeness)
_add("place", "ج.غ.ر.ف")    # جغرافيا (geography) — 4-letter
_add("move", "#.ف.ر")        # سفر (travel)
_add("social", "ع.ض.#")      # عضو (member)
_add("science", "م.ر.س")     # ممارسة (practice)
_add("structure", "ن.ق.ط")   # نقطة (point)
_add("speak", "د.ع.#")       # دعوة (call/invitation)
_add("gather", "ح.ص.#")      # حصة (share/portion)
_add("measure", "ر.ب.ع")     # ربع (quarter)
_add("decide", "ع.م.د")      # عمد (intend)
_add("size", "ح.ج.م")        # حجم (volume/size)
_add("move", "س.ر.ع")        # سرعة (speed)

# ── Round 5: remaining top missed roots ──
_add("nature", "ن.ج.م")      # نجم (star)
_add("know", "د.ل.ل")        # دليل (guide/evidence)
_add("place", "ج.ز.ر")       # جزيرة (island)
_add("force", "ر.غ.م")       # رغم (despite)
_add("social", "ح.ل.ف")      # حلف (alliance)
_add("help", "د.ع.م")        # دعم (support)
_add("trade", "س.ه.م")       # سهم (share/arrow)
_add("make", "ج.ع.ل")        # جعل (make/cause)
_add("structure", "ص.ن.ف")   # تصنيف (classification)
_add("nature", "ق.م.ر")      # قمر (moon)
_add("person", "ن.ف.س")      # نفس (self/soul)
_add("see", "ب.ر.ز")         # بروز (prominence)
_add("know", "ك.#.ف")        # كشف (discover)
_add("create", "ن.#.س")      # تأسيس (founding)
_add("give", "#.ز.ع")        # توزيع (distribution)
_add("force", "س.ل.ب")       # سلب (negative)
_add("write", "#.ت.ر")       # وتر (string/chord)
_add("structure", "ع.ن.ص.ر") # عنصر (element) — 4-letter
_add("contain", "خ.ز.ن")     # خزانة (cabinet/storage)
_add("kind", "ن.#.ع")        # نوع (type/kind)
_add("sign", "ر.#.م")        # رقم (number/digit)
_add("measure", "د.ق.ق")     # دقيقة (minute/precise)
_add("see", "#.ض.ح")         # واضح (clear)
_add("work", "ن.ف.ذ")        # تنفيذ (execution)
_add("know", "ج.ر.ب")        # تجربة (experiment)
_add("structure", "ف.ر.ع")   # فرع (branch)
_add("exist", "#.ن.#")       # weak


# ── Arabic function words → token type ───────────────────────────
ARABIC_FUNCTION_WORDS = {
    # Prepositions
    "في": "PREP", "من": "PREP", "إلى": "PREP", "على": "PREP",
    "عن": "PREP", "مع": "PREP", "بين": "PREP", "حول": "PREP",
    "خلال": "PREP", "منذ": "PREP", "حتى": "PREP", "نحو": "PREP",
    "لدى": "PREP", "عند": "PREP", "فوق": "PREP", "تحت": "PREP",
    "أمام": "PREP", "خلف": "PREP", "بعد": "PREP", "قبل": "PREP",
    "دون": "PREP", "ضد": "PREP", "عبر": "PREP", "ضمن": "PREP",
    "لأجل": "PREP",
    # Conjunctions
    "و": "CONJ", "أو": "CONJ", "ثم": "CONJ", "لكن": "CONJ",
    "بل": "CONJ", "أم": "CONJ", "إذا": "CONJ", "لو": "CONJ",
    "إذ": "CONJ", "كي": "CONJ", "حيث": "CONJ", "لأن": "CONJ",
    "بينما": "CONJ", "كما": "CONJ", "مثل": "CONJ", "حين": "CONJ",
    "عندما": "CONJ", "لما": "CONJ",
    # Pronouns
    "هو": "PRON", "هي": "PRON", "هم": "PRON", "هن": "PRON",
    "أنا": "PRON", "نحن": "PRON", "أنت": "PRON", "أنتم": "PRON",
    "هذا": "PRON", "هذه": "PRON", "ذلك": "PRON", "تلك": "PRON",
    "الذي": "PRON", "التي": "PRON", "الذين": "PRON", "اللذين": "PRON",
    "اللاتي": "PRON", "ما": "PRON", "هؤلاء": "PRON",
    # Determiners / particles
    "كل": "DET", "بعض": "DET", "أي": "DET", "غير": "DET",
    "كلا": "DET", "أحد": "DET", "جميع": "DET", "سائر": "DET",
    "معظم": "DET", "أغلب": "DET", "عدة": "DET", "كثير": "DET",
    "قليل": "DET", "نفس": "DET", "ذات": "DET", "أكثر": "DET",
    # Negation
    "لا": "NEG", "لم": "NEG", "لن": "NEG",
    "ليس": "NEG",
    # Auxiliary / modal
    "كان": "AUX", "يكون": "AUX", "أصبح": "AUX", "ظل": "AUX",
    "بات": "AUX", "صار": "AUX",
    "قد": "PART", "سوف": "PART", "لقد": "PART", "إن": "PART",
    "أن": "PART", "إنّ": "PART", "لعل": "PART", "أنّ": "PART",
    # Numbers
    "واحد": "NUM", "اثنان": "NUM", "ثلاثة": "NUM", "أربعة": "NUM",
    "خمسة": "NUM", "ستة": "NUM", "سبعة": "NUM", "ثمانية": "NUM",
    "تسعة": "NUM", "عشرة": "NUM", "عشر": "NUM",
    "مئة": "NUM", "مائة": "NUM", "ألف": "NUM", "مليون": "NUM",
    # Common adverbs (mapped as function)
    "أيضا": "ADV", "أيضاً": "ADV", "جدا": "ADV", "جداً": "ADV",
    "فقط": "ADV", "تقريبا": "ADV", "تقريباً": "ADV",
    "حاليا": "ADV", "حالياً": "ADV",
    # Extra
    "بما": "PREP", "ثلث": "NUM",
}


# Common Arabic proclitics to strip for re-analysis
PROCLITICS = ["وال", "وب", "ول", "وك", "فال", "فب", "فل",
              "ال", "لل", "بال", "كال"]


# ═══════════════════════════════════════════════════════════════
# STEP 1: Download Arabic Sentences
# ═══════════════════════════════════════════════════════════════

def download_arabic_sentences(target: int, output_path: str) -> list[str]:
    """Download Arabic sentences from Arabic Wikipedia via HuggingFace API."""
    if os.path.exists(output_path):
        print(f"  Loading cached sentences from {output_path}...")
        with open(output_path) as f:
            return json.load(f)

    print(f"  Downloading {target:,} Arabic sentences from Wikipedia...")
    base_url = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=wikimedia%2Fwikipedia&config=20231101.ar&split=train"
    )
    batch_size = 100
    sentences = []
    offset = 0
    max_offset = target * 5

    while len(sentences) < target and offset < max_offset:
        url = f"{base_url}&offset={offset}&length={batch_size}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            for row in data.get("rows", []):
                text = row["row"].get("text", "")
                for sent in re.split(r'[.؟!]\s*', text):
                    sent = sent.strip()
                    if len(sent) < 20 or len(sent) > 300:
                        continue
                    arabic_chars = sum(1 for c in sent if '\u0600' <= c <= '\u06FF')
                    if arabic_chars < len(sent) * 0.5:
                        continue
                    sentences.append(sent)
                    if len(sentences) >= target:
                        break
                if len(sentences) >= target:
                    break

            offset += batch_size
            if len(sentences) % 5000 < batch_size * 3:
                print(f"    {len(sentences):,} / {target:,} sentences...")

        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            offset += batch_size
            time.sleep(1)

    sentences = sentences[:target]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=0)

    print(f"  Saved {len(sentences):,} sentences to {output_path}")
    return sentences


# ═══════════════════════════════════════════════════════════════
# STEP 2: Arabic CST Tokenizer
# ═══════════════════════════════════════════════════════════════

def _build_wildcard_index() -> dict[str, str]:
    """
    Build a lookup that also matches camel-tools weak roots with #.
    For each root in ARABIC_ROOT_TO_FIELD, generate patterns where
    weak letters (و ي أ ا ء إ آ ئ ؤ) are replaced with #.
    """
    index = dict(ARABIC_ROOT_TO_FIELD)  # start with exact
    weak_letters = set("وياأإآءئؤ")

    for root, field in list(ARABIC_ROOT_TO_FIELD.items()):
        parts = root.split(".")
        if len(parts) != 3:
            continue
        # Generate all weak variants
        for i in range(3):
            if parts[i] in weak_letters:
                variant = list(parts)
                variant[i] = "#"
                key = ".".join(variant)
                if key not in index:
                    index[key] = field
        # Generate double-weak
        for i in range(3):
            for j in range(i+1, 3):
                if parts[i] in weak_letters and parts[j] in weak_letters:
                    variant = list(parts)
                    variant[i] = "#"
                    variant[j] = "#"
                    key = ".".join(variant)
                    if key not in index:
                        index[key] = field

    return index


class ArabicCSTTokenizer:
    """
    Contextual Semantic Tokenizer for Arabic.

    Token format (mirrors English CST):
      ROOT:<field>     — semantic root token
      FUNC:<type>      — function word
      SURF:<word>      — surface fallback for unknown words
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.vocab: dict[str, int] = {}
        self.next_id = 0
        self.root_index = _build_wildcard_index()
        self._init_special_tokens()
        self._init_semantic_tokens()
        self.stats = Counter()

    def _init_special_tokens(self):
        for tok in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]:
            self._get_id(tok)

    def _init_semantic_tokens(self):
        fields = sorted(set(self.root_index.values()))
        for f in fields:
            self._get_id(f"ROOT:{f}")
        for t in sorted(set(ARABIC_FUNCTION_WORDS.values())):
            self._get_id(f"FUNC:{t}")

    def _get_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        tid = self.next_id
        self.vocab[token] = tid
        self.next_id += 1
        return tid

    def _strip_arabic(self, word: str) -> str:
        """Remove tashkeel (diacritics) and tatweel."""
        word = re.sub(r'[\u064B-\u065F\u0670]', '', word)
        word = word.replace('\u0640', '')
        return word

    def _find_field(self, roots: list[str]) -> str | None:
        """Find semantic field for any of the candidate roots."""
        for r in roots:
            if r in self.root_index:
                return self.root_index[r]
        return None

    def _analyze_word(self, clean: str) -> tuple[str | None, list[str]]:
        """Analyze word, try proclitic stripping if needed. Returns (field, roots)."""
        analyses = self.analyzer.analyze(clean)
        roots = []
        for a in analyses:
            r = a.get("root", "")
            if r and r not in ("NTWS", "PUNC", "DIGIT", "FOREIGN"):
                roots.append(r)

        field = self._find_field(roots)
        if field:
            return field, roots

        # Try stripping proclitics and re-analyzing
        for prefix in PROCLITICS:
            if clean.startswith(prefix) and len(clean) > len(prefix) + 1:
                stem = clean[len(prefix):]
                analyses2 = self.analyzer.analyze(stem)
                roots2 = []
                for a in analyses2:
                    r = a.get("root", "")
                    if r and r not in ("NTWS", "PUNC", "DIGIT", "FOREIGN"):
                        roots2.append(r)
                field2 = self._find_field(roots2)
                if field2:
                    return field2, roots2

        return None, roots

    def tokenize(self, sentence: str) -> dict:
        """Tokenize an Arabic sentence into CST tokens."""
        tokens = []
        ids = []

        tokens.append("[BOS]")
        ids.append(self.vocab["[BOS]"])

        words = re.findall(r'[\u0600-\u06FF\u0750-\u077F]+', sentence)

        for word in words:
            clean = self._strip_arabic(word)

            # Check function words first
            if clean in ARABIC_FUNCTION_WORDS:
                tok_type = ARABIC_FUNCTION_WORDS[clean]
                tok = f"FUNC:{tok_type}"
                tokens.append(tok)
                ids.append(self._get_id(tok))
                self.stats["func"] += 1
                continue

            # Try morphological analysis with weak-root matching
            field, roots = self._analyze_word(clean)

            if field:
                tok = f"ROOT:{field}"
                tokens.append(tok)
                ids.append(self._get_id(tok))
                self.stats["root"] += 1
            elif roots:
                # Root found but no field mapping
                tok = f"SURF:{clean}"
                tokens.append(tok)
                ids.append(self._get_id(tok))
                self.stats["surf_with_root"] += 1
            else:
                tok = f"SURF:{clean}"
                tokens.append(tok)
                ids.append(self._get_id(tok))
                self.stats["surf_no_root"] += 1

        tokens.append("[EOS]")
        ids.append(self.vocab["[EOS]"])

        return {"ids": ids, "tokens": tokens, "text": sentence}

    def get_vocab_size(self):
        return len(self.vocab)

    def save_vocab(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# STEP 3: Tokenize with CST
# ═══════════════════════════════════════════════════════════════

def tokenize_cst(sentences: list[str], output_dir: str) -> dict:
    """Tokenize Arabic sentences with CST and write .jsonl."""
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer

    print("  Loading camel-tools morphology database...")
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)

    tokenizer = ArabicCSTTokenizer(analyzer)
    n = len(sentences)
    output_path = os.path.join(output_dir, f"train-{n}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Tokenizing {n:,} sentences with Arabic CST...")
    lines = []
    t0 = time.time()

    for i, sent in enumerate(sentences):
        result = tokenizer.tokenize(sent)
        if len(result["ids"]) < 4:
            continue
        lines.append(json.dumps(result, ensure_ascii=False))

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:,} / {n:,} ({elapsed:.0f}s)")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    vocab_path = output_path.replace(".jsonl", "-vocab.json")
    tokenizer.save_vocab(vocab_path)

    elapsed = time.time() - t0
    stats = dict(tokenizer.stats)
    total = sum(stats.values())

    print(f"\n  ═══ Arabic CST Stats ═══")
    print(f"  Sentences:    {len(lines):,}")
    print(f"  Vocab size:   {tokenizer.get_vocab_size():,}")
    print(f"  Time:         {elapsed:.0f}s")
    print(f"  Token breakdown:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        pct = v / total * 100 if total else 0
        print(f"    {k:20s} {v:8,} ({pct:5.1f}%)")
    print(f"    {'TOTAL':20s} {total:8,}")

    total_tokens = sum(len(json.loads(l)["ids"]) for l in lines)
    avg = total_tokens / len(lines) if lines else 0
    print(f"  Avg tokens/sent: {avg:.1f}")

    return {
        "examples": len(lines),
        "vocab_size": tokenizer.get_vocab_size(),
        "output_path": output_path,
        "vocab_path": vocab_path,
        "stats": stats,
        "total_tokens": total_tokens,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 4: Tokenize with SentencePiece BPE
# ═══════════════════════════════════════════════════════════════

def tokenize_spm(sentences: list[str], output_dir: str, vocab_size: int = 8000) -> dict:
    """Train SentencePiece BPE on Arabic and tokenize."""
    import sentencepiece as spm

    n = len(sentences)
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, f"ar-bpe-{vocab_size}")
    output_path = os.path.join(output_dir, f"train-{n}-{vocab_size // 1000}k.jsonl")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
        tmp_path = f.name

    print(f"  Training SentencePiece BPE (vocab={vocab_size:,})...")
    t0 = time.time()
    try:
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            num_threads=os.cpu_count() or 4,
        )
    finally:
        os.unlink(tmp_path)

    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    print(f"  Actual SPM vocab size: {sp.get_piece_size():,}")

    print(f"  Tokenizing {n:,} sentences with SentencePiece...")
    count = 0
    total_tokens = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            tids = sp.encode(sent, out_type=int)
            toks = sp.encode(sent, out_type=str)
            if len(tids) < 3:
                continue
            example = {"ids": tids, "tokens": toks, "text": sent}
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
            total_tokens += len(tids)

    vocab_path = output_path.replace(".jsonl", "-vocab.json")
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    with open(vocab_path, "w", encoding="utf-8") as vf:
        json.dump(vocab, vf, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    avg = total_tokens / count if count else 0
    print(f"\n  ═══ Arabic SPM BPE-{vocab_size // 1000}K Stats ═══")
    print(f"  Sentences:       {count:,}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Avg tokens/sent: {avg:.1f}")
    print(f"  Vocab size:      {sp.get_piece_size():,}")
    print(f"  Time:            {elapsed:.0f}s")

    return {
        "examples": count,
        "vocab_size": sp.get_piece_size(),
        "output_path": output_path,
        "vocab_path": vocab_path,
        "total_tokens": total_tokens,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Arabic CST Experiment")
    parser.add_argument("--sentences", type=int, default=100000,
                        help="Number of Arabic sentences to download")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-spm", action="store_true")
    parser.add_argument("--skip-cst", action="store_true")
    args = parser.parse_args()

    n = args.sentences
    data_dir = "data/arabic"
    sentences_path = f"{data_dir}/sentences-{n}.json"

    print("=" * 60)
    print(f"  Arabic CST Experiment — {n:,} sentences")
    print("=" * 60)

    # Step 1: Download
    print(f"\n── Step 1: Download Arabic Sentences ──")
    sentences = download_arabic_sentences(n, sentences_path)
    print(f"  Total: {len(sentences):,} sentences")

    # Step 2: CST tokenization
    if not args.skip_cst:
        print(f"\n── Step 2: CST Tokenization ──")
        cst_dir = "data/tokenized/cst-ar"
        cst_result = tokenize_cst(sentences, cst_dir)
    else:
        cst_result = None
        print("\n── Step 2: CST (SKIPPED) ──")

    # Step 3: SentencePiece BPE — 8K and 32K (matching English)
    spm_results = {}
    if not args.skip_spm:
        spm_dir = "data/tokenized/spm-ar"
        for vs in [8000, 32000]:
            print(f"\n── Step 3: SentencePiece BPE-{vs // 1000}K ──")
            spm_results[vs] = tokenize_spm(sentences, spm_dir, vocab_size=vs)
    else:
        print("\n── Step 3: SentencePiece BPE (SKIPPED) ──")

    # Step 4: Summary
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON SUMMARY — {n:,} Arabic sentences")
    print(f"{'=' * 60}")
    if cst_result:
        print(f"\n  Arabic CST:")
        print(f"    Examples:       {cst_result['examples']:,}")
        print(f"    Vocab size:     {cst_result['vocab_size']:,}")
        print(f"    Total tokens:   {cst_result['total_tokens']:,}")
        avg_cst = cst_result['total_tokens'] / cst_result['examples']
        print(f"    Avg tokens/sent:{avg_cst:.1f}")

    for vs, res in spm_results.items():
        print(f"\n  Arabic SPM BPE-{vs // 1000}K:")
        print(f"    Examples:       {res['examples']:,}")
        print(f"    Vocab size:     {res['vocab_size']:,}")
        print(f"    Total tokens:   {res['total_tokens']:,}")
        avg_spm = res['total_tokens'] / res['examples']
        print(f"    Avg tokens/sent:{avg_spm:.1f}")

    if cst_result:
        for vs, res in spm_results.items():
            comp = res['total_tokens'] / cst_result['total_tokens']
            print(f"\n  CST vs BPE-{vs // 1000}K: CST uses {1/comp:.1%} of BPE tokens ({comp:.2f}x compression)")

    print(f"\n  Output files:")
    if cst_result:
        print(f"    CST:     {cst_result['output_path']}")
    for vs, res in spm_results.items():
        print(f"    SPM-{vs // 1000}K: {res['output_path']}")

    print(f"\n  Next: python training/train_gpt2.py (adjust paths for Arabic)")


if __name__ == "__main__":
    main()
