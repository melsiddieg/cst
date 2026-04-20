"""
Arabic CST 1M — Download 1M sentences, CST tokenize, produce train-1000000.jsonl

Run on Colab:
  !pip install camel-tools
  !camel_data -i morphology-db-msa-r13
  !python tokenize_1m.py

Output: /content/cst_1m/train-1000000.jsonl + train-1000000-vocab.json
Upload these to Colab for training with colab_edge.py (swap DATA_FILE/VOCAB_FILE).
"""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Arabic root → semantic field mapping (from arabic_experiment_v2.py)
# ═══════════════════════════════════════════════════════════════

ARABIC_ROOT_TO_FIELD: dict[str, str] = {}

def _add(field: str, *roots: str):
    for r in roots:
        ARABIC_ROOT_TO_FIELD[r] = field

_add("write", "ك.ت.ب", "خ.ط.ط", "س.ج.ل", "د.و.ن", "ر.ق.م", "ن.س.خ", "ط.ب.ع", "ن.ش.ر", "ص.د.ر", "و.ث.ق", "ص.ح.ف", "د.#.ن")
_add("know", "ع.ل.م", "ع.ر.ف", "د.ر.س", "ف.ه.م", "ث.ق.ف", "خ.ب.ر", "ف.ق.ه", "ب.ح.ث", "ر.ش.د", "ل.ق.ن", "و.ع.ي", "ح.ف.ظ", "ع.ل.#", "ع.#.م")
_add("speak", "ق.و.ل", "ك.ل.م", "ح.د.ث", "ن.ط.ق", "خ.ط.ب", "ص.ر.خ", "ن.د.ي", "ل.غ.و", "ح.ك.ي", "ع.ل.ن", "ذ.ك.ر", "ر.و.ي", "س.أ.ل", "ج.و.ب", "ف.س.ر", "و.ص.ف", "ب.ي.ن", "ش.ر.ح", "ق.#.ل", "ح.#.ث", "ب.#.ن")
_add("think", "ف.ك.ر", "ع.ق.ل", "ر.أ.ي", "ظ.ن.ن", "ح.س.ب", "ن.ظ.ر", "خ.م.ن", "ق.ر.ر", "ز.ع.م", "ر.#.ي")
_add("see", "ب.ص.ر", "ش.ه.د", "ل.ح.ظ", "ل.م.ح", "ر.ق.ب", "ت.ب.ع", "ر.ص.د")
_add("feel", "ح.ب.ب", "ش.ع.ر", "ح.ز.ن", "ف.ر.ح", "خ.و.ف", "غ.ض.ب", "ق.ل.ق", "ر.ض.ي", "أ.م.ل", "ن.د.م", "أ.ل.م", "س.ع.د", "ح.ن.ن", "ع.ش.ق", "ك.ر.ه", "ح.ي.ر", "ذ.ع.ر", "ف.ز.ع")
_add("move", "م.ش.ي", "ذ.ه.ب", "ر.ج.ع", "س.ي.ر", "ق.د.م", "ر.ح.ل", "ه.ج.ر", "ج.ر.ي", "ط.ي.ر", "ع.ب.ر", "ه.ب.ط", "ص.ع.د", "د.خ.ل", "خ.ر.ج", "ف.ر.ر", "س.ب.ح", "ق.ف.ز", "ز.ح.ف", "ر.ك.ب", "ذ.#.ب", "ج.#.ز", "ر.#.ح")
_add("give", "ع.ط.ي", "و.ه.ب", "ت.ب.ر", "م.ن.ح", "ق.د.م")
_add("take", "أ.خ.ذ", "ق.ب.ل", "س.ر.ق", "ن.ه.ب", "خ.ط.ف", "س.ل.ب")
_add("make", "ص.ن.ع", "ب.ن.ي", "ع.م.ل", "ش.ي.د", "خ.ل.ق", "أ.ن.ش")
_add("destroy", "ه.د.م", "ك.س.ر", "ح.ط.م", "ق.ت.ل", "م.ح.ق", "ف.ن.ي", "ح.ر.ق", "غ.ر.ق", "خ.ر.ب", "ع.د.م")
_add("change", "ب.د.ل", "غ.ي.ر", "ح.و.ل", "ط.و.ر", "ن.م.و", "ز.ي.د")
_add("exist", "ك.و.ن", "و.ج.د", "ح.ي.و", "ب.ق.ي", "ع.ي.ش")
_add("time", "و.ق.ت", "ز.م.ن", "ت.ر.خ", "ب.د.ء", "ن.ه.ي", "خ.ت.م", "م.ه.ل")
_add("place", "م.ك.ن", "م.و.ض", "ب.ل.د", "م.د.ن", "ق.ر.ي", "م.ن.ط", "ح.د.د", "ق.ط.ر", "و.ل.ي")
_add("possess", "م.ل.ك", "ح.و.ز", "ك.س.ب", "ف.ق.د", "ح.ر.م")
_add("trade", "ب.ي.ع", "ش.ر.ي", "ت.ج.ر", "ر.ب.ح", "خ.س.ر", "س.و.ق", "ث.م.ن")
_add("fight", "ح.ر.ب", "ق.ت.ل", "ج.ه.د", "ن.ض.ل", "د.ف.ع", "ه.ج.م", "ق.و.م", "غ.ز.و", "ف.ت.ح")
_add("enable", "ع.و.ن", "ن.ص.ر", "س.ع.ف", "غ.و.ث", "أ.ن.ق")
_add("govern", "ح.ك.م", "س.ي.س", "م.ل.ك", "أ.م.ر", "ق.و.د", "ر.ئ.س")
_add("create", "خ.ل.ق", "ب.د.ع", "أ.ن.ش", "و.ل.د", "ف.ط.ر", "ح.د.ث", "ك.و.ن")
_add("force", "ق.و.ي", "ض.غ.ط", "ج.ب.ر", "ق.ه.ر", "أ.ر.غ", "ش.د.د")
_add("body", "ج.س.م", "ر.أ.س", "ي.د.ي", "ق.ل.ب", "ع.ي.ن", "س.م.ع", "د.م.م", "ع.ظ.م", "ل.ح.م", "ج.ل.د")
_add("food", "أ.ك.ل", "ش.ر.ب", "ط.ع.م", "ط.ب.خ", "ج.و.ع", "ع.ط.ش", "ذ.و.ق", "ه.ض.م", "غ.ذ.ي", "#.ك.ل")
_add("nature", "ط.ب.ع", "أ.ر.ض", "ب.ح.ر", "ن.ه.ر", "ج.ب.ل", "ب.ر.ر", "ص.ح.ر", "غ.ا.ب", "و.ا.د", "س.ه.ل", "#.ر.ض", "ر.#.ض")
_add("weather", "م.ط.ر", "ر.ي.ح", "ث.ل.ج", "ح.ر.ر", "ب.ر.د", "ش.م.س", "غ.ي.م", "ع.ص.ف", "ف.ي.ض", "ج.ف.ف")
_add("animal", "ح.ي.و", "ط.ي.ر", "س.م.ك", "ح.ش.ر", "ذ.ئ.ب", "أ.س.د", "ف.ر.س", "ب.ق.ر", "غ.ن.م", "ج.م.ل", "ك.ل.ب")
_add("plant", "ز.ر.ع", "ن.ب.ت", "ش.ج.ر", "ث.م.ر", "ز.ه.ر", "ح.ص.د", "غ.ر.س", "ر.و.ض")
_add("color", "ل.و.ن", "ب.ي.ض", "س.و.د", "ح.م.ر", "خ.ض.ر", "ز.ر.ق", "ص.ف.ر")
_add("size", "ك.ب.ر", "ص.غ.ر", "ط.و.ل", "ق.ص.ر", "ع.ر.ض", "و.س.ع", "ض.ي.ق", "ع.م.ق", "ك.ث.ر", "ق.ل.ل")
_add("measure", "ق.ي.س", "و.ز.ن", "ع.د.د", "ح.س.ب", "م.س.ح", "ب.ع.د", "ق.ر.ب", "ن.ص.ف", "ع.#.د")
_add("connect", "و.ص.ل", "ر.ب.ط", "ج.م.ع", "ض.م.م", "ل.ح.م", "ش.ب.ك", "ع.ل.ق", "ز.و.ج")
_add("contain", "ض.م.ن", "ح.و.ي", "ش.م.ل", "م.ل.أ", "ف.ر.غ", "ض.#.ف")
_add("open", "ف.ت.ح", "غ.ل.ق", "ب.و.ب", "ق.ف.ل", "ك.ش.ف", "س.ت.ر")
_add("hold", "م.س.ك", "ق.ب.ض", "ع.ل.ق", "ح.م.ل", "ر.ف.ع")
_add("hide", "خ.ف.ي", "س.ت.ر", "ك.ت.م", "غ.ي.ب", "ح.ج.ب", "خ.ب.أ", "ب.ط.ن")
_add("gather", "ج.م.ع", "ح.ش.د", "ض.م.م", "ل.م.م", "ج.ن.ي", "ح.ص.ل", "ح.#.ل")
_add("send", "ر.س.ل", "ب.ع.ث", "و.ج.ه", "ن.ق.ل", "ب.ث.ث")
_add("social", "ش.ر.ك", "ج.و.ر", "أ.ه.ل", "ق.و.م", "ش.ع.ب", "أ.م.م", "ق.ب.ل", "ح.ز.ب")
_add("dwell", "س.ك.ن", "ع.م.ر", "ب.ن.ي", "ن.ز.ل", "أ.ق.م")
_add("need", "ح.و.ج", "ل.ز.م", "ض.ر.ر", "و.ج.ب")
_add("want", "ط.ل.ب", "ر.غ.ب", "ت.م.ن", "ش.ه.و", "ب.غ.ي")
_add("enable", "م.ك.ن", "أ.ذ.ن", "س.م.ح", "ق.د.ر", "ي.س.ر")
_add("decide", "ق.ر.ر", "ح.ك.م", "ف.ص.ل", "ع.ز.م")
_add("fix", "ص.ل.ح", "ر.م.م", "ع.د.ل", "ض.ب.ط")
_add("rest", "ر.ا.ح", "ن.و.م", "ه.د.أ", "و.ق.ف", "ت.و.ق")
_add("person", "ب.ش.ر", "إ.ن.س", "ر.ج.ل", "م.ر.أ", "ط.ف.ل", "ش.ي.خ", "ش.ب.ب", "ن.س.ب")
_add("name", "س.م.ي", "ل.ق.ب", "ع.ن.و", "و.س.م")
_add("art", "ف.ن.ن", "ج.م.ل", "ز.خ.ر", "ن.ق.ش", "ر.س.م", "ل.ح.ن", "غ.ن.ي", "ع.ز.ف", "ر.ق.ص", "م.ث.ل", "ص.و.ر")
_add("science", "ب.ح.ث", "ن.ظ.ر", "ح.ل.ل", "ق.ي.س", "ك.ش.ف", "ف.ح.ص")
_add("tech", "ت.ق.ن", "ب.ر.م", "ش.ب.ك", "ه.ن.د")
_add("material", "م.ع.د", "ح.ج.ر", "ح.د.د", "ذ.ه.ب", "ف.ض.ض", "ن.ح.س", "خ.ش.ب", "ز.ج.ج", "ق.م.ش", "ن.س.ج", "ذ.ل.ل")
_add("structure", "ش.ك.ل", "ه.ي.ك", "ن.ظ.م", "ص.ف.ف", "ر.ت.ب", "ط.ب.ق")
_add("quality", "ص.ف.ي", "ج.و.د", "ح.س.ن", "س.و.أ", "ن.ظ.ف", "ق.ب.ح", "ج.د.د", "ق.د.م", "ص.ع.ب", "س.ه.ل", "ك.م.م")
_add("sport", "ل.ع.ب", "ر.ي.ض", "س.ب.ق", "ف.و.ز", "ه.ز.م")
_add("work", "ف.ع.ل", "ن.ف.ذ")

# Additional high-frequency roots
_add("exist", "ك.#.ن", "م.#.#", "#.ج.د", "ح.#.#", "#.ل.#", "#.#.ض", "#.ف.#", "ج.#.#", "#.#.ن", "ه.#.#", "ش.#.#", "#.ن.ن", "#.ن.#", "#.#.م")
_add("speak", "#.#.ل", "ح.#.ث", "ل.غ.#", "#.ل", "م.ن", "ب.م", "س.ب.ب", "ب.ل.غ", "د.ع.#", "ل.#", "س.#.ل")
_add("move", "د.#.ل", "خ.#.ر", "م.ر.ر", "ط.ر.ق", "س.ب.ل", "#.ف.ر", "س.ر.ع")
_add("time", "#.خ.ر", "ب.د.#", "م.د.د", "د.#.ر", "ع.ن.د", "#.ر.خ", "س.ن.#", "خ.ل.ف", "ص.ب.ح", "ش.ه.ر")
_add("change", "ح.#.ل", "غ.#.ر", "#.ث.ر", "م.#.ز", "خ.ف.ض")
_add("measure", "#.ح.د", "ع.#.د", "ث.ن.#", "ث.ل.ث", "د.ر.ج", "ر.ب.ع", "د.ق.ق")
_add("quality", "ت.م.م", "ك.م.م", "ف.ض.ل", "ك.م.ل")
_add("body", "ر.#.س")
_add("place", "#.ر.ض", "#.س.ط", "#.ق.ع", "ع.ر.ب", "ج.ز.ر", "ج.ز.#")
_add("nature", "ر.#.ض", "س.ط.ح", "ح.#.ط", "ع.ذ.ب", "ج.ر.د", "م.ل.ح", "ن.ج.م", "ق.م.ر")
_add("govern", "ق.#.د", "ن.#.خ", "ج.م.ه.ر")
_add("force", "ق.#.ي", "غ.ل.ب", "ج.ب.ر", "ر.غ.م", "س.ل.ب")
_add("contain", "ض.#.ف", "غ.ل.ف", "خ.ز.ن")
_add("connect", "#.ل", "ع.ق.د")
_add("person", "ن.س.ب", "ف.ر.د", "ن.ف.س")
_add("weather", "ب.خ.ر")
_add("know", "ح.ق.ق", "ث.ب.ت", "ب.د.ه", "ك.#.ف", "د.ل.ل", "ج.ر.ب")
_add("tech", "ه.ن.#")
_add("enable", "#.ف.ق", "ج.#.ز", "ك.ف.#", "ط.#.ق")
_add("see", "ش.#.ر", "ظ.ه.ر", "ب.ر.ز", "#.ز.ر", "#.ض.ح")
_add("want", "#.ر.د", "ق.ص.د")
_add("write", "ر.م.ز", "#.ت.ر")
_add("structure", "ق.ط.ع", "ق.س.م", "ص.ن.ف", "ن.ق.ط", "ف.ر.ع", "ع.ن.ص.ر")
_add("trade", "#.ق.د", "س.ه.م")
_add("fight", "ج.#.ش", "ع.س.ك.ر")
_add("size", "#.ل.ف", "ك.ث.ف", "ح.ج.م", "ك.ل.ل")
_add("social", "ه.م.م", "س.ل.م", "ح.ل.ف", "ع.ض.#")
_add("make", "ج.#.ل", "ج.ع.ل", "ص.#.غ")
_add("destroy", "ع.د.م")
_add("think", "ف.ل.س.ف")
_add("enable", "د.ع.م")
_add("create", "ن.#.س")
_add("give", "#.ز.ع")
_add("contain", "ن.#.ع")
_add("name", "ر.#.م")
_add("science", "ر.ك.ز", "م.ر.س")
_add("place", "ج.غ.ر.ف")
_add("decide", "ع.م.د")
_add("work", "ف.ع.ل", "ن.ف.ذ")
_add("exist", "م.#", "ب", "ل", "ه", "ف.#")

# ═══════════════════════════════════════════════════════════════
# Arabic function words → CST tokens (aligned with cst-spec.ts v1.0)
#
# Old: FUNC:PREP, FUNC:CONJ, etc.
# New: REL:<relation>, STR:<marker>, LIT:<surface>, ROOT:size
# ═══════════════════════════════════════════════════════════════

# Word → REL token (prepositions, conjunctions, quantifiers, etc.)
ARABIC_REL_MAP = {
    # Prepositions → REL:<specific>
    "في": "REL:in", "من": "REL:from", "إلى": "REL:to", "على": "REL:on",
    "عن": "REL:about", "مع": "REL:with", "بين": "REL:between", "حول": "REL:around",
    "خلال": "REL:through", "منذ": "REL:from", "حتى": "REL:until", "نحو": "REL:to",
    "لدى": "REL:at", "عند": "REL:at", "فوق": "REL:above", "تحت": "REL:under",
    "أمام": "REL:before", "خلف": "REL:behind", "بعد": "REL:after", "قبل": "REL:before",
    "دون": "REL:without", "ضد": "REL:against", "عبر": "REL:across", "ضمن": "REL:within",
    "لأجل": "REL:for", "بما": "REL:with",
    # Conjunctions → REL:<specific>
    "و": "REL:and", "أو": "REL:or", "ثم": "REL:then", "لكن": "REL:but",
    "بل": "REL:instead", "أم": "REL:or", "إذ": "REL:as",
    "كي": "REL:for", "حيث": "REL:where", "لأن": "REL:causes",
    "بينما": "REL:contrast", "كما": "REL:like", "مثل": "REL:like",
    "حين": "REL:when", "عندما": "REL:when", "لما": "REL:when",
    # إنّ وأخواتها (sisters of إنّ) → mapped to specific relations
    "لكنّ": "REL:but", "لكنه": "REL:but", "لكنها": "REL:but",
    "كأن": "REL:like", "كأنّ": "REL:like", "كأنه": "REL:like", "كأنها": "REL:like",
    "لعلّ": "REL:maybe", "لعله": "REL:maybe", "لعلها": "REL:maybe",
    # أدوات الاستثناء (exception) → REL:except
    "إلا": "REL:except", "سوى": "REL:except", "عدا": "REL:except", "خلا": "REL:except",
    # Demonstratives/Relatives → REL:<referential>
    "هذا": "REL:this", "هذه": "REL:this", "ذلك": "REL:those", "تلك": "REL:those",
    "الذي": "REL:which", "التي": "REL:which", "الذين": "REL:who", "اللذين": "REL:who",
    "اللاتي": "REL:who", "ما": "REL:what", "هؤلاء": "REL:these",
    # Determiners/Quantifiers → REL:<quantifier>
    "كل": "REL:all", "بعض": "REL:some", "أي": "REL:any", "غير": "REL:unlike",
    "كلا": "REL:both", "جميع": "REL:all", "سائر": "REL:all", "معظم": "REL:most",
    "أغلب": "REL:most", "عدة": "REL:several", "كثير": "REL:many", "قليل": "REL:few",
    "أكثر": "REL:more", "أحد": "REL:some", "أقل": "REL:less",
    # Restriction → REL:only
    "إنما": "REL:only", "إنّما": "REL:only",
    # Adverbs → REL:<specific>
    "أيضا": "REL:also", "أيضاً": "REL:also", "جدا": "REL:emphasis", "جداً": "REL:emphasis",
    "فقط": "REL:only", "تقريبا": "REL:almost", "تقريباً": "REL:almost",
    "حاليا": "REL:now", "حالياً": "REL:now",
}

# Words that emit as LIT:<word> (personal pronouns, auxiliaries, particles)
ARABIC_LIT_WORDS = {
    # Personal pronouns → LIT (like English I/he/she)
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "أنتم",
    "أنتِ", "أنتن", "أنتنّ", "هما",
    # Possessive/reflexive
    "نفس", "ذات",
    # Auxiliaries (كان وأخواتها) → LIT
    "كان", "يكون", "أصبح", "ظل", "بات", "صار", "ليس",
    # Subordinating particles → LIT
    "إن", "أن", "أنّ", "لعل",
    # Vocative → LIT (يا has low semantic content)
    "يا",
}

# Words that trigger STR markers (sentence-level, detected separately)
ARABIC_STR_TRIGGERS = {
    # Negation → STR:negation
    "لا": "STR:negation", "لم": "STR:negation", "لن": "STR:negation", "ليس": "STR:negation",
    # Conditional → STR:condition
    "إذا": "STR:condition", "لو": "STR:condition", "لولا": "STR:condition",
    # Future → STR:future
    "سوف": "STR:future",
    # Question → STR:question
    "هل": "STR:question",
    # Emphasis/past → STR:emphasis / STR:past
    "قد": "STR:past", "لقد": "STR:emphasis",
    # إنّ as emphasis (when standalone, not لكنّ/كأنّ which are in REL)
    "إنّ": "STR:emphasis",
}

# Numerals → ROOT:size
ARABIC_NUMERALS = {
    "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية",
    "تسعة", "عشرة", "عشر", "مئة", "مائة", "ألف", "مليون", "ثلث",
}

PROCLITICS = ["وال", "وب", "ول", "وك", "فال", "فب", "فل", "ال", "لل", "بال", "كال"]


# ═══════════════════════════════════════════════════════════════
# Arabic pattern (وزن) → CMP role mapping
# The core of the Arabic algebra: root × pattern = concept
# Patterns normalized: vowel diacritics stripped, shadda (ّ) preserved
# ═══════════════════════════════════════════════════════════════

def _strip_vowels(text):
    """Strip vowel diacritics but keep shadda (ّ \u0651)."""
    if not text: return ""
    return re.sub(r'[\u064B-\u0650\u0652\u0670]', '', text)

ARABIC_PATTERN_TO_ROLE = {
    # Active participle (فَاعِل) → agent (the doer)
    "فاعل": "agent", "فاعلة": "agent", "فاعلون": "agent",
    "فاعلات": "agent", "فاعلين": "agent", "فواعل": "agent",
    # Passive participle (مَفْعُول) → patient (the receiver)
    "مفعول": "patient", "مفعولة": "patient",
    # Place noun (مَفْعَلَة / مَفْعَل)
    "مفعلة": "place", "مفاعل": "place",
    # Instrument (مِفْعَال / مِفْعَل)
    "مفعال": "place",
    # Verbal nouns → instance (the thing) / state (the act)
    "فعال": "instance",       # كِتَاب (book)
    "فعول": "instance",       # دُخُول (entry)
    "فعل": "instance",        # عِلْم (knowledge)
    "فعالة": "state",         # كِتَابَة (writing)
    "فعولة": "state",         # عُبُودَة
    "تفعيل": "instance",      # Form II VN: تعليم (teaching)
    "تفعلة": "instance",      # Form II VN variant
    "انفعال": "instance",     # Form VII VN
    "افتعال": "instance",     # Form VIII VN
    "استفعال": "instance",    # Form X VN
    # Mutual action (Form VI)
    "تفاعل": "mutual",        # تَعَاوُن (cooperation)
    # Process (Form III verbal noun)
    "مفاعلة": "process",      # مُكَاتَبَة (correspondence)
    # Intensifier (فَعَّال — has shadda, distinct from فَعَال)
    "فعّال": "intensifier", "فعّالة": "intensifier",
    # Form II active participle (مُفَعِّل → causer)
    "مفعّل": "causer", "مفعّلة": "causer",
    # Form X active participle (مُسْتَفْعِل → seeker)
    "مستفعل": "seeker", "مستفعلة": "seeker",
    # Quality / adjective patterns
    "فعيل": "quality", "فعيلة": "quality",
    "فعلان": "quality",
    "فعلى": "quality",        # feminine elative
}

# POS-based fallback (when pattern doesn't match or is absent)
POS_TO_ROLE = {
    "adj": "quality",
    "adj_comp": "quality",
    "adj_num": "quality",
}

# POS values that indicate named entities → emit LIT:<surface>
NER_POS = frozenset({"noun_prop"})


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

def _build_wildcard_index():
    index = dict(ARABIC_ROOT_TO_FIELD)
    weak_letters = set("وياأإآءئؤ")
    for root, field in list(ARABIC_ROOT_TO_FIELD.items()):
        parts = root.split(".")
        if len(parts) != 3:
            continue
        for i in range(3):
            if parts[i] in weak_letters:
                v = list(parts); v[i] = "#"
                k = ".".join(v)
                if k not in index: index[k] = field
        for i in range(3):
            for j in range(i+1, 3):
                if parts[i] in weak_letters and parts[j] in weak_letters:
                    v = list(parts); v[i] = "#"; v[j] = "#"
                    k = ".".join(v)
                    if k not in index: index[k] = field
    return index


class ArabicCSTTokenizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.vocab: dict[str, int] = {}
        self.next_id = 0
        self.root_index = _build_wildcard_index()
        self.stats = Counter()

        # Special tokens (aligned with cst-spec.ts v1.0)
        for tok in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]:
            self._get_id(tok)

        # Pre-register ROOT tokens
        for f in sorted(set(self.root_index.values())):
            self._get_id(f"ROOT:{f}")

        # Pre-register REL tokens
        for rel in sorted(set(ARABIC_REL_MAP.values())):
            self._get_id(rel)

        # Pre-register STR tokens
        for stk in sorted(set(ARABIC_STR_TRIGGERS.values())):
            self._get_id(stk)

        # ROOT:size for numerals
        self._get_id("ROOT:size")

    def _get_id(self, token):
        if token in self.vocab: return self.vocab[token]
        tid = self.next_id; self.vocab[token] = tid; self.next_id += 1
        return tid

    def _strip(self, word):
        word = re.sub(r'[\u064B-\u065F\u0670]', '', word)
        return word.replace('\u0640', '')

    def _find_field(self, roots):
        for r in roots:
            if r in self.root_index: return self.root_index[r]
        return None

    def _analyze(self, clean):
        """Return (field, role, is_named_entity).

        field: semantic field str or None
        role:  CMP role str or None (emit ROOT if None, CMP if set)
        is_named_entity: bool (emit LIT:<surface> if True)
        """
        analyses = self.analyzer.analyze(clean)
        valid = [a for a in analyses
                 if a.get("root","") not in ("","NTWS","PUNC","DIGIT","FOREIGN")]
        if not valid:
            # Proclitic fallback: strip common prefixes and re-analyze
            for prefix in PROCLITICS:
                if clean.startswith(prefix) and len(clean) > len(prefix) + 1:
                    stem = clean[len(prefix):]
                    stem_a = self.analyzer.analyze(stem)
                    valid = [a for a in stem_a
                             if a.get("root","") not in ("","NTWS","PUNC","DIGIT","FOREIGN")]
                    if valid:
                        break
        if not valid:
            return None, None, False

        # Named entity detection (noun_prop → LIT)
        if any(a.get("pos") in NER_POS for a in valid):
            return None, None, True

        # Find semantic field from roots
        roots = [a.get("root","") for a in valid]
        field = self._find_field(roots)
        if not field:
            return None, None, False

        # Extract morphological role from pattern or POS
        role = self._extract_role(valid)
        return field, role, False

    def _extract_role(self, analyses):
        """Extract CMP role from camel-tools pattern or POS."""
        for a in analyses:
            # 1. Pattern-based (most precise — the وزن system)
            pattern = a.get("pattern") or ""
            norm = _strip_vowels(pattern)
            if norm and norm in ARABIC_PATTERN_TO_ROLE:
                return ARABIC_PATTERN_TO_ROLE[norm]
            # 2. POS-based fallback
            pos = a.get("pos", "")
            if pos in POS_TO_ROLE:
                return POS_TO_ROLE[pos]
        return None

    def tokenize(self, sentence):
        tokens, ids = ["[BOS]"], [self.vocab["[BOS]"]]

        # Detect sentence-level STR markers first
        words = re.findall(r'[\u0600-\u06FF\u0750-\u077F]+', sentence)
        str_emitted = set()
        for word in words:
            clean = self._strip(word)
            if clean in ARABIC_STR_TRIGGERS:
                marker = ARABIC_STR_TRIGGERS[clean]
                if marker not in str_emitted:
                    tokens.append(marker); ids.append(self._get_id(marker))
                    str_emitted.add(marker)
                    self.stats["str"] += 1
        # Also check punctuation
        if sentence.rstrip().endswith("؟") or sentence.rstrip().endswith("?"):
            if "STR:question" not in str_emitted:
                self._get_id("STR:question")  # register if needed
                tokens.append("STR:question"); ids.append(self._get_id("STR:question"))
                self.stats["str"] += 1
        if sentence.rstrip().endswith("!"):
            if "STR:emphasis" not in str_emitted:
                self._get_id("STR:emphasis")
                tokens.append("STR:emphasis"); ids.append(self._get_id("STR:emphasis"))
                self.stats["str"] += 1
        # Check for سـ future prefix (sa + imperfective verb prefix)
        for word in words:
            clean = self._strip(word)
            if len(clean) > 3 and clean[0] == 'س' and clean[1] in 'يتنأ':
                if "STR:future" not in str_emitted:
                    self._get_id("STR:future")
                    tokens.append("STR:future"); ids.append(self._get_id("STR:future"))
                    str_emitted.add("STR:future")
                    self.stats["str"] += 1
                break

        # Word-by-word tokenization
        for word in words:
            clean = self._strip(word)

            # STR trigger words are consumed above, skip as word tokens
            if clean in ARABIC_STR_TRIGGERS:
                continue

            # 1. REL tokens (prepositions, conjunctions, quantifiers, etc.)
            if clean in ARABIC_REL_MAP:
                tok = ARABIC_REL_MAP[clean]
                tokens.append(tok); ids.append(self._get_id(tok))
                self.stats["rel"] += 1; continue

            # 2. LIT tokens (personal pronouns, auxiliaries)
            if clean in ARABIC_LIT_WORDS:
                tok = f"LIT:{clean}"
                tokens.append(tok); ids.append(self._get_id(tok))
                self.stats["lit"] += 1; continue

            # 3. Numerals → ROOT:size
            if clean in ARABIC_NUMERALS:
                tok = "ROOT:size"
                tokens.append(tok); ids.append(self._get_id(tok))
                self.stats["root"] += 1; continue

            # 4. Morphological analysis → CMP:<field>:<role> or ROOT:<field>
            field, role, is_ner = self._analyze(clean)
            if is_ner:
                tok = f"LIT:{clean}"
                tokens.append(tok); ids.append(self._get_id(tok))
                self.stats["ner"] += 1
            elif field and role:
                tok = f"CMP:{field}:{role}"
                tokens.append(tok); ids.append(self._get_id(tok))
                self.stats["cmp"] += 1
            elif field:
                tok = f"ROOT:{field}"
                tokens.append(tok); ids.append(self._get_id(tok))
                self.stats["root"] += 1
            else:
                # 5. Surface fallback → LIT:<surface>
                tok = f"LIT:{clean}"
                tokens.append(tok); ids.append(self._get_id(tok))
                self.stats["lit"] += 1

        tokens.append("[EOS]"); ids.append(self.vocab["[EOS]"])
        return {"ids": ids, "tokens": tokens, "text": sentence}

    def save_vocab(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# Download — uses HuggingFace `datasets` library (no rate limits)
# ═══════════════════════════════════════════════════════════════

def download_sentences(target, output_path):
    if os.path.exists(output_path):
        print(f"  Loading cached: {output_path}")
        with open(output_path) as f: return json.load(f)

    from datasets import load_dataset

    print(f"  Downloading Arabic Wikipedia via `datasets` library...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)

    sentences = []
    articles = 0
    t0 = time.time()

    for row in ds:
        text = row.get("text", "")
        for sent in re.split(r'[.؟!]\s*', text):
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 300:
                continue
            if sum(1 for c in sent if '\u0600' <= c <= '\u06FF') < len(sent) * 0.5:
                continue
            sentences.append(sent)
            if len(sentences) >= target:
                break
        if len(sentences) >= target:
            break
        articles += 1
        if articles % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {len(sentences):,} / {target:,} sentences from {articles:,} articles ({elapsed:.0f}s)")

    sentences = sentences[:target]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=0)
    elapsed = time.time() - t0
    print(f"  Saved {len(sentences):,} sentences ({elapsed:.0f}s)")
    return sentences


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    TARGET = 1_000_000
    DATA_DIR = "/content"
    GDRIVE_DIR = "/content/drive/MyDrive/cst-data"  # from download_data.py
    OUT_DIR = "/content/cst_1m"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Try Google Drive first (pre-downloaded), then local, then download
    sentences_path = f"{DATA_DIR}/sentences-{TARGET}.json"
    gdrive_path = os.path.join(GDRIVE_DIR, "sentences-1M.json")

    print("=" * 60)
    print(f"  Arabic CST Tokenization — {TARGET:,} sentences")
    print("=" * 60)

    # Step 1: Load sentences
    print("\n── Step 1: Load sentences ──")
    if os.path.exists(gdrive_path):
        print(f"  Loading from Google Drive: {gdrive_path}")
        with open(gdrive_path) as f:
            sentences = json.load(f)
        print(f"  Loaded {len(sentences):,} sentences")
    else:
        sentences = download_sentences(TARGET, sentences_path)
    print(f"  Total: {len(sentences):,} sentences")

    # Step 2: Tokenize
    print("\n── Step 2: CST Tokenize ──")
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer

    print("  Loading camel-tools...")
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)
    tokenizer = ArabicCSTTokenizer(analyzer)

    n = len(sentences)
    output_path = os.path.join(OUT_DIR, f"train-{n}.jsonl")
    lines = []
    t0 = time.time()

    for i, sent in enumerate(sentences):
        result = tokenizer.tokenize(sent)
        if len(result["ids"]) < 4: continue
        lines.append(json.dumps(result, ensure_ascii=False))
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate / 60
            print(f"    {i+1:,} / {n:,} ({elapsed:.0f}s, {rate:.0f} sent/s, ETA {eta:.0f}min)")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    vocab_path = output_path.replace(".jsonl", "-vocab.json")
    tokenizer.save_vocab(vocab_path)

    elapsed = time.time() - t0
    total_tok = sum(len(json.loads(l)["ids"]) for l in lines)
    stats = dict(tokenizer.stats)
    total_s = sum(stats.values())

    print(f"\n  ═══ Done ═══")
    print(f"  Sentences:  {len(lines):,}")
    print(f"  Vocab size: {tokenizer.next_id:,}")
    print(f"  Tokens:     {total_tok:,}")
    print(f"  Avg tok/s:  {total_tok/len(lines):.1f}")
    print(f"  Time:       {elapsed/60:.1f} min")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {k:12s} {v:10,} ({v/total_s*100:5.1f}%)")

    print(f"\n  Output: {output_path}")
    print(f"  Vocab:  {vocab_path}")
    print(f"\n  Next: upload to Colab and run colab_edge.py with:")
    print(f'    DATA_FILE = "train-{n}.jsonl"')
    print(f'    VOCAB_FILE = "train-{n}-vocab.json"')


if __name__ == "__main__":
    main()
