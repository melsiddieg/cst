# CST Arabic Coverage — What Arabic Gives to CST

> CST takes the algebraic structure of Arabic morphology and makes it the universal token format.
> Arabic is not just another language in CST — **Arabic is the blueprint.**

## The Core Idea

Arabic morphology is built on a real algebraic system:

```
Root × Pattern = Meaning
ك.ت.ب × فاعل = كاتب (writer)
ك.ت.ب × مفعول = مكتوب (written)
ك.ت.ب × مفعلة = مكتبة (library)
```

CST encodes this algebra directly:

```
Root × Pattern → CMP:field:role
ك.ت.ب × فاعل  → CMP:write:agent
ك.ت.ب × مفعول  → CMP:write:patient
ك.ت.ب × مفعلة  → CMP:write:place
```

Other languages approximate the same output through their own morphology:

```
English: writer  → strip -er  → CMP:write:agent
Spanish: escritor → strip -or → CMP:write:agent
Arabic:  كاتب    → pattern فاعل → CMP:write:agent   (precise, algebraic)
```

---

## 1. الأوزان — Morphological Patterns (CMP Tokens)

The heart of CST. Arabic's وزن system maps directly to CMP roles.

### Noun / Participle Patterns

| Pattern (وزن) | Role               | Example           | CST Token           |
| ------------- | ------------------ | ----------------- | ------------------- |
| فَاعِل        | agent (doer)       | كاتب (writer)     | `CMP:write:agent`   |
| فَاعِلة       | agent (f.)         | كاتبة (writer f.) | `CMP:write:agent`   |
| فَاعِلون      | agent (pl.)        | كاتبون (writers)  | `CMP:write:agent`   |
| فَاعِلات      | agent (f.pl.)      | كاتبات            | `CMP:write:agent`   |
| فَواعِل       | agent (broken pl.) | كواتب             | `CMP:write:agent`   |
| مَفْعُول      | patient (receiver) | مكتوب (written)   | `CMP:write:patient` |
| مَفْعُولة     | patient (f.)       | مكتوبة            | `CMP:write:patient` |
| مَفْعَلة      | place              | مكتبة (library)   | `CMP:write:place`   |
| مَفاعِل       | place (pl.)        | مكاتب (offices)   | `CMP:write:place`   |
| مِفْعال       | instrument/place   | مفتاح (key)       | `CMP:open:place`    |
| فَعِيل        | quality            | كبير (big)        | `CMP:size:quality`  |
| فَعِيلة       | quality (f.)       | كبيرة             | `CMP:size:quality`  |
| فَعْلان       | quality            | عطشان (thirsty)   | `CMP:food:quality`  |
| فَعْلى        | quality (f.)       | كبرى              | `CMP:size:quality`  |

### Verbal Noun Patterns (المصادر)

| Pattern     | Role                 | Example            | CST Token              |
| ----------- | -------------------- | ------------------ | ---------------------- |
| فِعال       | instance (thing)     | كِتاب (book)       | `CMP:write:instance`   |
| فُعول       | instance             | دُخول (entry)      | `CMP:move:instance`    |
| فَعْل       | instance             | عِلْم (knowledge)  | `CMP:know:instance`    |
| فِعالة      | state (act)          | كِتابة (writing)   | `CMP:write:state`      |
| فُعولة      | state                | عُبودة (servitude) | `CMP:work:state`       |
| تَفْعيل     | instance (Form II)   | تَعليم (teaching)  | `CMP:know:instance`    |
| تَفْعِلة    | instance (Form II)   | تجربة (experiment) | `CMP:science:instance` |
| اِنْفِعال   | instance (Form VII)  | انكسار (breaking)  | `CMP:destroy:instance` |
| اِفْتِعال   | instance (Form VIII) | اجتماع (meeting)   | `CMP:gather:instance`  |
| اِسْتِفْعال | instance (Form X)    | استخدام (usage)    | `CMP:work:instance`    |

### Derived Agent Patterns

| Pattern      | Role             | Example                  | CST Token                 |
| ------------ | ---------------- | ------------------------ | ------------------------- |
| فَعَّال      | intensifier      | كتّاب (prolific writer)  | `CMP:write:intensifier`   |
| فَعَّالة     | intensifier (f.) | غسّالة (washing machine) | `CMP:quality:intensifier` |
| مُفَعِّل     | causer (Form II) | مُعلِّم (teacher)        | `CMP:know:causer`         |
| مُفَعِّلة    | causer (f.)      | مُعلِّمة                 | `CMP:know:causer`         |
| مُسْتَفْعِل  | seeker (Form X)  | مُستخدِم (user)          | `CMP:work:seeker`         |
| مُسْتَفْعِلة | seeker (f.)      | مُستشفية                 | `CMP:health:seeker`       |

### Reciprocal / Process Patterns

| Pattern  | Role               | Example                   | CST Token           |
| -------- | ------------------ | ------------------------- | ------------------- |
| تَفاعُل  | mutual (Form VI)   | تعاوُن (cooperation)      | `CMP:enable:mutual` |
| مُفاعَلة | process (Form III) | مُكاتَبة (correspondence) | `CMP:write:process` |

### POS Fallback

When pattern is not recognized but POS is adjective:

| POS      | Role    | Example               | CST Token         |
| -------- | ------- | --------------------- | ----------------- |
| adj      | quality | جميل (beautiful)      | `CMP:art:quality` |
| adj_comp | quality | أجمل (more beautiful) | `CMP:art:quality` |

---

## 2. الجذور — Trilateral Roots (ROOT Tokens)

When a root is recognized but no pattern match → `ROOT:field`.

58 semantic fields, each mapped from Arabic trilateral roots:

| Field | Sample Roots        | Example                    |
| ----- | ------------------- | -------------------------- |
| write | ك.ت.ب، خ.ط.ط، س.ج.ل | كتب → `ROOT:write`         |
| know  | ع.ل.م، ع.ر.ف، د.ر.س | علم → `ROOT:know`          |
| speak | ق.و.ل، ك.ل.م، ح.د.ث | قال → `ROOT:speak`         |
| move  | م.ش.ي، ذ.ه.ب، ر.ج.ع | ذهب → `ROOT:move`          |
| feel  | ح.ب.ب، ش.ع.ر، ح.ز.ن | حب → `ROOT:feel`           |
| ...   | ...                 | (500+ root mappings total) |

### Weak Root Handling

Arabic weak roots (containing و/ي/ا) have variant forms. CST uses a wildcard index:

```
و.ج.د → exist    (standard)
#.ج.د → exist    (wildcard: و replaced)
```

This catches conjugated forms where weak letters shift or disappear.

---

## 3. حروف الجر — Prepositions (REL Tokens)

Every Arabic preposition maps to a specific spatial/logical relation:

| Arabic | Meaning            | CST Token     |
| ------ | ------------------ | ------------- |
| في     | in                 | `REL:in`      |
| من     | from               | `REL:from`    |
| إلى    | to                 | `REL:to`      |
| على    | on                 | `REL:on`      |
| عن     | about              | `REL:about`   |
| مع     | with               | `REL:with`    |
| بين    | between            | `REL:between` |
| حول    | around             | `REL:around`  |
| خلال   | through            | `REL:through` |
| منذ    | since              | `REL:from`    |
| حتى    | until              | `REL:until`   |
| نحو    | toward             | `REL:to`      |
| لدى    | at (possession)    | `REL:at`      |
| عند    | at (location/time) | `REL:at`      |
| فوق    | above              | `REL:above`   |
| تحت    | under              | `REL:under`   |
| أمام   | in front of        | `REL:before`  |
| خلف    | behind             | `REL:behind`  |
| بعد    | after              | `REL:after`   |
| قبل    | before             | `REL:before`  |
| دون    | without            | `REL:without` |
| ضد     | against            | `REL:against` |
| عبر    | across             | `REL:across`  |
| ضمن    | within             | `REL:within`  |
| لأجل   | for                | `REL:for`     |

---

## 4. حروف العطف — Conjunctions (REL Tokens)

| Arabic | Meaning        | CST Token      |
| ------ | -------------- | -------------- |
| و      | and            | `REL:and`      |
| أو     | or             | `REL:or`       |
| ثم     | then           | `REL:then`     |
| لكن    | but            | `REL:but`      |
| بل     | rather/instead | `REL:instead`  |
| أم     | or (question)  | `REL:or`       |
| إذ     | as/since       | `REL:as`       |
| كي     | in order to    | `REL:for`      |
| حيث    | where          | `REL:where`    |
| لأن    | because        | `REL:causes`   |
| بينما  | while          | `REL:contrast` |
| كما    | as/like        | `REL:like`     |
| مثل    | like           | `REL:like`     |
| حين    | when           | `REL:when`     |
| عندما  | when           | `REL:when`     |
| لما    | when           | `REL:when`     |

---

## 5. إنّ وأخواتها — Sisters of إنّ (REL / STR Tokens)

| Arabic       | Meaning            | CST Token      |
| ------------ | ------------------ | -------------- |
| إنّ          | indeed (emphasis)  | `STR:emphasis` |
| لكنّ         | but (contrast)     | `REL:but`      |
| كأنّ         | as if (comparison) | `REL:like`     |
| لعلّ         | perhaps            | `REL:maybe`    |
| لكنه / لكنها | but + pronoun      | `REL:but`      |
| كأنه / كأنها | as if + pronoun    | `REL:like`     |
| لعله / لعلها | perhaps + pronoun  | `REL:maybe`    |

---

## 6. أدوات النفي — Negation (STR Tokens)

| Arabic | Context                       | CST Token      |
| ------ | ----------------------------- | -------------- |
| لا     | general negation              | `STR:negation` |
| لم     | past negation (jussive)       | `STR:negation` |
| لن     | future negation (subjunctive) | `STR:negation` |
| ليس    | nominal negation              | `STR:negation` |

> Note: ما is **not** mapped as negation because it is ambiguous (ما النافية، ما الموصولة، ما التعجبية). It maps to `REL:what`. The model learns the distinction from context.

---

## 7. أدوات التوكيد — Emphasis (STR Tokens)

| Arabic | Context           | CST Token      |
| ------ | ----------------- | -------------- |
| إنّ    | emphasis particle | `STR:emphasis` |
| لقد    | past emphasis     | `STR:emphasis` |
| !      | exclamation mark  | `STR:emphasis` |

---

## 8. أدوات الشرط — Conditional (STR Tokens)

| Arabic | Meaning           | CST Token       |
| ------ | ----------------- | --------------- |
| إذا    | if (likely)       | `STR:condition` |
| لو     | if (hypothetical) | `STR:condition` |
| لولا   | if not for        | `STR:condition` |

---

## 9. أدوات الاستفهام — Question (STR Tokens)

| Arabic | Context               | CST Token      |
| ------ | --------------------- | -------------- |
| هل     | yes/no question       | `STR:question` |
| ؟      | question mark         | `STR:question` |
| ?      | question mark (Latin) | `STR:question` |

---

## 10. أدوات الاستقبال — Future (STR Tokens)

| Arabic            | Context         | CST Token    |
| ----------------- | --------------- | ------------ |
| سوف               | will (explicit) | `STR:future` |
| سـ + imperfective | will (prefix)   | `STR:future` |

Detection: سـ prefix is detected when the word starts with س followed by an imperfective verb prefix (ي/ت/ن/أ), e.g. سيذهب → `STR:future`.

---

## 11. الماضي — Past (STR Tokens)

| Arabic         | Context          | CST Token  |
| -------------- | ---------------- | ---------- |
| قد + past verb | completed action | `STR:past` |

---

## 12. أدوات الاستثناء — Exception (REL Tokens)

| Arabic | Meaning    | CST Token    |
| ------ | ---------- | ------------ |
| إلا    | except     | `REL:except` |
| سوى    | other than | `REL:except` |
| عدا    | apart from | `REL:except` |
| خلا    | except for | `REL:except` |

---

## 13. أدوات التقييد — Restriction (REL Tokens)

| Arabic | Meaning   | CST Token  |
| ------ | --------- | ---------- |
| إنما   | only/just | `REL:only` |
| فقط    | only      | `REL:only` |

---

## 14. أسماء الإشارة والموصول — Demonstratives & Relatives (REL Tokens)

| Arabic         | Meaning     | CST Token   |
| -------------- | ----------- | ----------- |
| هذا / هذه      | this        | `REL:this`  |
| ذلك / تلك      | that        | `REL:those` |
| هؤلاء          | these       | `REL:these` |
| الذي / التي    | which       | `REL:which` |
| الذين / اللذين | who (pl.)   | `REL:who`   |
| اللاتي         | who (f.pl.) | `REL:who`   |
| ما             | what        | `REL:what`  |

---

## 15. المحددات والكمّيات — Quantifiers (REL Tokens)

| Arabic           | Meaning      | CST Token     |
| ---------------- | ------------ | ------------- |
| كل / جميع / سائر | all          | `REL:all`     |
| بعض / أحد        | some         | `REL:some`    |
| أي               | any          | `REL:any`     |
| كلا              | both         | `REL:both`    |
| معظم / أغلب      | most         | `REL:most`    |
| عدة              | several      | `REL:several` |
| كثير             | many         | `REL:many`    |
| قليل             | few          | `REL:few`     |
| أكثر             | more         | `REL:more`    |
| أقل              | less         | `REL:less`    |
| غير              | unlike/other | `REL:unlike`  |

---

## 16. الضمائر — Pronouns (LIT Tokens)

| Arabic                | CST Token                  |
| --------------------- | -------------------------- |
| أنا، نحن              | `LIT:أنا`, `LIT:نحن`       |
| أنت، أنتِ، أنتم، أنتن | `LIT:أنت`, `LIT:أنتِ`, ... |
| هو، هي، هم، هن، هما   | `LIT:هو`, `LIT:هي`, ...    |

Pronouns are `LIT` because they are referential — they point to entities, they don't carry semantic field content.

---

## 17. كان وأخواتها — Auxiliary Verbs (LIT Tokens)

| Arabic     | Meaning         | CST Token              |
| ---------- | --------------- | ---------------------- |
| كان / يكون | was/is          | `LIT:كان` / `LIT:يكون` |
| أصبح       | became          | `LIT:أصبح`             |
| ظل         | remained        | `LIT:ظل`               |
| بات        | spent the night | `LIT:بات`              |
| صار        | became          | `LIT:صار`              |

---

## 18. الأعداد — Numerals (ROOT Tokens)

| Arabic                      | CST Token   |
| --------------------------- | ----------- |
| واحد، اثنان، ثلاثة ... تسعة | `ROOT:size` |
| عشر، عشرة، مئة، مائة        | `ROOT:size` |
| ألف، مليون                  | `ROOT:size` |

---

## 19. أسماء العلم — Named Entities (LIT Tokens)

Detected via camel-tools `pos=noun_prop`:

```
محمد → LIT:محمد     (proper noun, not analyzed for root)
القاهرة → LIT:القاهرة  (city name, preserved as surface)
```

Named entities are emitted as `LIT` by design — their identity matters more than their etymology.

---

## 20. Proclitic Handling

Arabic attaches prepositions and conjunctions as prefixes. The tokenizer strips common proclitics before morphological analysis:

| Proclitic | Components | Example                |
| --------- | ---------- | ---------------------- |
| وال       | و + ال     | والكتاب → و + الكتاب   |
| بال       | ب + ال     | بالمدرسة → ب + المدرسة |
| لل        | ل + ال     | للعلم → ل + العلم      |
| فال       | ف + ال     | فالعمل → ف + العمل     |
| ال        | the        | الكاتب → كاتب          |

After stripping, the stem is re-analyzed for root and pattern extraction.

---

## What Arabic Provides That Other Languages Approximate

| Feature           | Arabic (precise)                        | English (approximate)                        |
| ----------------- | --------------------------------------- | -------------------------------------------- |
| Root extraction   | Trilateral root system (ك.ت.ب)          | Lemmatization (writer → write)               |
| Role from pattern | وزن system (فاعل → agent) ~95% accurate | Suffix stripping (-er → agent) ~60% accurate |
| CMP emission      | Pattern-based, systematic               | Affix-based, irregular                       |
| Negation          | Dedicated particles (لا/لم/لن/ليس)      | Word detection (not/never/can't)             |
| Future            | Dedicated prefix سـ + particle سوف      | Modal detection (will/shall)                 |
| Question          | Dedicated particle هل                   | Punctuation only (?)                         |

---

## What We Intentionally Exclude

| Feature                 | Reason                                                        |
| ----------------------- | ------------------------------------------------------------- |
| إعراب (case endings)    | Dropped in modern Arabic, unreliable in unvoweled text        |
| Broken plurals          | Complex mapping, marginal gain for model training             |
| Dual number (المثنى)    | Rare in modern text                                           |
| ما as negation          | Ambiguous — same form for negation, relative, and exclamatory |
| همزة الاستفهام (أ)      | Rare, hard to distinguish from regular أ prefix               |
| Full dependency parsing | Beyond tokenizer scope — the model learns syntax from context |

---

## Token Distribution (Expected)

Based on Arabic Wikipedia text:

| Token Type       | Expected % | Meaning                                                |
| ---------------- | ---------- | ------------------------------------------------------ |
| `CMP:field:role` | ~30-40%    | Content words with detected pattern → **new with وزن** |
| `ROOT:field`     | ~15-25%    | Content words, no pattern match (fallback)             |
| `REL:relation`   | ~20-25%    | Prepositions, conjunctions, quantifiers                |
| `STR:marker`     | ~3-5%      | Sentence-level markers (negation, question, etc.)      |
| `LIT:surface`    | ~15-25%    | Pronouns, auxiliaries, unknown words, named entities   |

---

## The Algebra in One Sentence

> Arabic gives CST its algebraic foundation: **root × pattern = meaning**.
> Every other language is translated into this algebra. The model sees one universal structure, regardless of source language.
