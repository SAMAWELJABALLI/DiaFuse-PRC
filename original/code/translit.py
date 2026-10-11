"""Rule-based Arabic-script -> Latin-script (Arabizi) generators and perturbations.

G_seen  : deterministic, digit-heavy Tunisian conventions (3 7 5 8 9 2, ch, ou, i, el).
G_unseen: different rule set with digraphs and stochastic choices (a/e, h, kh, gh, q,
          sh, u/oo, ee/i, l/il, hamza dropped), seeded by the sentence so that the output
          is reproducible. G_unseen is used only to build evaluation sets.
These implement exactly the convention table of Appendix D of the paper.
"""
import hashlib
import random
import re
import unicodedata

AR_RANGE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LAT_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")
DIACRITICS = re.compile(r"[ً-ْٰـ]")  # harakat, dagger alef, tatweel

VOWEL_LETTERS = set("اوي")
SEEN = {
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "7", "خ": "5", "د": "d", "ذ": "dh",
    "ر": "r", "ز": "z", "س": "s", "ش": "ch", "ص": "s", "ض": "dh", "ط": "t", "ظ": "dh",
    "ع": "3", "غ": "8", "ف": "f", "ق": "9", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "ة": "a", "ء": "2", "ئ": "2", "ؤ": "2", "ى": "a", "پ": "p", "ڤ": "v",
    "گ": "g", "ک": "k", "ی": "i", "ڨ": "g",
}
UNSEEN = dict(SEEN)
UNSEEN.update({"ح": "h", "خ": "kh", "غ": "gh", "ق": "q", "ع": "a", "ش": "sh", "ذ": "d",
               "ض": "d", "ظ": "z", "ء": "", "ئ": "", "ؤ": ""})


def script_of(text):
    ar = len(AR_RANGE.findall(text)); la = len(LAT_LETTER.findall(text))
    if ar + la == 0:
        return "none"
    if min(ar, la) / (ar + la) > 0.2:
        return "mixed"
    return "ar" if ar > la else "lat"


def _rng(word, salt):
    h = hashlib.md5((salt + word).encode("utf-8")).hexdigest()
    return random.Random(int(h[:8], 16))


def _translit_word(w, table, unseen):
    w = DIACRITICS.sub("", w)
    if not AR_RANGE.search(w):
        return w
    rng = _rng(w, "gu") if unseen else None
    out = []
    if w.startswith("ال") and len(w) > 3:
        out.append(rng.choice(["l", "il"]) if unseen else "el ")
        w = w[2:]
    prev_cons = False
    n = len(w)
    for i, ch in enumerate(w):
        nxt = w[i + 1] if i + 1 < n else ""
        if ch in "اأإآ":
            out.append("i" if ch == "إ" else "a"); prev_cons = False
        elif ch == "و":
            if i == 0 or nxt in VOWEL_LETTERS or (not prev_cons):
                out.append("w"); prev_cons = True
            else:
                out.append(rng.choice(["u", "oo"]) if unseen else "ou"); prev_cons = False
        elif ch == "ي":
            if i == 0 or nxt in VOWEL_LETTERS or (not prev_cons):
                out.append("y"); prev_cons = True
            else:
                out.append(rng.choice(["ee", "i"]) if unseen else "i"); prev_cons = False
        elif ch in table:
            lat = table[ch]
            # vowel epenthesis inside consonant runs
            if prev_cons and nxt and nxt not in VOWEL_LETTERS and lat:
                if unseen:
                    if rng.random() < 0.5:
                        out.append(rng.choice(["a", "e"]))
                else:
                    out.append("e")
            out.append(lat); prev_cons = bool(lat)
        elif AR_RANGE.match(ch):
            continue  # punctuation or rare letters in the Arabic block
        else:
            out.append(ch); prev_cons = False
    return "".join(out)


def g_seen(text):
    return " ".join(_translit_word(w, SEEN, False) for w in text.split())


def g_unseen(text):
    return " ".join(_translit_word(w, UNSEEN, True) for w in text.split())


def partial(text, gen, frac, seed=0):
    toks = text.split()
    r = random.Random(seed)
    k = round(frac * len(toks))
    idx = set(r.sample(range(len(toks)), k)) if toks else set()
    return " ".join(gen(t) if i in idx else t for i, t in enumerate(toks))


# ---------------------------------------------------------------- perturbations on Latin text
def inf_spelling(text, frac=0.2, seed=0):
    """Letter repetition and vowel variation (family INF)."""
    r = random.Random(seed); out = []
    swaps = {"ou": "u", "e": "a", "a": "e", "i": "y"}
    for t in text.split():
        if r.random() < frac and len(t) > 2:
            if r.random() < 0.5:
                t = t + t[-1] * r.randint(1, 3)
            else:
                k = r.choice(list(swaps))
                t = t.replace(k, swaps[k], 1)
        out.append(t)
    return " ".join(out)


def inf_abbrev(text, frac=0.2, seed=0):
    """Vowel deletion (consonant skeletons) of long tokens (family INF)."""
    r = random.Random(seed); out = []
    for t in text.split():
        if r.random() < frac and len(t) > 3:
            t = t[0] + re.sub(r"[aeiouy]", "", t[1:])
        out.append(t)
    return " ".join(out)


def char_noise(text, frac=0.2, seed=0):
    r = random.Random(seed); out = []
    letters = "abcdefghijklmnopqrstuvwxyz"
    for t in text.split():
        if r.random() < frac and len(t) > 2:
            i = r.randrange(len(t) - 1); op = r.choice(["swap", "del", "ins", "sub"])
            if op == "swap":
                t = t[:i] + t[i + 1] + t[i] + t[i + 2:]
            elif op == "del":
                t = t[:i] + t[i + 1:]
            elif op == "ins":
                t = t[:i] + r.choice(letters) + t[i:]
            else:
                t = t[:i] + r.choice(letters) + t[i + 1:]
        out.append(t)
    return " ".join(out)


# French insertion (family CS) on Arabic-script input, before transliteration.
CS_LEX = {
    "برشا": "trop", "باهي": "bien", "بهي": "bien", "مزيان": "beau", "مزيانة": "belle",
    "خايب": "nul", "خايبة": "nulle", "يعطيك": "merci", "نحب": "j'aime", "نحبك": "je t'aime",
    "فيلم": "film", "برنامج": "emission", "توا": "maintenant", "موش": "pas", "ماهوش": "pas",
    "ياسر": "trop", "صحة": "bravo", "عيشك": "merci", "الله": "allah",
}


def code_switch(text, frac=0.2, seed=0):
    r = random.Random(seed); out = []
    for t in text.split():
        key = DIACRITICS.sub("", t)
        if key in CS_LEX and r.random() < max(frac * 3, 0.6):
            out.append(CS_LEX[key])
        else:
            out.append(t)
    return " ".join(out)


if __name__ == "__main__":
    for s in ["الفيلم باهي برشا", "عجبني برشا", "يعطيك الصحة على البرنامج", "موش مزيان"]:
        print(s, "|", g_seen(s), "|", g_unseen(s), "|", g_seen(code_switch(s)))


# ---------------------------------------------------------------- inverse (for B0-TT)
_INV_MULTI = [("ch", "ش"), ("sh", "ش"), ("kh", "خ"), ("gh", "غ"), ("th", "ث"), ("dh", "ذ"),
              ("ou", "و"), ("oo", "و"), ("ee", "ي")]
_INV_ONE = {"3": "ع", "7": "ح", "5": "خ", "8": "غ", "9": "ق", "2": "ء", "b": "ب", "t": "ت",
            "j": "ج", "d": "د", "r": "ر", "z": "ز", "s": "س", "f": "ف", "k": "ك", "q": "ق",
            "l": "ل", "m": "م", "n": "ن", "h": "ه", "w": "و", "y": "ي", "g": "ڨ", "v": "ڤ",
            "p": "ب", "x": "كس", "c": "ك", "i": "ي", "o": "و", "u": "و"}


def latin_to_arabic(text):
    """Deterministic inverse of the G_seen rules: digits and digraphs to letters, long
    vowels to matres lectionis, short vowels dropped except word-final 'a'."""
    out_words = []
    for w in text.lower().split():
        if not re.search(r"[a-z0-9]", w) or w.isdigit():
            out_words.append(w); continue
        prefix = ""
        if w in ("el", "il", "l") :
            out_words.append("ال"); continue
        res, i = [], 0
        while i < len(w):
            two = w[i:i + 2]
            m = next((ar for lat, ar in _INV_MULTI if lat == two), None)
            if m:
                res.append(m); i += 2; continue
            ch = w[i]
            if ch == "a":
                res.append("ا" if i == 0 or i == len(w) - 1 else "")
            elif ch == "e":
                res.append("ا" if i == 0 else "")
            elif ch in _INV_ONE:
                res.append(_INV_ONE[ch])
            elif not ch.isascii():
                res.append(ch)
            i += 1
        out_words.append(prefix + "".join(res))
    # re-attach a detached article
    joined = []
    for w in out_words:
        if joined and joined[-1] == "ال":
            joined[-1] = "ال" + w
        else:
            joined.append(w)
    return " ".join(joined)
