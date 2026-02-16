import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Indian Name Match Confidence", page_icon="🧾", layout="wide")


HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "shri",
    "sri",
    "kumari",
    "smt",
    "prof",
}

COMMON_CANONICAL = {
    "mohd": "mohammad",
    "md": "mohammad",
    "mohammed": "mohammad",
    "muhammad": "mohammad",
    "syed": "sayed",
    "shaik": "sheikh",
    "shiekh": "sheikh",
}

TRANSLIT_PATTERNS = [
    (r"aa", "a"),
    (r"ee", "i"),
    (r"oo", "u"),
    (r"ou", "u"),
    (r"ph", "f"),
    (r"kh", "k"),
    (r"th", "t"),
    (r"dh", "d"),
    (r"bh", "b"),
    (r"gh", "g"),
    (r"sh", "s"),
    (r"zh", "j"),
]


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def clean_name(raw_name: str) -> str:
    text = strip_accents(raw_name.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [COMMON_CANONICAL.get(tok, tok) for tok in text.split() if tok not in HONORIFICS]
    return " ".join(tokens)


def phonetic_normalize(name: str) -> str:
    text = clean_name(name)
    for pattern, repl in TRANSLIT_PATTERNS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"(.)\1+", r"\1", text)
    return text.strip()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def damerau_levenshtein_distance(a: str, b: str) -> int:
    da = {}
    maxdist = len(a) + len(b)
    d = [[0] * (len(b) + 2) for _ in range(len(a) + 2)]
    d[0][0] = maxdist

    for i in range(len(a) + 1):
        d[i + 1][0] = maxdist
        d[i + 1][1] = i
    for j in range(len(b) + 1):
        d[0][j + 1] = maxdist
        d[1][j + 1] = j

    for i in range(1, len(a) + 1):
        db = 0
        for j in range(1, len(b) + 1):
            i1 = da.get(b[j - 1], 0)
            j1 = db
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if cost == 0:
                db = j

            d[i + 1][j + 1] = min(
                d[i][j] + cost,
                d[i + 1][j] + 1,
                d[i][j + 1] + 1,
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
        da[a[i - 1]] = i
    return d[len(a) + 1][len(b) + 1]


def jaro_winkler_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    match_distance = max(len(a), len(b)) // 2 - 1
    a_matches = [False] * len(a)
    b_matches = [False] * len(b)

    matches = 0
    for i, ca in enumerate(a):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(b))
        for j in range(start, end):
            if b_matches[j] or ca != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    t = 0
    k = 0
    for i in range(len(a)):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            t += 1
        k += 1
    transpositions = t / 2

    jaro = (
        (matches / len(a))
        + (matches / len(b))
        + ((matches - transpositions) / matches)
    ) / 3

    prefix = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix += 1
        else:
            break
        if prefix == 4:
            break

    return jaro + 0.1 * prefix * (1 - jaro)


def soundex_token(token: str) -> str:
    if not token:
        return ""

    mapping = {
        "b": "1",
        "f": "1",
        "p": "1",
        "v": "1",
        "c": "2",
        "g": "2",
        "j": "2",
        "k": "2",
        "q": "2",
        "s": "2",
        "x": "2",
        "z": "2",
        "d": "3",
        "t": "3",
        "l": "4",
        "m": "5",
        "n": "5",
        "r": "6",
    }

    first = token[0].upper()
    encoded = [mapping.get(ch, "") for ch in token[1:]]

    deduped = []
    prev = ""
    for code in encoded:
        if code != prev:
            deduped.append(code)
        prev = code

    digits = "".join(deduped)
    digits = re.sub(r"[^1-6]", "", digits)
    return (first + digits + "000")[:4]


def phonetic_score(a: str, b: str) -> float:
    at = [soundex_token(t) for t in a.split() if t]
    bt = [soundex_token(t) for t in b.split() if t]
    if not at or not bt:
        return 0.0

    aset = set(at)
    bset = set(bt)
    return len(aset & bset) / len(aset | bset)


def token_sort_ratio(a: str, b: str) -> float:
    sa = " ".join(sorted(a.split()))
    sb = " ".join(sorted(b.split()))
    return SequenceMatcher(None, sa, sb).ratio()


def similarity_from_distance(distance: int, a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return max(0.0, 1 - (distance / denom))


def match_names(name_a: str, name_b: str):
    clean_a = clean_name(name_a)
    clean_b = clean_name(name_b)
    phon_a = phonetic_normalize(name_a)
    phon_b = phonetic_normalize(name_b)

    lev_dist = levenshtein_distance(clean_a, clean_b)
    dam_dist = damerau_levenshtein_distance(clean_a, clean_b)
    lev = similarity_from_distance(lev_dist, clean_a, clean_b)
    dam = similarity_from_distance(dam_dist, clean_a, clean_b)
    seq_matcher = SequenceMatcher(None, clean_a, clean_b)
    seq = seq_matcher.ratio()
    jaro = jaro_winkler_similarity(clean_a, clean_b)
    tsr = token_sort_ratio(clean_a, clean_b)
    pho = phonetic_score(phon_a, phon_b)
    max_len = max(len(clean_a), len(clean_b), 1)
    prefix_len = 0
    for ca, cb in zip(clean_a, clean_b):
        if ca == cb and prefix_len < 4:
            prefix_len += 1
        else:
            break

    sa = " ".join(sorted(clean_a.split()))
    sb = " ".join(sorted(clean_b.split()))
    tokens_a = set(clean_a.split())
    tokens_b = set(clean_b.split())
    soundex_a = {soundex_token(t) for t in phon_a.split() if t}
    soundex_b = {soundex_token(t) for t in phon_b.split() if t}
    soundex_overlap = soundex_a & soundex_b

    rows = [
        {
            "Algorithm": "Levenshtein similarity (character edits needed to convert one name into another)",
            "Weight (%)": 20,
            "Confidence (%)": round(lev * 100, 2),
            "How Matched (This Case)": (
                f"{lev_dist} edits over max length {max_len}; fewer edits means better match."
            ),
        },
        {
            "Algorithm": "Damerau-Levenshtein similarity (character edits + adjacent letter swap handling)",
            "Weight (%)": 20,
            "Confidence (%)": round(dam * 100, 2),
            "How Matched (This Case)": (
                f"{dam_dist} edits including transposition handling (e.g., adjacent letter swaps)."
            ),
        },
        {
            "Algorithm": "Jaro-Winkler similarity (stronger boost for same starting characters)",
            "Weight (%)": 22,
            "Confidence (%)": round(jaro * 100, 2),
            "How Matched (This Case)": (
                f"Common prefix length used for boost: {prefix_len} (max 4)."
            ),
        },
        {
            "Algorithm": "SequenceMatcher ratio (longest common character sequence overlap)",
            "Weight (%)": 20,
            "Confidence (%)": round(seq * 100, 2),
            "How Matched (This Case)": (
                f"Longest aligned block length: {max((b.size for b in seq_matcher.get_matching_blocks()), default=0)}."
            ),
        },
        {
            "Algorithm": "Token sort ratio (word-level similarity after sorting name tokens)",
            "Weight (%)": 12,
            "Confidence (%)": round(tsr * 100, 2),
            "How Matched (This Case)": (
                f"Sorted tokens compared as '{sa}' vs '{sb}'; overlap words: {len(tokens_a & tokens_b)}."
            ),
        },
        {
            "Algorithm": "Soundex phonetic overlap (similar pronunciation mapping)",
            "Weight (%)": 6,
            "Confidence (%)": round(pho * 100, 2),
            "How Matched (This Case)": (
                f"Soundex overlap {len(soundex_overlap)}/{len(soundex_a | soundex_b) or 1}: {sorted(soundex_overlap)}."
            ),
        },
    ]

    weighted_score = (
        (0.20 * lev)
        + (0.20 * dam)
        + (0.22 * jaro)
        + (0.20 * seq)
        + (0.12 * tsr)
        + (0.06 * pho)
    ) * 100

    return {
        "clean_a": clean_a,
        "clean_b": clean_b,
        "phon_a": phon_a,
        "phon_b": phon_b,
        "rows": rows,
        "overall": round(weighted_score, 2),
    }


def verdict(score: float) -> str:
    if score >= 90:
        return "Very High Match (Auto-approve possible)"
    if score >= 80:
        return "High Match (Usually safe)"
    if score >= 65:
        return "Moderate Match (Manual review suggested)"
    if score >= 50:
        return "Low Match (Likely mismatch)"
    return "Very Low Match (Reject likely)"


def confidence_cell_style(value: float) -> str:
    if value >= 85:
        return "background-color: #d1fae5; color: #065f46; font-weight: 600;"
    if value >= 65:
        return "background-color: #fef3c7; color: #92400e; font-weight: 600;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 600;"


st.title("Name Similarity & Confidence Engine (India)")
st.caption(
    "Compare two names with multiple algorithms to handle spelling variation and transliteration issues."
)

left, right = st.columns(2)
with left:
    name_1 = st.text_input("Name 1", placeholder="e.g. Mohammad Faizan Shaikh")
with right:
    name_2 = st.text_input("Name 2", placeholder="e.g. Mohd Faizan Sheikh")

compare = st.button("Compare Names", type="primary")

if compare:
    if not name_1.strip() or not name_2.strip():
        st.warning("Please enter both names.")
    else:
        result = match_names(name_1, name_2)

        score = result["overall"]
        st.subheader("Overall Confidence")
        st.metric("Weighted Match Confidence", f"{score}%")
        st.progress(min(max(score / 100, 0.0), 1.0))
        st.info(f"Decision Hint: {verdict(score)}")

        st.subheader("Normalized View")
        n1, n2 = st.columns(2)
        with n1:
            st.write(f"Input 1 normalized: `{result['clean_a']}`")
            st.write(f"Input 1 phonetic: `{result['phon_a']}`")
        with n2:
            st.write(f"Input 2 normalized: `{result['clean_b']}`")
            st.write(f"Input 2 phonetic: `{result['phon_b']}`")

        st.subheader("Algorithm-Wise Confidence")
        result_df = pd.DataFrame(result["rows"])
        styled_df = result_df.style.map(confidence_cell_style, subset=["Confidence (%)"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption("Color code: green = high confidence, yellow = moderate confidence, red = low confidence.")
        st.caption("Overall score is the weighted sum of the algorithm confidences shown above.")

        st.caption(
            "Tip: for financial KYC, combine this score with DOB/ID matching to reduce false approvals."
        )

st.divider()
st.markdown(
    "**Examples to try:** `Rakesh Kumar` vs `Raakesh Kumaar`, `Mohammad Irfan` vs `Mohd Irfan`, `Shreya Nair` vs `Sreya Nayar`"
)
