import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Firefox Query Normalizer",
    page_icon="🔍",
    layout="wide",
)

HERE = Path(__file__).parent


# ─── Normalizer (loaded once, cached across reruns) ───────────────────────────

@st.cache_resource(show_spinner="Loading normalizer…")
def load_normalizer():
    from benchmark import CombinedV2Normalizer  # noqa: PLC0415
    return CombinedV2Normalizer()


# ─── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(HERE / "results.csv")
    df["should_change"] = df["should_change"].astype(bool)
    df["em"]            = df["em"].astype(bool)
    df["outcome"]       = df.apply(_classify_outcome, axis=1)
    return df


def _classify_outcome(row) -> str:
    if row["should_change"]:
        if row["em"]:
            return "✅ Fixed correctly"
        elif str(row["pred"]).strip().lower() == str(row["noisy"]).strip().lower():
            return "❌ Not fixed"
        else:
            return "⚠️ Fixed incorrectly"
    else:
        return "✅ Left unchanged" if row["em"] else "❌ Over-corrected"


CATEGORY_INFO: dict[str, tuple[str, str]] = {
    "single_typo":   ("✏️ Single Typo",    "One misspelled word (e.g. 'wheather' → 'weather')"),
    "multi_typo":    ("✏️ Multi Typo",     "Two or more typos in the same query"),
    "brand_typo":    ("🏷️ Brand Typo",     "Brand name misspelled (e.g. 'bestbuyt' → 'best buy')"),
    "flight_order":  ("✈️ Flight Order",   "Flight number tokens reordered (e.g. '163 SQ' → 'SQ163')"),
    "product_order": ("📱 Product Order",  "Product tokens reordered (e.g. '15 iphone' → 'iphone 15')"),
    "stock_canon":   ("📈 Stock Ticker",   "Stock query → ticker only (e.g. 'AAPL stock' → 'AAPL')"),
    "spacing":       ("⎵  Spacing",        "Missing spaces fixed (e.g. 'nearme' → 'near me')"),
    "no_change":     ("🔒 No Change",      "Should not be modified — tests over-correction resistance"),
}

OUTCOME_ORDER = [
    "✅ Fixed correctly",
    "✅ Left unchanged",
    "❌ Not fixed",
    "⚠️ Fixed incorrectly",
    "❌ Over-corrected",
]


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🔍 Query Normalizer")
st.caption("**CombinedV2** pipeline · Preprocessing stage for Merino intent classification")

with st.expander("ℹ️ What is this and why does it matter?", expanded=False):
    st.markdown("""
    Intent detection tries to classify user queries by intents —
    navigational, local, commercial, etc. — to surface the right suggestions.
    Real queries are noisy: users make typos, omit spaces, or enter tokens in the
    wrong order.

    **CombinedV2** is a lightweight rule + dictionary normalizer that runs in **< 1 ms**
    per query. It runs 4 steps in sequence and short-circuits as soon as a fix is made:

    | Step | What it handles | Example |
    |------|----------------|---------|
    | **1 · Rules** | Flight IDs, stock tickers, product token reordering | `163 SQ` → `SQ163` |
    | **2 · RapidFuzz** | Fuzzy brand matching (single-token only) | `bestbuyt` → `best buy` |
    | **3 · SymSpell** | Concatenated word splitting | `nearme` → `near me` |
    | **4 · GuardedPySpell** | Spell correction (skips ≤4-char tokens & ALL_CAPS) | `wheather nyc` → `weather nyc` |

    **Benchmark results across 299 queries in 8 categories:**

    | Metric | Score |
    |--------|-------|
    | Exact match on queries that need fixing | **73.2%** |
    | Precision on queries that should NOT change | **98.5%** |
    | Median latency (p50) | **0.03 ms** |
    """)

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_try, tab_browse, tab_perf = st.tabs(["🔤 Try It", "📋 Browse Examples", "📊 Performance"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Try It
# ══════════════════════════════════════════════════════════════════════

with tab_try:
    norm = load_normalizer()
    df   = load_data()

    # ── Free-form input (prominent) ───────────────────────────────────
    st.subheader("Type a query to normalize")
    st.caption("Try typos, missing spaces, scrambled product names, flight numbers, stock tickers…")

    user_query = st.text_input(
        "Query input",
        placeholder="e.g.  wheather nyc  ·  163 SQ  ·  bestbuyt  ·  nearme  ·  15 iphone  ·  AAPL stock",
        label_visibility="collapsed",
        key="user_query",
    )

    if user_query.strip():
        result = norm.normalize(user_query.strip())

        if result.lower() == user_query.strip().lower():
            st.success(f"**`{user_query.strip()}`** → no change needed → **`{result}`**")
        else:
            st.info(f"**`{user_query.strip()}`** → **`{result}`**")

        # Check if it's in the benchmark dataset
        match = df[df["noisy"].str.lower() == user_query.strip().lower()]
        if len(match):
            row = match.iloc[0]
            cat_label = CATEGORY_INFO.get(row["category"], (row["category"], ""))[0]
            if result == row["canonical"]:
                note = f"✅ Matches expected output `{row['canonical']}`"
            else:
                note = f"Expected `{row['canonical']}` · benchmark outcome: **{row['outcome']}**"
            st.caption(f"_Found in benchmark · {cat_label} · {note}_")

    st.divider()

    # ── Example picker ────────────────────────────────────────────────
    st.subheader("Or pick an example from the benchmark")

    pick_col1, pick_col2 = st.columns(2)
    with pick_col1:
        cat_pick = st.selectbox(
            "Category",
            ["All"] + list(CATEGORY_INFO.keys()),
            format_func=lambda k: "All categories" if k == "All" else CATEGORY_INFO[k][0],
            key="cat_pick",
        )
    with pick_col2:
        show_errors_only = st.checkbox("Errors / failures only", value=False)

    sub = df if cat_pick == "All" else df[df["category"] == cat_pick]
    if show_errors_only:
        sub = sub[~sub["em"]]

    if len(sub) == 0:
        st.info("No examples match these filters.")
    else:
        example_labels = [
            f"{row.noisy}   [{CATEGORY_INFO.get(row.category, (row.category,''))[0]}]"
            for row in sub.itertuples()
        ]
        picked_label = st.selectbox("Example", example_labels, key="example_pick")
        picked_noisy = picked_label.split("   [")[0]
        row = sub[sub["noisy"] == picked_noisy].iloc[0]

        ex_left, ex_right = st.columns([3, 1])
        with ex_left:
            t0 = time.perf_counter()
            ex_result  = norm.normalize(picked_noisy)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            st.markdown(f"**Input:** `{picked_noisy}`")
            st.markdown(f"**Expected:** `{row['canonical']}`")

            if ex_result == row["canonical"]:
                st.success(f"**Got:** `{ex_result}` ✅")
            elif ex_result.lower() == picked_noisy.lower():
                st.error(f"**Got:** `{ex_result}` — normalizer didn't fix it")
            else:
                st.warning(f"**Got:** `{ex_result}` — expected `{row['canonical']}`")

        with ex_right:
            st.metric("Latency", f"{elapsed_ms:.2f} ms")
            cat_label = CATEGORY_INFO.get(row["category"], (row["category"], ""))[0]
            st.caption(cat_label)
            st.caption(CATEGORY_INFO.get(row["category"], ("", row["category"]))[1])


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Browse Examples
# ══════════════════════════════════════════════════════════════════════

with tab_browse:
    df = load_data()

    f1, f2 = st.columns(2)
    with f1:
        cats = st.multiselect(
            "Categories",
            options=list(CATEGORY_INFO.keys()),
            default=list(CATEGORY_INFO.keys()),
            format_func=lambda k: CATEGORY_INFO[k][0],
        )
    with f2:
        outcomes = st.multiselect(
            "Outcomes",
            options=OUTCOME_ORDER,
            default=OUTCOME_ORDER,
        )

    filtered = df[df["category"].isin(cats) & df["outcome"].isin(outcomes)]
    st.caption(f"Showing **{len(filtered)}** of {len(df)} examples")

    display = filtered[["noisy", "pred", "canonical", "category", "outcome"]].copy()
    display.columns = ["Input (noisy)", "Predicted", "Expected", "Category", "Outcome"]
    display["Category"] = display["Category"].map(
        lambda k: CATEGORY_INFO.get(k, (k, ""))[0]
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=540,
        column_config={
            "Input (noisy)": st.column_config.TextColumn(width="medium"),
            "Predicted":     st.column_config.TextColumn(width="medium"),
            "Expected":      st.column_config.TextColumn(width="medium"),
            "Category":      st.column_config.TextColumn(width="medium"),
            "Outcome":       st.column_config.TextColumn(width="small"),
        },
    )


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Performance
# ══════════════════════════════════════════════════════════════════════

with tab_perf:
    df = load_data()

    needs_change = df[df["should_change"]]
    no_change    = df[~df["should_change"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total examples",      f"{len(df)}")
    c2.metric("Overall EM",          f"{df['em'].mean():.1%}")
    c3.metric("Fix accuracy",        f"{needs_change['em'].mean():.1%}",
              help="Exact match on queries that SHOULD change")
    c4.metric("No-change precision", f"{no_change['em'].mean():.1%}",
              help="Correctly left unchanged queries that should NOT change")

    st.markdown("---")
    st.subheader("Per-category breakdown")

    rows = []
    for cat, (label, desc) in CATEGORY_INFO.items():
        sub   = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        needs = sub[sub["should_change"]]
        ok    = sub[~sub["should_change"]]
        rows.append({
            "Category":        label,
            "n":               len(sub),
            "EM %":            f"{sub['em'].mean():.0%}",
            "Fix accuracy":    f"{needs['em'].mean():.0%}" if len(needs) else "—",
            "No-change prec.": f"{ok['em'].mean():.0%}"    if len(ok)    else "—",
            "Errors":          int((~sub["em"]).sum()),
            "What it tests":   desc,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Failure cases by category")
    st.caption("All queries where the normalizer produced a wrong output.")

    failures = df[~df["em"]]
    if len(failures) == 0:
        st.success("No failures!")
    else:
        for cat, (label, _) in CATEGORY_INFO.items():
            sub = failures[failures["category"] == cat]
            if len(sub) == 0:
                continue
            with st.expander(f"{label} — {len(sub)} failure{'s' if len(sub) != 1 else ''}"):
                show = sub[["noisy", "pred", "canonical", "outcome"]].copy()
                show.columns = ["Input", "Predicted", "Expected", "Outcome"]
                st.dataframe(show, use_container_width=True, hide_index=True)
