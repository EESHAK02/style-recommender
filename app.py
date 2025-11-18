# app.py
import io
from typing import List, Tuple

from PIL import Image
import streamlit as st

from core import (
    load_trained_models_for_inference,
    predict_pair,
    sample_test_images,
    rank_candidates_for_anchor,
    score_outfit,
    IMAGES_DIR,
    call_ollama,
)

st.set_page_config(page_title="Fashion CLIP Stylist", layout="wide")
st.title("Stylo AI")

# st.write(
#     "This app uses a CLIP image encoder (via `open_clip`) and a small MLP trained on "
#     "Polyvore outfit pairs. The model predicts how compatible items are based on style. "
#     "Training happens offline; this app only runs inference."
# )

# ---- Session state for persisting results ----
if "pair_results" not in st.session_state:
    st.session_state["pair_results"] = None

if "matches_ranked" not in st.session_state:
    st.session_state["matches_ranked"] = None
if "matches_anchor_desc" not in st.session_state:
    st.session_state["matches_anchor_desc"] = ""

if "outfit_results" not in st.session_state:
    st.session_state["outfit_results"] = None  # dict with score, probs, labels, descs


@st.cache_resource
def _load_models_cached():
    return load_trained_models_for_inference()


clip_model, preprocess, classifier = _load_models_cached()


# ---------- small helpers ----------

def _load_dataset_image(rel_path: str) -> Image.Image:
    return Image.open(IMAGES_DIR / rel_path).convert("RGB")


def _llm_stylist_for_pair(
    desc1: str, desc2: str, cos_sim: float, prob: float
) -> str:
    from math import isnan

    if isnan(prob):
        prob_val = "N/A"
    else:
        prob_val = f"{prob*100:.1f}%"

    prompt = f"""
You are a friendly fashion stylist.

Two clothing items are being evaluated by an ML model trained on Polyvore outfits.
The model outputs:
- CLIP cosine similarity: {cos_sim:.3f}
- Compatibility probability (Polyvore-style): {prob_val}

Item 1 description (user-provided, may be empty):
{desc1 or "[no description]"}

Item 2 description (user-provided, may be empty):
{desc2 or "[no description]"}

In 3–4 sentences, explain:
1) Whether these pieces likely go well together and why.
2) Any caveats (for example if the model might be biased toward studio product photos).
3) A simple styling tip (shoes, layers, accessories) to make the outfit work better.

Keep the tone concise, friendly, and non-repetitive.
"""
    resp = call_ollama(prompt)
    if resp is None:
        return "Stylist is unavailable (Ollama not running or model not pulled)."
    return resp


def _llm_stylist_for_outfit(
    descriptions: List[str], outfit_score: float
) -> str:
    desc_block = "\n".join(
        f"- Item {i+1}: {d or '[no description]'}"
        for i, d in enumerate(descriptions)
    )
    prompt = f"""
You are a fashion stylist helping evaluate a full outfit.

We have an ML model trained on Polyvore outfit compatibility. For this outfit, the
overall compatibility score (average pairwise probability) is {outfit_score*100:.1f}%.

Here are rough descriptions of each item:
{desc_block}

In 4–5 sentences:
1) Comment on how cohesive this outfit feels (colors, style, formality).
2) Mention at least one potential mismatch or risk if any.
3) Suggest 1–2 small tweaks that could improve the outfit (change shoes, add layer, etc.).

Be practical and concrete. Avoid generic phrases like "it's up to personal style".
"""
    resp = call_ollama(prompt)
    if resp is None:
        return "Stylist is unavailable (Ollama not running or model not pulled)."
    return resp


# ---------- UI ----------

tab1, tab2 = st.tabs(["🔍 Check Pair", "🧩 Outfit Builder & Recommendations"])


# ===== Tab 1: pair checker =====
with tab1:
    st.subheader("Check compatibility between two items")

    col_inputs = st.columns(2)

    with col_inputs[0]:
        img_file1 = st.file_uploader(
            "Item 1", type=["jpg", "jpeg", "png"], key="pair_img1"
        )
        desc1 = st.text_input("Describe item 1 (optional)", key="pair_desc1")
    with col_inputs[1]:
        img_file2 = st.file_uploader(
            "Item 2", type=["jpg", "jpeg", "png"], key="pair_img2"
        )
        desc2 = st.text_input("Describe item 2 (optional)", key="pair_desc2")

    # show smaller previews if available
    if img_file1 or img_file2:
        col_imgs = st.columns(2)
        if img_file1:
            img1 = Image.open(img_file1).convert("RGB")
            with col_imgs[0]:
                st.image(img1, caption="Item 1", width=260)
        if img_file2:
            img2 = Image.open(img_file2).convert("RGB")
            with col_imgs[1]:
                st.image(img2, caption="Item 2", width=260)

    # button to compute + store results
    if img_file1 and img_file2:
        if st.button("Check compatibility", key="pair_check"):
            img1 = Image.open(img_file1).convert("RGB")
            img2 = Image.open(img_file2).convert("RGB")
            with st.spinner("Scoring pair..."):
                cos_sim, prob = predict_pair(
                    clip_model, preprocess, classifier, img1, img2
                )
            st.session_state["pair_results"] = {
                "cos_sim": cos_sim,
                "prob": prob,
                "desc1": desc1,
                "desc2": desc2,
            }
    else:
        st.info("Upload two images of clothing items to check their compatibility.")

    # show stored results (persists across reruns)
    pair_res = st.session_state.get("pair_results")
    if pair_res is not None and img_file1 and img_file2:
        st.markdown("### Results")
        cos_sim = pair_res["cos_sim"]
        prob = pair_res["prob"]
        res_desc1 = pair_res["desc1"]
        res_desc2 = pair_res["desc2"]

        r1, r2, _ = st.columns(3)
        r1.metric("CLIP cosine similarity", f"{cos_sim:.3f}")
        r2.metric("Polyvore-compatibility", f"{prob*100:.1f}%")

        # verdict
        if prob >= 0.7:
            verdict = "High compatibility — model thinks these likely belong in the same outfit."
            st.success("✅ " + verdict)
        elif prob >= 0.4:
            verdict = "Medium compatibility — could work, depending on styling."
            st.warning("🤔 " + verdict)
        else:
            verdict = "Low compatibility — model sees this pairing as weak."
            st.error("❌ " + verdict)

        # domain shift hint
        if cos_sim >= 0.5 and prob < 0.3:
            st.info(
                "Note: Visual similarity is moderate, but compatibility is low. "
                "This can happen if the photo style differs from the Polyvore "
                "product images the model was trained on."
            )

        # stylist section
        st.markdown("##### Stylist (LLM, optional)")
        if st.checkbox("Ask stylist (requires Ollama running)", key="pair_llm_on"):
            if st.button("Ask stylist for advice", key="pair_llm_btn"):
                with st.spinner("Stylist is thinking..."):
                    response = _llm_stylist_for_pair(
                        res_desc1, res_desc2, cos_sim, prob
                    )
                st.markdown("###### Stylist says:")
                st.write(response)


# ===== Tab 2: outfit builder & recommendations =====
with tab2:
    st.subheader("Build outfits and get recommendations")

    mode = st.radio(
        "Choose mode",
        ["Find matches for this item", "Rate my full outfit"],
        key="builder_mode",
    )

    # ---- Mode 2.1: find matches for an anchor item ----
    if mode == "Find matches for this item":
        st.markdown(
            "#### Upload an anchor item and we'll suggest matches from the Polyvore test set."
        )

        col_anchor = st.columns(2)
        with col_anchor[0]:
            anchor_file = st.file_uploader(
                "Anchor item", type=["jpg", "jpeg", "png"], key="anchor_img"
            )
        with col_anchor[1]:
            anchor_desc = st.text_input(
                "Describe this anchor item (optional)", key="anchor_desc"
            )

        num_candidates = st.slider(
            "How many candidates to sample from the dataset?",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
        )
        top_k = st.slider(
            "How many top matches to display?",
            min_value=4,
            max_value=20,
            value=8,
            step=2,
        )

        anchor_img = None
        if anchor_file:
            anchor_img = Image.open(anchor_file).convert("RGB")
            st.image(anchor_img, caption="Anchor item", width=260)

        if anchor_img is not None:
            if st.button("Find matching items", key="find_matches_btn"):
                with st.spinner("Sampling candidates and scoring matches..."):
                    candidate_paths = sample_test_images(num_candidates)
                    ranked = rank_candidates_for_anchor(
                        clip_model,
                        preprocess,
                        classifier,
                        anchor_img,
                        candidate_paths,
                        top_k=top_k,
                    )

                if not ranked:
                    st.error("No candidate images found in the test set.")
                    st.session_state["matches_ranked"] = None
                else:
                    st.session_state["matches_ranked"] = ranked
                    st.session_state["matches_anchor_desc"] = anchor_desc

        # show ranked matches from session_state (if any)
        ranked = st.session_state.get("matches_ranked")
        if anchor_img is not None and ranked:
            st.markdown("### Top matches from Polyvore test set")
            n_cols = 4
            cols = st.columns(n_cols)
            for i, cand in enumerate(ranked):
                c = cols[i % n_cols]
                with c:
                    img = _load_dataset_image(cand["path"])
                    st.image(img, use_container_width=True)
                    st.caption(
                        f"{cand['path']}\n\n"
                        f"Compat: {cand['prob']*100:.1f}% | "
                        f"CLIP sim: {cand['cos_sim']:.3f}"
                    )

            # stylist summary of top matches
            st.markdown("##### Stylist (LLM, optional)")
            if st.checkbox(
                "Ask stylist about these matches", key="matches_llm_on"
            ):
                if st.button("Ask stylist", key="matches_llm_btn"):
                    ranked = st.session_state.get("matches_ranked") or []
                    if ranked:
                        avg_prob = sum(c["prob"] for c in ranked) / len(ranked)
                        scores_summary = ", ".join(
                            f"{i+1}:{c['prob']*100:.1f}%"
                            for i, c in enumerate(ranked[:5])
                        )
                        anchor_d = st.session_state.get(
                            "matches_anchor_desc", ""
                        )
                        prompt = f"""
You are a fashion stylist.

We have an anchor clothing item (user description: "{anchor_d or '[no description]'}")
and we retrieved several matches from a Polyvore-like catalog.

The model we use is trained on Polyvore outfit pairs. It outputs compatibility probabilities.
For the top matches, probabilities (in %) are roughly: {scores_summary}.
The average compatibility over the shown matches is {avg_prob*100:.1f}%.

In 3–5 sentences:
1) Comment on what kinds of matches these likely are (e.g., mostly casual, mostly neutral colors, etc.).
2) Suggest what kind of item the user should focus on (e.g., darker shoes, more structured jackets).
3) Give one concrete styling tip to build a full outfit around the anchor piece.
"""
                        with st.spinner("Stylist is thinking..."):
                            resp = call_ollama(prompt)
                        if resp is None:
                            st.info(
                                "Stylist is unavailable (Ollama not running or model not pulled)."
                            )
                        else:
                            st.markdown("###### Stylist says:")
                            st.write(resp)

        elif anchor_img is None:
            st.info("Upload an anchor item image to get recommendations.")

    # ---- Mode 2.2: rate my full outfit ----
    else:
        st.markdown(
            "#### Upload 2–4 items and get an overall outfit score + pairwise compatibilities."
        )

        cols = st.columns(4)
        labels = ["Top", "Bottom", "Shoes", "Accessory"]
        files = []
        descs = []

        for i, label in enumerate(labels):
            with cols[i]:
                f = st.file_uploader(
                    label, type=["jpg", "jpeg", "png"], key=f"outfit_{label}"
                )
                d = st.text_input(
                    f"{label} description (optional)", key=f"outfit_desc_{label}"
                )
                files.append(f)
                descs.append(d)

        # collect actual provided items
        provided_imgs: List[Tuple[str, Image.Image]] = []
        provided_descs: List[str] = []
        for label, f, d in zip(labels, files, descs):
            if f is not None:
                img = Image.open(f).convert("RGB")
                provided_imgs.append((label, img))
                provided_descs.append(d)

        if len(provided_imgs) < 2:
            st.info("Upload at least two items to rate an outfit.")
            st.session_state["outfit_results"] = None
        else:
            # show small previews
            st.markdown("##### Outfit preview")
            cols_prev = st.columns(len(provided_imgs))
            for (label, img), col in zip(provided_imgs, cols_prev):
                with col:
                    st.image(img, caption=label, width=220)

            if st.button("Rate this outfit", key="rate_outfit_btn"):
                with st.spinner("Scoring outfit..."):
                    imgs_only = [img for _, img in provided_imgs]
                    probs_matrix, outfit_score = score_outfit(
                        clip_model, preprocess, classifier, imgs_only
                    )

                labels_short = [lbl for lbl, _ in provided_imgs]
                st.session_state["outfit_results"] = {
                    "score": outfit_score,
                    "probs": probs_matrix.tolist(),
                    "labels": labels_short,
                    "descs": provided_descs,
                }

        # display stored outfit results (if any)
        out_res = st.session_state.get("outfit_results")
        if out_res is not None:
            outfit_score = out_res["score"]
            probs_matrix = out_res["probs"]
            labels_short = out_res["labels"]
            stored_descs = out_res["descs"]

            st.markdown("### Outfit results")
            st.metric(
                "Overall outfit compatibility", f"{outfit_score*100:.1f}%"
            )

            # pairwise table
            st.markdown("#### Pairwise compatibility matrix")
            header = "| | " + " | ".join(labels_short) + " |"
            sep = "|---" * (len(labels_short) + 1) + "|"
            rows = []
            n = len(labels_short)
            for i, r_label in enumerate(labels_short):
                cells = [r_label]
                for j in range(n):
                    cells.append(f"{probs_matrix[i][j]*100:.1f}%")
                rows.append("| " + " | ".join(cells) + " |")

            st.markdown("\n".join([header, sep] + rows))

            # stylist
            st.markdown("##### Stylist (LLM, optional)")
            if st.checkbox(
                "Ask stylist about this outfit", key="outfit_llm_on"
            ):
                if st.button("Ask stylist", key="outfit_llm_btn"):
                    with st.spinner("Stylist is thinking..."):
                        resp = _llm_stylist_for_outfit(
                            stored_descs, outfit_score
                        )
                    st.markdown("###### Stylist says:")
                    st.write(resp)
