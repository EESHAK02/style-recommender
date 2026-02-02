# Style Recommender

- A simple CLIP-based fashion recommendation tool that predicts how well clothing items match. 
- The system uses OpenCLIP embeddings along with a lightweight MLP classifier trained on the Polyvore Outfits dataset.
- Streamlit app provides an easy interface to check compatibility, build outfits, and optionally get styling advice using a local LLM (Ollama).

# Interesting Features

### Pair Compatibility Checker

Upload two clothing item images and get:
- CLIP similarity
- Compatibility probability
- High/Low compatibility verdict
- Optional stylist explanation (LLM)

### Outfit Builder
- Upload multiple items (top, bottom, shoes, accessories)
- Get pairwise compatibility and an overall outfit score

### Finding matching items
- Upload one item and get recommended matches from the Polyvore dataset

### LLM Stylist
- Uses a local LLM (llama3.1) via Ollama
- Provides human-like outfit advice and explanations
- Works only when this is run locally - the streamlit deployed version doesn't give LLM responses

# Notes

- Feel free to check out the details in the report - [Project Report](VLR_Project_Report.pdf)
- Deployed on Streamlit cloud - [Style Recommender](https://semantic-outfit-compatibility-vlr.streamlit.app/)
- The model works best on clean product-style images (Polyvore-like).
- Real-world photos may score lower due to domain shift.
- This is an early version; more features and refinements will be added later.
