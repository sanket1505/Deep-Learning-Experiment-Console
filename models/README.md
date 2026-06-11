# Model Layout

This project keeps trained artifacts under `models` so the Flask app can load local files without downloading models at startup.

- `models/vision`: image and vision models such as ANN digit recognition, CNN cat/dog classification, emotion detection, and image captioning.
- `models/language`: text and sequence models such as sentiment, translation, LSTM next-word prediction, and GRU-style generation.
- `models`: the app also supports the current flat artifact layout directly, so files like `models/ann_model.h5` and `models/tokenizer.pkl` are picked up without moving them.

`app.py` checks organized paths first, then the flat `models` folder, then old project-root paths as a fallback.
