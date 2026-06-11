from __future__ import annotations

import base64
import json
import pickle
import sys
import tempfile
import types
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
VISION_DIR = MODEL_ROOT / "vision"
LANGUAGE_DIR = MODEL_ROOT / "language"


@dataclass(frozen=True)
class Artifact:
    label: str
    path: Path
    legacy_path: Path | None = None
    fallback_path: Path | None = None
    required: bool = True

    @property
    def resolved(self) -> Path:
        for candidate in (self.path, self.fallback_path, self.legacy_path):
            if candidate and candidate.exists():
                return candidate
        return self.path

    @property
    def exists(self) -> bool:
        return self.resolved.exists()


EXPERIMENTS = [
    {
        "id": "ann",
        "number": "02",
        "name": "ANN Digit Recognition",
        "short": "ANN",
        "theme": "digit bench",
        "endpoint": "/predict_ann",
        "accent": "#2563eb",
        "artifacts": [Artifact("Keras model", VISION_DIR / "ann_model.h5", BASE_DIR / "ann_model.h5", MODEL_ROOT / "ann_model.h5")],
    },
    {
        "id": "cnn",
        "number": "03",
        "name": "CNN Cat/Dog Classifier",
        "short": "CNN",
        "theme": "vision bay",
        "endpoint": "/predict_cnn",
        "accent": "#0891b2",
        "artifacts": [
            Artifact("Keras model", VISION_DIR / "cat_dog_model.h5", BASE_DIR / "cat_dog_model.h5", MODEL_ROOT / "cat_dog_model.h5"),
            Artifact("Pickle archive", VISION_DIR / "CNNModel.pkl", BASE_DIR / "CNNModel.pkl", MODEL_ROOT / "CNNModel.pkl"),
        ],
    },
    {
        "id": "emotion",
        "number": "04",
        "name": "Emotion Detection",
        "short": "Emotion",
        "theme": "affect studio",
        "endpoint": "/predict_emotion",
        "accent": "#dc2626",
        "artifacts": [
            Artifact("Keras model", VISION_DIR / "emotion_model.h5", BASE_DIR / "emotion_model.h5", MODEL_ROOT / "emotion_model.h5")
        ],
    },
    {
        "id": "sentiment",
        "number": "05",
        "name": "Sentiment Analysis",
        "short": "Sentiment",
        "theme": "text signal",
        "endpoint": "/predict_sentiment",
        "accent": "#9333ea",
        "artifacts": [
            Artifact("IMDB RNN model", LANGUAGE_DIR / "imdb_rnn_model.h5", BASE_DIR / "imdb_rnn_model.h5", MODEL_ROOT / "imdb_rnn_model.h5"),
            Artifact("Tokenizer", LANGUAGE_DIR / "tokenizer.pkl", BASE_DIR / "tokenizer.pkl", MODEL_ROOT / "tokenizer.pkl"),
        ],
    },
    {
        "id": "translate",
        "number": "06",
        "name": "English to French Translation",
        "short": "Translate",
        "theme": "language bridge",
        "endpoint": "/translate",
        "accent": "#16a34a",
        "artifacts": [],
        "runtime_artifacts": [{"label": "Rule translator", "path": "offline English-French engine", "exists": True}],
    },
    {
        "id": "caption",
        "number": "07",
        "name": "Image Captioning",
        "short": "Caption",
        "theme": "caption deck",
        "endpoint": "/caption",
        "accent": "#ea580c",
        "artifacts": [],
        "runtime_artifacts": [{"label": "Vision captioner", "path": "local image analysis engine", "exists": True}],
    },
    {
        "id": "lstm",
        "number": "08",
        "name": "LSTM Next Word",
        "short": "LSTM",
        "theme": "sequence lab",
        "endpoint": "/predict_next_word",
        "accent": "#4f46e5",
        "artifacts": [
            Artifact("LSTM model", LANGUAGE_DIR / "LSTM_model.h5", BASE_DIR / "LSTM_model.h5", MODEL_ROOT / "LSTM_model.h5"),
            Artifact("Tokenizer", LANGUAGE_DIR / "tokenizer.pkl", BASE_DIR / "tokenizer.pkl", MODEL_ROOT / "tokenizer.pkl"),
        ],
    },
    {
        "id": "gru",
        "number": "09",
        "name": "GRU Text Classifier",
        "short": "GRU",
        "theme": "sequence classifier",
        "endpoint": "/generate_text",
        "accent": "#0f766e",
        "artifacts": [
            Artifact("Sequence model", LANGUAGE_DIR / "model.h5", BASE_DIR / "model.h5", MODEL_ROOT / "model.h5"),
            Artifact("Tokenizer", LANGUAGE_DIR / "tokenizer.pkl", BASE_DIR / "tokenizer.pkl", MODEL_ROOT / "tokenizer.pkl"),
        ],
    },
    {
        "id": "pos",
        "number": "10",
        "name": "POS Tagging",
        "short": "POS",
        "theme": "grammar grid",
        "endpoint": "/pos_tag",
        "accent": "#ca8a04",
        "artifacts": [],
        "runtime_artifacts": [{"label": "POS runtime", "path": "spaCy or rule-based tagger", "exists": True}],
    },
]

LOAD_STATE: dict[str, dict[str, str]] = {}
MODEL_CACHE: dict[str, Any] = {}


def experiment_by_id(experiment_id: str) -> dict[str, Any] | None:
    for experiment in EXPERIMENTS:
        if experiment["id"] == experiment_id:
            return experiment
    return None


def has_local_fallback(experiment_id: str) -> bool:
    experiment = experiment_by_id(experiment_id)
    return bool(experiment.get("fallback", True)) if experiment else True


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def artifact_manifest(artifact: Artifact) -> dict[str, Any]:
    resolved = artifact.resolved
    return {
        "label": artifact.label,
        "path": rel_path(resolved),
        "exists": artifact.exists,
    }


def experiment_manifest() -> list[dict[str, Any]]:
    manifest = []
    for experiment in EXPERIMENTS:
        state = LOAD_STATE.get(experiment["id"], {"state": "ready", "detail": "Local artifacts available"})
        required_artifacts = [artifact for artifact in experiment["artifacts"] if artifact.required]
        optional_artifacts = [artifact for artifact in experiment["artifacts"] if not artifact.required]
        runtime_artifacts = experiment.get("runtime_artifacts", [])
        fallback_available = has_local_fallback(experiment["id"])
        if required_artifacts and not all(a.exists for a in required_artifacts):
            state = (
                {"state": "fallback", "detail": "Model artifact missing; local fallback is available"}
                if fallback_available
                else {"state": "missing", "detail": "One or more artifacts are missing"}
            )
        elif optional_artifacts and not all(a.exists for a in optional_artifacts) and not runtime_artifacts:
            state = {"state": "fallback", "detail": "Optional model artifact missing; local fallback is available"}
        elif runtime_artifacts and not experiment["artifacts"]:
            state = {"state": "ready", "detail": "Local runtime engine available"}

        manifest.append(
            {
                "id": experiment["id"],
                "number": experiment["number"],
                "name": experiment["name"],
                "short": experiment["short"],
                "theme": experiment["theme"],
                "endpoint": experiment["endpoint"],
                "accent": experiment["accent"],
                "status": state["state"],
                "detail": state["detail"],
                "artifacts": [artifact_manifest(a) for a in experiment["artifacts"]] + runtime_artifacts,
            }
        )
    return manifest


def mark_state(experiment_id: str, state: str, detail: str) -> None:
    if state == "loaded":
        state = "ready"
    LOAD_STATE[experiment_id] = {"state": state, "detail": detail}


def load_keras_model(cache_key: str, model_path: Path, experiment_id: str):
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    if not model_path.exists():
        mark_state(experiment_id, "fallback", f"{rel_path(model_path)} not found")
        return None

    try:
        from tensorflow.keras.models import load_model

        model = load_model(model_path, compile=False)
        MODEL_CACHE[cache_key] = model
        mark_state(experiment_id, "loaded", f"Loaded {rel_path(model_path)}")
        return model
    except Exception as exc:
        try:
            model = load_keras_model_compat(model_path)
            MODEL_CACHE[cache_key] = model
            mark_state(experiment_id, "loaded", f"Loaded {rel_path(model_path)} with compatibility loader")
            return model
        except Exception as compat_exc:
            mark_state(
                experiment_id,
                "fallback",
                f"Could not load {rel_path(model_path)}: {exc}; compatibility loader: {compat_exc}",
            )
            return None


def extract_keras_tensor_refs(value: Any) -> list[list[Any]]:
    refs: list[list[Any]] = []
    if isinstance(value, dict):
        if value.get("class_name") == "__keras_tensor__":
            history = value.get("config", {}).get("keras_history")
            if isinstance(history, list) and len(history) == 3:
                refs.append([history[0], history[1], history[2], {}])
        else:
            for child in value.values():
                refs.extend(extract_keras_tensor_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(extract_keras_tensor_refs(child))
    return refs


def normalize_inbound_nodes(inbound_nodes: Any) -> Any:
    if not isinstance(inbound_nodes, list) or not inbound_nodes:
        return inbound_nodes
    if not all(isinstance(node, dict) and "args" in node for node in inbound_nodes):
        return inbound_nodes

    converted = []
    for node in inbound_nodes:
        refs = extract_keras_tensor_refs(node.get("args", []))
        converted.append(refs)
    return converted


def normalize_keras_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        normalized = {key: normalize_keras_config(value) for key, value in obj.items()}
        if normalized.get("class_name") == "DTypePolicy":
            config = normalized.get("config", {})
            return config.get("name", "float32") if isinstance(config, dict) else "float32"

        class_name = normalized.get("class_name")
        config = normalized.get("config")
        if "inbound_nodes" in normalized:
            normalized["inbound_nodes"] = normalize_inbound_nodes(normalized["inbound_nodes"])
        if isinstance(config, dict):
            config.pop("quantization_config", None)
            config.pop("synchronized", None)
        if class_name == "InputLayer" and isinstance(config, dict):
            if "batch_shape" in config and "batch_input_shape" not in config:
                config["batch_input_shape"] = config.pop("batch_shape")
            config.pop("optional", None)
        return normalized
    if isinstance(obj, list):
        return [normalize_keras_config(value) for value in obj]
    return obj


def load_keras_model_compat(model_path: Path):
    import h5py
    from tensorflow.keras.models import model_from_json

    with h5py.File(model_path, "r") as handle:
        model_config = handle.attrs.get("model_config")
        if model_config is None:
            raise ValueError("No model_config found in H5 file")
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

    config = normalize_keras_config(json.loads(model_config))
    model = model_from_json(json.dumps(config))
    model.load_weights(str(model_path))
    return model


def install_keras_pickle_aliases() -> None:
    try:
        import keras.preprocessing.text as keras_text
    except Exception:
        return

    legacy = types.ModuleType("keras.src.legacy")
    preprocessing = types.ModuleType("keras.src.legacy.preprocessing")
    src = types.ModuleType("keras.src")

    sys.modules.setdefault("keras.src", src)
    sys.modules.setdefault("keras.src.legacy", legacy)
    sys.modules.setdefault("keras.src.legacy.preprocessing", preprocessing)
    sys.modules.setdefault("keras.src.legacy.preprocessing.text", keras_text)


def load_pickle(cache_key: str, pickle_path: Path):
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    if not pickle_path.exists():
        return None

    try:
        install_keras_pickle_aliases()
        with pickle_path.open("rb") as handle:
            obj = pickle.load(handle)
        MODEL_CACHE[cache_key] = obj
        return obj
    except Exception:
        return None


def get_artifact(experiment_id: str, label: str) -> Path:
    for experiment in EXPERIMENTS:
        if experiment["id"] == experiment_id:
            for artifact in experiment["artifacts"]:
                if artifact.label == label:
                    return artifact.resolved
    return BASE_DIR / label


def model_input_shape(model: Any) -> tuple[Any, ...] | None:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if shape is None:
        return None
    return tuple(shape)


def prepare_image_tensor(
    image: Image.Image,
    model: Any,
    default_size: tuple[int, int],
    default_channels: int,
    normalization: str = "scale01",
) -> np.ndarray:
    shape = model_input_shape(model)
    height, width = default_size
    channels = default_channels

    if shape:
        dims = list(shape)
        if len(dims) == 4:
            if isinstance(dims[1], int) and dims[1] in (1, 3):
                channels = dims[1]
                height = int(dims[2] or height)
                width = int(dims[3] or width)
            else:
                height = int(dims[1] or height)
                width = int(dims[2] or width)
                channels = int(dims[3] or channels)
        elif len(dims) == 3:
            height = int(dims[1] or height)
            width = int(dims[2] or width)
        elif len(dims) == 2 and isinstance(dims[1], int):
            features = dims[1]
            side = int(np.sqrt(features))
            if side * side == features:
                height = width = side
                channels = 1

    mode = "L" if channels == 1 else "RGB"
    resized = image.convert(mode).resize((width, height))
    arr = np.asarray(resized).astype("float32")

    if normalization == "mobilenet":
        arr = (arr / 127.5) - 1.0
    else:
        arr = arr / 255.0

    if shape and len(shape) == 2:
        return arr.reshape(1, -1)
    if channels == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)


def prepare_digit_tensor(image: Image.Image, model: Any) -> np.ndarray:
    shape = model_input_shape(model) or (None, 28, 28)
    target_height = 28
    target_width = 28
    if len(shape) == 2 and isinstance(shape[1], int):
        side = int(np.sqrt(shape[1]))
        if side * side == shape[1]:
            target_height = target_width = side
    elif len(shape) >= 3:
        target_height = int(shape[1] or target_height)
        target_width = int(shape[2] or target_width)

    gray = image.convert("L")
    arr = np.asarray(gray).astype("uint8")
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(border.mean()) > 127:
        arr = 255 - arr

    foreground = arr > 24
    if foreground.any():
        y_coords, x_coords = np.where(foreground)
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        cropped = arr[y_min : y_max + 1, x_min : x_max + 1]
    else:
        cropped = arr

    side = max(cropped.shape)
    square = np.zeros((side, side), dtype="uint8")
    y_offset = (side - cropped.shape[0]) // 2
    x_offset = (side - cropped.shape[1]) // 2
    square[y_offset : y_offset + cropped.shape[0], x_offset : x_offset + cropped.shape[1]] = cropped

    inner_size = max(1, min(target_height, target_width) - 8)
    digit = Image.fromarray(square).resize((inner_size, inner_size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (target_width, target_height), 0)
    canvas.paste(digit, ((target_width - inner_size) // 2, (target_height - inner_size) // 2))
    normalized = np.asarray(canvas).astype("float32") / 255.0

    if len(shape) == 2:
        return normalized.reshape(1, -1)
    if len(shape) == 4:
        return normalized.reshape(1, target_height, target_width, 1)
    return normalized.reshape(1, target_height, target_width)


def top_labels_from_prediction(prediction: np.ndarray, labels: list[str]) -> tuple[str, float]:
    values = np.asarray(prediction).reshape(-1)
    if values.size == 0:
        return labels[0], 0.0
    if values.size == 1:
        confidence = float(values[0])
        label = labels[1] if confidence >= 0.5 and len(labels) > 1 else labels[0]
        return label, confidence if label == labels[-1] else 1.0 - confidence
    index = int(np.argmax(values))
    label = labels[index] if index < len(labels) else labels[-1]
    return label, float(values[index])


def fallback_digit_prediction(image: Image.Image) -> tuple[int, float]:
    gray = image.convert("L").resize((28, 28))
    arr = np.asarray(gray).astype("float32")
    if float(np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]).mean()) > 127:
        arr = 255 - arr
    ink = arr > 32
    if not ink.any():
        return 0, 0.0

    y_coords, x_coords = np.where(ink)
    height = max(1, int(y_coords.max() - y_coords.min() + 1))
    width = max(1, int(x_coords.max() - x_coords.min() + 1))
    density = float(ink.mean())
    aspect = height / width

    if density > 0.34:
        digit = 8
    elif aspect > 2.2:
        digit = 1
    elif aspect < 0.75:
        digit = 7
    elif density < 0.12:
        digit = 1
    else:
        digit = 0
    return digit, max(0.35, min(0.7, 0.35 + density))


def fallback_cat_dog_prediction(image: Image.Image) -> tuple[str, float]:
    rgb = image.convert("RGB").resize((96, 96))
    arr = np.asarray(rgb).astype("float32")
    brightness = float(arr.mean())
    contrast = float(arr.std())
    warm_minus_cool = float(arr[:, :, 0].mean() - arr[:, :, 2].mean())

    if contrast < 18:
        return "Uncertain image", 0.0
    if warm_minus_cool > 8 or brightness > 145:
        return "Dog-like image", 0.55
    return "Cat-like image", 0.55


def load_ann_model():
    return load_keras_model("ann_model", get_artifact("ann", "Keras model"), "ann")


def load_cat_dog_model():
    return load_keras_model("cat_dog_model", get_artifact("cnn", "Keras model"), "cnn")


def load_emotion_model():
    return load_keras_model("emotion_model", get_artifact("emotion", "Keras model"), "emotion")


def load_sentiment_assets():
    model = load_keras_model("sentiment_model", get_artifact("sentiment", "IMDB RNN model"), "sentiment")
    tokenizer = load_pickle("sentiment_tokenizer", get_artifact("sentiment", "Tokenizer"))
    if model is None or tokenizer is None:
        mark_state("sentiment", "fallback", "Using lexical sentiment rules")
    return model, tokenizer


def load_lstm_assets():
    model = load_keras_model("lstm_model", get_artifact("lstm", "LSTM model"), "lstm")
    tokenizer = load_pickle("lstm_tokenizer", get_artifact("lstm", "Tokenizer"))
    if model is None or tokenizer is None:
        mark_state("lstm", "fallback", "Using curated next-word suggestions")
    return model, tokenizer


def load_gru_assets():
    model = load_keras_model("gru_model", get_artifact("gru", "Sequence model"), "gru")
    tokenizer = load_pickle("gru_tokenizer", get_artifact("gru", "Tokenizer"))
    if model is None or tokenizer is None:
        mark_state("gru", "fallback", "Using lexical GRU-classifier fallback")
    return model, tokenizer


def tokenizer_sequences(tokenizer: Any, text: str):
    if hasattr(tokenizer, "texts_to_sequences"):
        return tokenizer.texts_to_sequences([text])
    return None


def sequence_length_for_model(model: Any, default: int = 100) -> int:
    shape = model_input_shape(model)
    if shape and len(shape) > 1 and isinstance(shape[1], int):
        return int(shape[1])
    return default


def pad_token_sequence(sequence: list[int], max_length: int) -> np.ndarray:
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        return pad_sequences([sequence], maxlen=max_length, padding="pre", truncating="pre")
    except Exception:
        arr = np.zeros((1, max_length), dtype="int32")
        trimmed = sequence[-max_length:]
        arr[0, -len(trimmed) :] = trimmed
        return arr


def predict_next_words_with_model(model: Any, tokenizer: Any, text: str, count: int = 5) -> list[str]:
    sequences = tokenizer_sequences(tokenizer, text.lower())
    if not sequences or not sequences[0]:
        return []

    max_length = sequence_length_for_model(model, default=10)
    padded = pad_token_sequence(sequences[0], max_length)
    prediction = np.asarray(model.predict(padded, verbose=0)).reshape(-1)
    if prediction.size == 0:
        return []

    index_to_word = {idx: word for word, idx in getattr(tokenizer, "word_index", {}).items()}
    top_indices = np.argsort(prediction)[-count:][::-1]
    words = [index_to_word[idx] for idx in top_indices if idx in index_to_word]
    return words[:count]


def fallback_next_words(text: str) -> list[str]:
    lower = text.lower().strip()
    if not lower:
        return ["learning", "model", "network", "data", "signal"]

    transitions = {
        "deep": ["learning", "neural", "models", "networks", "vision"],
        "machine": ["learning", "intelligence", "vision", "translation", "systems"],
        "artificial": ["intelligence", "neural", "systems", "models", "agents"],
        "natural": ["language", "text", "signals", "translation", "syntax"],
        "image": ["caption", "classification", "features", "model", "analysis"],
    }
    last_word = lower.split()[-1]
    return transitions.get(last_word, ["learning", "model", "network", "data", "prediction"])


def lexical_sentiment(text: str) -> tuple[str, float]:
    positive = {
        "good",
        "great",
        "excellent",
        "happy",
        "love",
        "amazing",
        "best",
        "positive",
        "wonderful",
        "enjoy",
    }
    negative = {
        "bad",
        "terrible",
        "sad",
        "hate",
        "awful",
        "worst",
        "negative",
        "angry",
        "poor",
        "pain",
    }
    words = [word.strip(".,!?;:").lower() for word in text.split()]
    score = sum(word in positive for word in words) - sum(word in negative for word in words)
    if score > 0:
        return "POSITIVE", min(0.99, 0.55 + score * 0.12)
    if score < 0:
        return "NEGATIVE", min(0.99, 0.55 + abs(score) * 0.12)
    return "NEUTRAL", 0.5


def load_translation_assets():
    if "translation_assets" in MODEL_CACHE:
        return MODEL_CACHE["translation_assets"]

    model_dir = get_artifact("translate", "Marian transformer")
    if not model_dir.exists():
        mark_state("translate", "ready", "Using offline rule-based translator")
        return None

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), local_files_only=True)
        MODEL_CACHE["translation_assets"] = (tokenizer, model)
        mark_state("translate", "loaded", f"Loaded {rel_path(model_dir)}")
        return tokenizer, model
    except Exception as exc:
        mark_state("translate", "fallback", f"Could not load {rel_path(model_dir)}: {exc}")
        return None


def fallback_translate(text: str) -> str:
    clean = " ".join(text.strip().split())
    lower = clean.lower().strip(" .!?")
    phrasebook = {
        "hello": "bonjour",
        "good morning": "bonjour",
        "good evening": "bonsoir",
        "good night": "bonne nuit",
        "thank you": "merci",
        "how are you": "comment allez-vous",
        "i am fine": "je vais bien",
        "what is your name": "comment vous appelez-vous",
        "my name is sanket": "Je m'appelle Sanket",
        "india is my country": "L'Inde est mon pays",
        "i love india": "J'aime l'Inde",
        "this is a useful project": "C'est un projet utile",
        "deep learning": "apprentissage profond",
        "machine learning": "apprentissage automatique",
        "artificial intelligence": "intelligence artificielle",
        "deep learning helps computers understand language and images": (
            "L'apprentissage profond aide les ordinateurs a comprendre la langue et les images"
        ),
    }
    if lower in phrasebook:
        return phrasebook[lower]

    countries = {
        "india": "l'Inde",
        "france": "la France",
        "america": "l'Amerique",
        "usa": "les Etats-Unis",
        "canada": "le Canada",
        "japan": "le Japon",
    }
    if lower.endswith(" is my country"):
        subject = lower.removesuffix(" is my country").strip()
        return f"{countries.get(subject, subject.title())} est mon pays"
    if lower.startswith("my name is "):
        name = clean[11:].strip()
        return f"Je m'appelle {name}"
    if lower.startswith("i am "):
        rest = lower[5:].strip()
        adjectives = {
            "happy": "heureux",
            "sad": "triste",
            "student": "etudiant",
            "a student": "un etudiant",
            "fine": "bien",
            "good": "bon",
        }
        return f"Je suis {adjectives.get(rest, translate_word_sequence(rest))}"
    if lower.startswith("i love "):
        rest = lower[7:].strip()
        return f"J'aime {translate_word_sequence(rest)}"
    if lower.startswith("this is "):
        rest = lower[8:].strip()
        return f"C'est {translate_word_sequence(rest)}"

    return translate_word_sequence(lower)


def translate_word_sequence(text: str) -> str:
    phrase_map = {
        "deep learning": "apprentissage profond",
        "machine learning": "apprentissage automatique",
        "artificial intelligence": "intelligence artificielle",
        "neural network": "reseau neuronal",
        "neural networks": "reseaux neuronaux",
        "my country": "mon pays",
        "your name": "votre nom",
    }
    output = text
    for english, french in sorted(phrase_map.items(), key=lambda item: len(item[0]), reverse=True):
        output = output.replace(english, french)

    dictionary = {
        "a": "un",
        "an": "un",
        "and": "et",
        "are": "sont",
        "beautiful": "beau",
        "book": "livre",
        "computer": "ordinateur",
        "computers": "ordinateurs",
        "country": "pays",
        "data": "donnees",
        "dog": "chien",
        "cat": "chat",
        "emotion": "emotion",
        "english": "anglais",
        "french": "francais",
        "future": "avenir",
        "good": "bon",
        "happy": "heureux",
        "helps": "aide",
        "image": "image",
        "images": "images",
        "india": "Inde",
        "is": "est",
        "language": "langue",
        "learning": "apprentissage",
        "model": "modele",
        "my": "mon",
        "name": "nom",
        "network": "reseau",
        "prediction": "prediction",
        "project": "projet",
        "student": "etudiant",
        "text": "texte",
        "the": "le",
        "this": "ce",
        "to": "a",
        "translation": "traduction",
        "understand": "comprendre",
        "useful": "utile",
        "vision": "vision",
    }
    translated = [dictionary.get(word.strip(".,!?;:"), word) for word in output.split()]
    result = " ".join(translated).strip()
    return result[:1].upper() + result[1:] if result else ""


def load_caption_assets():
    if "caption_assets" in MODEL_CACHE:
        return MODEL_CACHE["caption_assets"]

    model_dir = get_artifact("caption", "BLIP transformer")
    if not model_dir.exists():
        mark_state("caption", "ready", "Using local vision captioner")
        return None

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(str(model_dir), local_files_only=True)
        MODEL_CACHE["caption_assets"] = (processor, model)
        mark_state("caption", "loaded", f"Loaded {rel_path(model_dir)}")
        return processor, model
    except Exception as exc:
        mark_state("caption", "fallback", f"Could not load {rel_path(model_dir)}: {exc}")
        return None


def fallback_caption(image: Image.Image) -> str:
    smile_result = detect_smile_emotion(image)
    if smile_result is not None:
        return "A smiling person is visible in the image."

    _, face_box = detect_face_region(image)
    if face_box is not None:
        return "A person is visible in the image."

    try:
        model = load_cat_dog_model()
        if model is not None:
            tensor = prepare_image_tensor(
                image,
                model,
                default_size=(160, 160),
                default_channels=3,
                normalization="mobilenet",
            )
            prediction = model.predict(tensor, verbose=0)
            label, confidence = top_labels_from_prediction(prediction, ["cat", "dog"])
            if confidence >= 0.70:
                article = "A" if label == "cat" else "A"
                return f"{article} {label} is visible in the image."
    except Exception:
        pass

    rgb = image.convert("RGB").resize((96, 96))
    arr = np.asarray(rgb).astype("float32")
    brightness = float(arr.mean())
    dominant = int(np.argmax(arr.mean(axis=(0, 1))))
    color_name = ["red-toned", "green-toned", "blue-toned"][dominant]

    if brightness < 80:
        light = "low-light"
    elif brightness > 180:
        light = "bright"
    else:
        light = "balanced-light"
    return f"A {light}, {color_name} image with visible visual patterns."


def detect_face_region(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    try:
        import cv2

        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return image, None

        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        pad_x = int(width * 0.18)
        pad_y = int(height * 0.18)
        left = max(0, int(x - pad_x))
        top = max(0, int(y - pad_y))
        right = min(image.width, int(x + width + pad_x))
        bottom = min(image.height, int(y + height + pad_y))
        return image.crop((left, top, right, bottom)), (int(x), int(y), int(width), int(height))
    except Exception:
        return image, None


def detect_smile_emotion(image: Image.Image) -> tuple[str, float] | None:
    try:
        import cv2

        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        face_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        smile_path = Path(cv2.data.haarcascades) / "haarcascade_smile.xml"
        face_cascade = cv2.CascadeClassifier(str(face_path))
        smile_cascade = cv2.CascadeClassifier(str(smile_path))

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48))
        for x, y, width, height in sorted(faces, key=lambda face: face[2] * face[3], reverse=True):
            roi_gray = gray[y : y + height, x : x + width]
            lower_face = roi_gray[height // 2 :, :]
            smiles = smile_cascade.detectMultiScale(
                lower_face,
                scaleFactor=1.6,
                minNeighbors=12,
                minSize=(max(24, width // 5), max(8, height // 14)),
            )
            if len(smiles) > 0:
                best = max(smiles, key=lambda smile: smile[2] * smile[3])
                smile_ratio = min(1.0, float(best[2]) / max(1.0, float(width) * 0.72))
                return "Happy", max(0.82, min(0.98, 0.82 + smile_ratio * 0.16))
    except Exception:
        return None
    return None


def analyze_deepface_emotion(image: Image.Image) -> tuple[str, float] | None:
    try:
        import os

        os.environ.setdefault("DEEPFACE_HOME", str((BASE_DIR / "runtime_tmp" / "deepface").resolve()))
        from deepface import DeepFace

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            image.convert("RGB").save(temp_file, format="JPEG")
        try:
            result = DeepFace.analyze(str(temp_path), actions=["emotion"], enforce_detection=False)
        finally:
            temp_path.unlink(missing_ok=True)

        record = result[0] if isinstance(result, list) else result
        prediction = str(record.get("dominant_emotion", "neutral")).title()
        scores = record.get("emotion", {})
        confidence = float(scores.get(prediction.lower(), 0.0)) / 100.0 if isinstance(scores, dict) else 0.0
        return prediction, confidence
    except Exception as exc:
        mark_state("emotion", "fallback", f"DeepFace unavailable: {exc}")
        return None


def pos_tag_text(text: str) -> list[dict[str, str]]:
    if "spacy_nlp" not in MODEL_CACHE:
        try:
            import spacy

            MODEL_CACHE["spacy_nlp"] = spacy.load("en_core_web_sm")
            mark_state("pos", "loaded", "Loaded spaCy en_core_web_sm")
        except Exception:
            MODEL_CACHE["spacy_nlp"] = None
            mark_state("pos", "fallback", "Using rule-based POS tagging")

    nlp = MODEL_CACHE.get("spacy_nlp")
    if nlp is not None:
        doc = nlp(text)
        return [
            {
                "text": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "lemma": token.lemma_,
            }
            for token in doc
            if token.text.strip()
        ]

    tags = []
    for raw_word in text.split():
        word = raw_word.strip()
        lower = word.strip(".,!?;:").lower()
        if lower in {"the", "a", "an", "this", "that"}:
            pos, tag = "DET", "DT"
        elif lower in {"and", "or", "but", "because", "so"}:
            pos, tag = "CCONJ", "CC"
        elif lower in {"in", "on", "at", "from", "to", "with", "for", "by"}:
            pos, tag = "ADP", "IN"
        elif lower.endswith("ing") or lower.endswith("ed") or lower in {"is", "are", "was", "were", "be"}:
            pos, tag = "VERB", "VB"
        elif lower.endswith("ly"):
            pos, tag = "ADV", "RB"
        elif lower.endswith("ous") or lower.endswith("ful") or lower.endswith("ive"):
            pos, tag = "ADJ", "JJ"
        else:
            pos, tag = "NOUN", "NN"
        tags.append({"text": word, "pos": pos, "tag": tag, "lemma": lower})
    return tags


def image_from_request() -> Image.Image:
    if request.is_json and request.json and "image" in request.json:
        encoded = request.json["image"].split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(encoded)))

    file = request.files.get("file")
    if not file:
        raise ValueError("No image file was provided")
    return Image.open(file)


@app.route("/")
def home():
    return render_template("index.html", experiments=experiment_manifest(), model_root=rel_path(MODEL_ROOT))


@app.route("/model_manifest")
def model_manifest():
    return jsonify({"model_root": rel_path(MODEL_ROOT), "experiments": experiment_manifest()})


@app.route("/predict_ann", methods=["POST"])
def predict_ann():
    try:
        image = image_from_request()
        model = load_ann_model()
        if model is None:
            digit, confidence = fallback_digit_prediction(image)
            return jsonify(
                {
                    "prediction": digit,
                    "confidence": confidence,
                    "mode": "fallback",
                    "message": "ANN model unavailable; used image-shape fallback",
                }
            )

        try:
            tensor = prepare_digit_tensor(image, model)
            prediction = model.predict(tensor, verbose=0)
            digit = int(np.argmax(prediction, axis=-1).reshape(-1)[0])
            confidence = float(np.max(prediction))
            return jsonify({"prediction": digit, "confidence": confidence, "mode": "model"})
        except Exception as exc:
            mark_state("ann", "fallback", f"ANN prediction error: {exc}")
            digit, confidence = fallback_digit_prediction(image)
            return jsonify({"prediction": digit, "confidence": confidence, "mode": "fallback"})
    except Exception as exc:
        mark_state("ann", "fallback", f"ANN request error: {exc}")
        return jsonify({"error": str(exc), "prediction": 0, "confidence": 0.0, "mode": "fallback"}), 200


@app.route("/predict_cnn", methods=["POST"])
def predict_cnn():
    try:
        image = image_from_request()
        model = load_cat_dog_model()
        if model is None:
            label, confidence = fallback_cat_dog_prediction(image)
            return jsonify({"prediction": label, "confidence": confidence, "mode": "fallback"})

        try:
            tensor = prepare_image_tensor(
                image,
                model,
                default_size=(160, 160),
                default_channels=3,
                normalization="mobilenet",
            )
            prediction = model.predict(tensor, verbose=0)
            label, confidence = top_labels_from_prediction(prediction, ["Cat", "Dog"])
            return jsonify({"prediction": label, "confidence": confidence, "mode": "model"})
        except Exception as exc:
            mark_state("cnn", "fallback", f"CNN prediction error: {exc}")
            label, confidence = fallback_cat_dog_prediction(image)
            return jsonify({"prediction": label, "confidence": confidence, "mode": "fallback"})
    except Exception as exc:
        mark_state("cnn", "fallback", f"CNN request error: {exc}")
        return jsonify({"error": str(exc), "prediction": "Unable to classify", "confidence": 0.0, "mode": "fallback"}), 200


@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    try:
        image = image_from_request()
        labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        smile_result = detect_smile_emotion(image)
        if smile_result is not None:
            label, confidence = smile_result
            mark_state("emotion", "loaded", "Detected smile with local face analysis")
            return jsonify({"prediction": label, "confidence": confidence, "mode": "smile"})

        deepface_result = analyze_deepface_emotion(image)
        if deepface_result is not None:
            label, confidence = deepface_result
            return jsonify({"prediction": label, "confidence": confidence, "mode": "deepface"})

        model = load_emotion_model()
        if model is not None:
            try:
                face_image, face_box = detect_face_region(image)
                tensor = prepare_image_tensor(
                    face_image,
                    model,
                    default_size=(160, 160),
                    default_channels=3,
                    normalization="mobilenet",
                )
                prediction = model.predict(tensor, verbose=0)
                label, confidence = top_labels_from_prediction(prediction, labels)
                mode = "model-face" if face_box else "model"
                return jsonify({"prediction": label, "confidence": confidence, "mode": mode})
            except Exception as exc:
                mark_state("emotion", "fallback", f"Emotion model error: {exc}")

        return jsonify({"prediction": "Neutral", "confidence": 0.5, "mode": "fallback"})
    except Exception as exc:
        mark_state("emotion", "fallback", f"Emotion request error: {exc}")
        return jsonify({"error": str(exc), "prediction": "Neutral", "confidence": 0.5, "mode": "fallback"}), 200


@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    text = (request.json or {}).get("text", "")
    if not text.strip():
        return jsonify({"prediction": "NEUTRAL", "confidence": 0.0, "mode": "empty"})

    model, tokenizer = load_sentiment_assets()
    if model is not None and tokenizer is not None:
        try:
            sequences = tokenizer_sequences(tokenizer, text)
            if sequences and sequences[0]:
                max_length = sequence_length_for_model(model, default=200)
                padded = pad_token_sequence(sequences[0], max_length)
                prediction = model.predict(padded, verbose=0)
                label, confidence = top_labels_from_prediction(prediction, ["NEGATIVE", "POSITIVE"])
                return jsonify({"prediction": label, "confidence": confidence, "mode": "model"})
        except Exception as exc:
            mark_state("sentiment", "fallback", f"Sentiment model error: {exc}")

    lexical_label, lexical_confidence = lexical_sentiment(text)
    return jsonify({"prediction": lexical_label, "confidence": lexical_confidence, "mode": "fallback"})


@app.route("/translate", methods=["POST"])
def translate():
    text = (request.json or {}).get("text", "")
    if not text.strip():
        return jsonify({"translation": "", "mode": "empty"})

    assets = load_translation_assets()
    if assets is not None:
        try:
            tokenizer, model = assets
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs, max_length=128)
            output = tokenizer.decode(translated[0], skip_special_tokens=True)
            return jsonify({"translation": output, "mode": "model"})
        except Exception as exc:
            mark_state("translate", "fallback", f"Translation error: {exc}")

    mark_state("translate", "ready", "Using offline rule-based translator")
    return jsonify({"translation": fallback_translate(text), "mode": "rule-based"})


@app.route("/caption", methods=["POST"])
def caption():
    try:
        image = image_from_request().convert("RGB")
        assets = load_caption_assets()
        if assets is not None:
            try:
                processor, model = assets
                inputs = processor(images=image, return_tensors="pt")
                output_ids = model.generate(**inputs, max_new_tokens=28)
                caption_text = processor.decode(output_ids[0], skip_special_tokens=True).strip()
                return jsonify({"caption": caption_text, "mode": "model"})
            except Exception as exc:
                mark_state("caption", "fallback", f"Caption generation error: {exc}")

        mark_state("caption", "ready", "Using local vision captioner")
        return jsonify({"caption": fallback_caption(image), "mode": "local-vision"})
    except Exception as exc:
        return jsonify({"error": str(exc), "caption": "Unable to generate caption", "mode": "error"}), 200


@app.route("/predict_next_word", methods=["POST"])
def predict_next_word():
    text = (request.json or {}).get("text", "")
    if not text.strip():
        return jsonify({"predictions": fallback_next_words(text), "mode": "fallback"})

    model, tokenizer = load_lstm_assets()
    if model is not None and tokenizer is not None:
        try:
            words = predict_next_words_with_model(model, tokenizer, text, count=5)
            if words:
                return jsonify({"predictions": words, "mode": "model"})
        except Exception as exc:
            mark_state("lstm", "fallback", f"LSTM prediction error: {exc}")

    return jsonify({"predictions": fallback_next_words(text), "mode": "fallback"})


@app.route("/generate_text", methods=["POST"])
def generate_text():
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"generated_text": "Start with a text sample to classify.", "mode": "empty"})

    model, tokenizer = load_gru_assets()
    if model is not None and tokenizer is not None:
        try:
            sequences = tokenizer_sequences(tokenizer, text)
            if sequences and sequences[0]:
                max_length = sequence_length_for_model(model, default=100)
                padded = pad_token_sequence(sequences[0], max_length)
                prediction = np.asarray(model.predict(padded, verbose=0)).reshape(-1)
                confidence = float(prediction[0]) if prediction.size else 0.0
                label = "Spam" if confidence >= 0.5 else "Ham"
                shown_confidence = confidence if confidence >= 0.5 else 1.0 - confidence
                return jsonify(
                    {
                        "generated_text": f"{label} ({shown_confidence:.1%} confidence)",
                        "confidence": shown_confidence,
                        "mode": "model",
                    }
                )
        except Exception as exc:
            mark_state("gru", "fallback", f"GRU classifier error: {exc}")

    label, confidence = lexical_sentiment(text)
    return jsonify({"generated_text": f"{label} ({confidence:.1%} lexical confidence)", "confidence": confidence, "mode": "fallback"})


@app.route("/pos_tag", methods=["POST"])
def pos_tag():
    text = (request.json or {}).get("text", "")
    if not text.strip():
        return jsonify({"pos_tags": [], "mode": "empty"})

    try:
        return jsonify({"text": text, "pos_tags": pos_tag_text(text), "mode": LOAD_STATE.get("pos", {}).get("state", "ready")})
    except Exception as exc:
        return jsonify({"error": str(exc), "pos_tags": [], "mode": "error"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
