# Deep Learning Experiment Console

A Flask-based web console for running several deep learning experiments from one interface. The app loads local trained model artifacts, exposes prediction endpoints, and includes fallback behavior for features that can still work when optional model dependencies are unavailable.

## Features

- Handwritten digit prediction with an ANN model
- Cat vs dog image classification with a CNN model
- Facial emotion prediction
- IMDB-style text sentiment prediction
- English-to-Hindi style word/phrase translation fallback
- Image captioning fallback
- Next-word prediction and text generation
- Part-of-speech tagging utilities
- Model manifest endpoint for checking artifact availability

## Project Structure

```text
.
|-- app.py                 # Main Flask app and prediction endpoints
|-- run_server.py          # Simple production-style local runner
|-- templates/
|   `-- index.html         # Web console UI
|-- models/                # Local model artifacts tracked with Git LFS
|   |-- *.h5
|   |-- *.pkl
|   `-- README.md
|-- .gitattributes         # Git LFS rules for model files
|-- .gitignore
`-- LICENSE
```

## Requirements

- Python 3.10 recommended
- Git LFS for downloading the model artifacts from GitHub
- Python packages used by the app:
  - `flask`
  - `numpy`
  - `pillow`
  - `tensorflow`
  - `opencv-python`
  - `deepface` optional, used for enhanced emotion analysis
  - `transformers` optional, used when local caption/translation assets are available

## Setup

Clone the repository:

```bash
git clone https://github.com/sanket1505/Deep-Learning-Experiment-Console.git
cd Deep-Learning-Experiment-Console
```

Install Git LFS and pull model artifacts:

```bash
git lfs install
git lfs pull
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install flask numpy pillow tensorflow opencv-python deepface transformers
```

## Run Locally

```bash
python run_server.py
```

Open the app at:

```text
http://127.0.0.1:5000
```

For development, you can also run:

```bash
python app.py
```

## Model Files

Model artifacts are stored in `models/` and tracked with Git LFS because several files are too large for normal Git storage. If a model appears missing after cloning, run:

```bash
git lfs pull
```

The app checks organized model paths first, then the flat `models/` folder, and finally legacy project-root paths as a fallback.

## API Endpoints

- `GET /` - Web console
- `GET /model_manifest` - Model artifact status
- `POST /predict_ann` - Digit prediction
- `POST /predict_cnn` - Cat/dog prediction
- `POST /predict_emotion` - Emotion prediction
- `POST /predict_sentiment` - Text sentiment prediction
- `POST /translate` - Translation
- `POST /caption` - Image captioning
- `POST /predict_next_word` - Next-word prediction
- `POST /generate_text` - Text generation
- `POST /pos_tag` - Part-of-speech tagging

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
