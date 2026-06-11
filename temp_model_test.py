import importlib
import traceback

app = importlib.import_module('app')

for exp in app.EXPERIMENTS:
    print('EXP', exp['id'])
    for art in exp['artifacts']:
        print(' ', art.label, art.path, art.fallback_path, art.legacy_path, 'exists=', art.exists, 'resolved=', art.resolved)
    print()

for name, func in [
    ('ann', app.load_ann_model),
    ('cnn', app.load_cat_dog_model),
    ('emotion', app.load_emotion_model),
    ('sentiment', app.load_sentiment_assets),
    ('lstm', app.load_lstm_assets),
    ('gru', app.load_gru_assets),
]:
    print('loading', name)
    try:
        result = func()
        print(' result type:', type(result))
        print(' result repr:', repr(result)[:400])
    except Exception as e:
        print(' exception', type(e).__name__, e)
        traceback.print_exc()
