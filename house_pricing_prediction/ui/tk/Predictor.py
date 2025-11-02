import json
import zipfile

import pickle


def save_model(model, feature_names, zip_path):
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('model.pkl', pickle.dumps(model))
        zf.writestr('feature_names.json', json.dumps(list(feature_names)))


def load_model(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        model = pickle.loads(zf.read('model.pkl'))
        fns = json.loads(zf.read('feature_names.json').decode('utf-8'))
        return model, fns