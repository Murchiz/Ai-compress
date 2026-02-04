import os
import json
import py7zr
import datetime
from backend.core.engine import AICompressionEngine

class BrainManager:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.data_dir = os.path.join(base_dir, "data/training_data")
        self.history_file = os.path.join(self.models_dir, "history.json")

        for d in [self.models_dir, self.data_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        self.history = self._load_history()
        self.engine = AICompressionEngine()
        self._ensure_default_brain()

    def _ensure_default_brain(self):
        default_model_path = os.path.join(self.models_dir, "default.pt")
        if not os.path.exists(default_model_path):
            self.engine.predictor.save(default_model_path)

    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"current_brain": "default", "brains": {"default": {"created_at": str(datetime.datetime.now())}}}

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)

    def get_current_brain_id(self):
        return self.history["current_brain"]

    def compress_and_learn(self, input_path, output_path):
        brain_id = self.get_current_brain_id()
        model_path = os.path.join(self.models_dir, f"{brain_id}.pt")
        if os.path.exists(model_path):
            self.engine.predictor.load(model_path)

        # 1. Compress
        self.engine.compress(input_path, output_path, model_id=brain_id)

        # 2. Add to training data (using 7z)
        self._add_to_training_set(input_path)

        # 3. Learn (Automatic)
        with open(input_path, 'rb') as f:
            data = f.read()

        print(f"Learning from {input_path}...")
        self.engine.predictor.train_on_data(data, epochs=5)

        # 4. Save new brain version
        new_brain_id = f"brain_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_model_path = os.path.join(self.models_dir, f"{new_brain_id}.pt")
        self.engine.predictor.save(new_model_path)

        # Update history
        self.history["brains"][new_brain_id] = {
            "created_at": str(datetime.datetime.now()),
            "parent": brain_id,
            "learned_from": os.path.basename(input_path)
        }
        self.history["current_brain"] = new_brain_id
        self._save_history()

        return new_brain_id

    def _add_to_training_set(self, file_path):
        archive_path = os.path.join(self.data_dir, "training_set.7z")
        mode = 'a' if os.path.exists(archive_path) else 'w'
        with py7zr.SevenZipFile(archive_path, mode) as archive:
            archive.write(file_path, os.path.basename(file_path))

    def decompress(self, input_path, output_path):
        self.engine.decompress(input_path, output_path, model_library_path=self.models_dir)
