import os
import json
from typing import Optional

from app.services.pytorch_training import save_model as _save_model


def save_with_metadata(model, path: str, metadata: Optional[dict] = None):
	"""Save the model state dict to `path` and write accompanying metadata.json next to it."""
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	_save_model(model, path)
	if metadata is None:
		metadata = {}
	meta_path = os.path.splitext(path)[0] + ".json"
	with open(meta_path, "w", encoding="utf-8") as fh:
		json.dump(metadata, fh, indent=2)


# Example usage:
# from app.model_save import save_with_metadata
# save_with_metadata(my_model, "models/exp1.pth", {"dataset":"salinas", "epoch":10})


def list_models(models_dir: str = "models") -> list:
	"""Return a list of model files with their metadata (if available)."""
	out = []
	if not os.path.isdir(models_dir):
		return out
	for fname in os.listdir(models_dir):
		if fname.endswith(".pth"):
			path = os.path.join(models_dir, fname)
			meta_path = os.path.splitext(path)[0] + ".json"
			meta = None
			if os.path.exists(meta_path):
				try:
					with open(meta_path, "r", encoding="utf-8") as fh:
						meta = json.load(fh)
				except Exception:
					meta = None
			out.append({"model": path, "metadata": meta})
	return out


def load_metadata(model_path: str) -> Optional[dict]:
	meta_path = os.path.splitext(model_path)[0] + ".json"
	if os.path.exists(meta_path):
		with open(meta_path, "r", encoding="utf-8") as fh:
			return json.load(fh)
	return None
