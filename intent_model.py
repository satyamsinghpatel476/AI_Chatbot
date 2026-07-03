import os
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", CACHE_DIR)
if os.environ.get("ENABLE_CUDA") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import BertTokenizer, BertForSequenceClassification
import torch

LABELS = ["robotics", "daily", "personal", "mixed"]


def _configure_cpu_runtime():
    """Keep the small classifier responsive without saturating a laptop CPU."""
    if os.environ.get("ENABLE_CUDA") == "1":
        return

    thread_count = max(1, int(os.environ.get("TORCH_NUM_THREADS", "2")))
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits setting inter-op threads before parallel work
        # starts. Re-importing this module should not make the app fail.
        pass


_configure_cpu_runtime()


class IntentClassifier:

    def __init__(self):
        use_cuda = os.environ.get("ENABLE_CUDA") == "1" and torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"

        self.tokenizer = BertTokenizer.from_pretrained("models/intent_classifier")
        self.model = BertForSequenceClassification.from_pretrained("models/intent_classifier")

        self.model.to(self.device)
        self.model.eval()
        self._prediction_cache = {}

    def predict(self, text):
        normalized = " ".join(str(text or "").split()).lower()
        if normalized in self._prediction_cache:
            return self._prediction_cache[normalized]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=64,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        result = (LABELS[pred.item()], conf.item())
        self._prediction_cache[normalized] = result
        return result


@lru_cache(maxsize=1)
def get_intent_classifier():
    """Return one classifier shared by Systems B and C in this process."""
    return IntentClassifier()
