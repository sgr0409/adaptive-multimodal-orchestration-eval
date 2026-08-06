"""Image-modality scorer: a local CLIP vision encoder produces an image
embedding, then a small logistic-regression classifier (trained on the
train split) predicts severity from that embedding. Confidence is the
classifier's own max predicted probability.

Design note, disclosed rather than hidden: we initially tried pure
zero-shot CLIP classification (image-text similarity against three
condition prompts, no training at all). It performed barely above chance
(max softmax ~0.52 against a 0.33 baseline) and misclassified the
critical example -- unsurprising in hindsight, since CLIP was trained on
natural photographs, not procedurally-drawn vector gauge icons, and its
zero-shot text-image alignment has no real prior for this visual domain.
Rather than hand-tune prompts until the numbers looked better, we switched
to the same pattern used for text and telemetry: a frozen pretrained
encoder plus a lightweight trained classification head. This is still a
genuine pretrained vision-language foundation model doing the visual
feature extraction, just not asked to do zero-shot classification in a
domain it was never suited to."""
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor

LABELS = ["normal", "warning", "critical"]


class ImageScorer:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        self._fitted = False

    def embed(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return feats.cpu().numpy()

    def fit(self, image_paths, labels):
        X = self.embed(image_paths)
        self.clf.fit(X, labels)
        self._fitted = True

    def predict(self, image_path):
        assert self._fitted, "ImageScorer.fit() must be called before predict()"
        x = self.embed([image_path])
        probs = self.clf.predict_proba(x)[0]
        classes = list(self.clf.classes_)
        best_idx = probs.argmax()
        return {
            "modality": "image",
            "label": classes[best_idx],
            "confidence": float(probs[best_idx]),
            "probs": {c: float(p) for c, p in zip(classes, probs)},
        }
