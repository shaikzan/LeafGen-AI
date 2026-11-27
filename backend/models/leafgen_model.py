# leafgen_model.py - load and use the LeafGen AI models (PyTorch, CPU is enough)
#
#   pip install torch torchvision opencv-python-headless numpy huggingface_hub
#
#   from leafgen_model import load_classifier, classify, load_generator, generate
#   clf = load_classifier()                       # 'real' = recommended (best overall), 'aug' = real+synthetic experiment model
#   print(classify(clf, rgb_uint8_image))         # [(class_name, probability), ...] top-3
#   gen = load_generator()
#   imgs = generate(gen, 'Apple___Cedar_apple_rust', n=8)   # uint8 array (n, 64, 64, 3)
import json
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models

IMG, NZ = 64, 100
MODEL_DIR = Path(__file__).resolve().parent

def _file(name):
    path = MODEL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Model asset '{name}' is missing from {MODEL_DIR}. "
            "Run 'python backend/download_models.py' from the project root."
        )
    return str(path)

with open(_file('classes.json'), encoding='utf-8') as file:
    classes = json.load(file)
with open(_file('rare.json'), encoding='utf-8') as file:
    rare = json.load(file)
NC = len(classes)
if NC != 38 or len(set(classes)) != NC:
    raise ValueError(f"Expected 38 unique classifier classes in {MODEL_DIR / 'classes.json'}; found {NC} entries and {len(set(classes))} unique values.")

def pretty(c):
    return c.replace('___', ' - ').replace('_', ' ')

# ---- preprocessing: MUST be identical to training (resize 64x64 INTER_AREA, then CLAHE on L channel in LAB) ----
def preprocess(rgb):
    rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load_classifier(variant='real'):
    try:
        fname = {'real': 'cnn_real.pt', 'aug': 'cnn_aug.pt'}[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown classifier variant: {variant!r}. Choose 'real' or 'aug'.") from exc
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, NC)
    state = torch.load(_file(fname), map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or state.get('fc.weight', torch.empty(0)).shape != (NC, 512):
        raise ValueError(f"Checkpoint '{fname}' does not contain a ResNet18 classifier head with {NC} outputs.")
    # Strict loading intentionally fails on any missing/unexpected keys. Never
    # fall back to random initialization when a trained checkpoint is invalid.
    m.load_state_dict(state, strict=True)
    return m.eval()

@torch.inference_mode()
def predict_proba(model, rgb_uint8):
    """Return the full 38-class softmax vector for an RGB uint8 image."""
    rgb_uint8 = np.asarray(rgb_uint8)
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3 or rgb_uint8.shape[0] == 0 or rgb_uint8.shape[1] == 0:
        raise ValueError("Expected a non-empty HxWx3 RGB image.")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError("Expected RGB pixels with dtype uint8 in the range 0–255.")
    pre = preprocess(rgb_uint8)
    x = torch.from_numpy(pre.transpose(2, 0, 1)[None].copy()).float() / 255.
    x = F.interpolate(x, size=128, mode='bilinear', align_corners=False)
    logits = model((x - _MEAN) / _STD)
    if logits.ndim != 2 or logits.shape != (1, NC):
        raise ValueError(f"Classifier returned shape {tuple(logits.shape)}; expected (1, {NC}).")
    if not torch.isfinite(logits).all():
        raise ValueError("Classifier returned non-finite logits.")
    p = F.softmax(logits, dim=1)[0]
    if not torch.isfinite(p).all() or not torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5):
        raise ValueError("Classifier returned an invalid probability distribution.")
    return p.numpy()

def classify(model, rgb_uint8, k=3):
    # rgb_uint8: HxWx3 uint8 RGB image (any size)
    if not isinstance(k, int) or not 1 <= k <= NC:
        raise ValueError(f"k must be an integer between 1 and {NC}.")
    if not isinstance(k, int) or not 1 <= k <= NC:
        raise ValueError(f"k must be an integer between 1 and {NC}.")
    p = predict_proba(model, rgb_uint8)
    top = p.argsort()[::-1][:k]
    return [(classes[i], float(p[i])) for i in top]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(NZ + NC, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
    def forward(self, z, y):
        return self.net(torch.cat([z, F.one_hot(y, NC).float()], 1).view(-1, NZ + NC, 1, 1))

def load_generator():
    g = Generator()
    state = torch.load(_file('generator.pt'), map_location='cpu', weights_only=True)
    g.load_state_dict(state, strict=True)
    return g.eval()

@torch.inference_mode()
def generate(gen, class_name, n=8):
    if class_name not in classes:
        raise ValueError(f"Unsupported class code: {class_name!r}.")
    if not isinstance(n, int) or not 1 <= n <= 8:
        raise ValueError("n must be an integer between 1 and 8.")
    label = classes.index(class_name)
    z = torch.randn(n, NZ)
    y = torch.full((n,), label, dtype=torch.long)
    out = ((gen(z, y) + 1) * 127.5).clamp(0, 255).byte().numpy()
    return out.transpose(0, 2, 3, 1)       # (n, 64, 64, 3) uint8 RGB
