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
    fname = {'real': 'cnn_real.pt', 'aug': 'cnn_aug.pt'}[variant]
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, NC)
    m.load_state_dict(torch.load(_file(fname), map_location='cpu', weights_only=True))
    return m.eval()

@torch.no_grad()
def classify(model, rgb_uint8, k=3):
    # rgb_uint8: HxWx3 uint8 RGB image (any size)
    pre = preprocess(rgb_uint8)
    x = torch.from_numpy(pre.transpose(2, 0, 1)[None].copy()).float() / 255.
    x = F.interpolate(x, size=128, mode='bilinear', align_corners=False)
    p = F.softmax(model((x - _MEAN) / _STD), 1)[0].numpy()
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
    g.load_state_dict(torch.load(_file('generator.pt'), map_location='cpu', weights_only=True))
    return g.eval()

@torch.no_grad()
def generate(gen, class_name, n=8):
    label = classes.index(class_name)
    z = torch.randn(n, NZ)
    y = torch.full((n,), label, dtype=torch.long)
    out = ((gen(z, y) + 1) * 127.5).clamp(0, 255).byte().numpy()
    return out.transpose(0, 2, 3, 1)       # (n, 64, 64, 3) uint8 RGB
