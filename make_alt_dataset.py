from pathlib import Path

import cv2
import numpy as np


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _clean_image(seed: int, h: int = 256, w: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), dtype=np.uint8)
    color = rng.integers(30, 180, size=(3,), dtype=np.uint8)
    base[:] = color

    for _ in range(3):
        x1, y1 = rng.integers(0, w // 2), rng.integers(0, h // 2)
        x2, y2 = rng.integers(w // 2, w), rng.integers(h // 2, h)
        c = rng.integers(70, 220, size=(3,), dtype=np.uint8)
        cv2.rectangle(base, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(int(v) for v in c), thickness=-1)

    blur = cv2.GaussianBlur(base, (11, 11), 0)
    return blur


def _virus_like_image(seed: int, h: int = 256, w: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = _clean_image(seed, h, w)

    noise = rng.normal(0, 40, size=(h, w, 3)).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for _ in range(35):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        radius = int(rng.integers(3, 14))
        c = tuple(int(v) for v in rng.integers(120, 255, size=(3,)))
        cv2.circle(noisy, (cx, cy), radius, c, thickness=1)

    return noisy


def create_dataset(output_root: str = "data/dataset_alt", per_class: int = 50) -> None:
    root = Path(output_root)
    virus_dir = root / "virus"
    non_virus_dir = root / "non_virus"

    for i in range(per_class):
        _save(non_virus_dir / f"clean_{i:03d}.png", _clean_image(1000 + i))
        _save(virus_dir / f"virus_{i:03d}.png", _virus_like_image(5000 + i))

    print(f"Dataset généré: {root}")
    print(f" - virus: {per_class}")
    print(f" - non_virus: {per_class}")


if __name__ == "__main__":
    create_dataset()
