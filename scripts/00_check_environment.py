from __future__ import annotations

import sys
import platform


def main():
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    try:
        import torch
        print("Torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("Torch check failed:", exc)
    for lib in ["cv2", "skimage", "pywt", "sklearn", "pandas", "numpy", "omegaconf"]:
        try:
            mod = __import__(lib)
            print(f"{lib}: OK")
        except Exception as exc:
            print(f"{lib}: FAIL - {exc}")


if __name__ == "__main__":
    main()
