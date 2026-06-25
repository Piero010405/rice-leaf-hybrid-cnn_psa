from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    _ = parser.parse_args()
    raise NotImplementedError("La evaluación ya se ejecuta al final del entrenamiento. Si se requiere evaluación independiente, reutilizar runner.predict con config.yml y best_model.pt.")


if __name__ == "__main__":
    main()
