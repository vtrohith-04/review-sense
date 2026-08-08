from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    print("Review Sense transformer training scaffold")
    print(f"Project root: {project_root}")
    print("Next step: add tokenization, trainer config, and multi-label fine-tuning.")


if __name__ == "__main__":
    main()
