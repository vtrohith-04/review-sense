from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    print("Review Sense baseline training scaffold")
    print(f"Project root: {project_root}")
    print("Next step: add dataset loading, preprocessing, and TF-IDF training.")


if __name__ == "__main__":
    main()
