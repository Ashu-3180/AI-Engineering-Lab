"""Command-line entry point for reproducible Tiny Shakespeare acquisition."""

from pathlib import Path

from .config import load_data_config
from .corpus import acquire_corpus, resolve_dataset_paths


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_data_config(project_root / "config" / "data.toml")
    paths = resolve_dataset_paths(
        project_root,
        config.dataset_id,
        raw_data_dir=config.raw_data_dir,
        provenance_dir=config.provenance_dir,
    )
    provenance = acquire_corpus(
        dataset_id=config.dataset_id,
        source_url=config.source_url,
        destination=paths.raw_file,
        provenance_path=paths.provenance_file,
    )
    print(f"Acquired {provenance.dataset_id}: {provenance.sha256}")


if __name__ == "__main__":
    main()
