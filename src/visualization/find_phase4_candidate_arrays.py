from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIRS = [PROJECT_ROOT / "results", PROJECT_ROOT / "data"]


def looks_relevant(path: Path, name: str, shape: tuple[int, ...]) -> bool:
    text = f"{path.name.lower()} {name.lower()}"

    keyword_match = any(
        key in text
        for key in [
            "pos",
            "position",
            "trajectory",
            "traj",
            "activation",
            "act",
            "shear",
            "state",
            "platelet",
        ]
    )

    shape_match = (
        len(shape) == 3 and shape[-1] == 3
    ) or (
        len(shape) == 2
    ) or (
        len(shape) == 1
    )

    return keyword_match and shape_match


def print_candidate(path: Path, array_name: str, arr: np.ndarray) -> None:
    rel = path.relative_to(PROJECT_ROOT)

    print("\nCANDIDATE")
    print("-" * 70)
    print(f"file:  {rel}")
    print(f"array: {array_name}")
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")

    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        print(f"min:   {np.nanmin(arr):.6f}")
        print(f"max:   {np.nanmax(arr):.6f}")
        print(f"mean:  {np.nanmean(arr):.6f}")


def main() -> None:
    print("Searching for likely Phase 4 position / activation / shear arrays")
    print("=" * 70)

    candidate_count = 0

    files = []
    for search_dir in SEARCH_DIRS:
        if search_dir.exists():
            files.extend(search_dir.rglob("*.npy"))
            files.extend(search_dir.rglob("*.npz"))

    for path in sorted(set(files)):
        try:
            if path.suffix == ".npy":
                arr = np.load(path, allow_pickle=False)
                if looks_relevant(path, "array", arr.shape):
                    print_candidate(path, "array", arr)
                    candidate_count += 1

            elif path.suffix == ".npz":
                data = np.load(path, allow_pickle=False)
                for key in data.keys():
                    arr = data[key]
                    if looks_relevant(path, key, arr.shape):
                        print_candidate(path, key, arr)
                        candidate_count += 1

        except Exception as error:
            print(f"\nSkipped unreadable file: {path.relative_to(PROJECT_ROOT)}")
            print(f"Reason: {error}")

    print("\nDone.")
    print(f"Total likely candidates found: {candidate_count}")

    print("\nMost useful shapes:")
    print("positions  → (frames, platelets, 3) or (platelets, 3)")
    print("activation → (frames, platelets) or (platelets,)")
    print("shear      → (frames, platelets) or (platelets,)")


if __name__ == "__main__":
    main()