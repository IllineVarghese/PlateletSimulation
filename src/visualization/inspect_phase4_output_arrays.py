from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEARCH_DIRS = [
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "data",
]


def describe_array(path: Path) -> None:
    try:
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            print(f"\n{path.relative_to(PROJECT_ROOT)}")
            print("-" * 80)
            print(f"type:  npy")
            print(f"shape: {arr.shape}")
            print(f"dtype: {arr.dtype}")

            if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                print(f"min:   {np.nanmin(arr):.6f}")
                print(f"max:   {np.nanmax(arr):.6f}")
                print(f"mean:  {np.nanmean(arr):.6f}")

        elif path.suffix == ".npz":
            data = np.load(path, allow_pickle=False)
            print(f"\n{path.relative_to(PROJECT_ROOT)}")
            print("-" * 80)
            print(f"type:  npz")
            print(f"keys:  {list(data.keys())}")

            for key in data.keys():
                arr = data[key]
                print(f"  key={key}")
                print(f"    shape: {arr.shape}")
                print(f"    dtype: {arr.dtype}")
                if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                    print(f"    min:   {np.nanmin(arr):.6f}")
                    print(f"    max:   {np.nanmax(arr):.6f}")
                    print(f"    mean:  {np.nanmean(arr):.6f}")

    except Exception as error:
        print(f"\nCould not read {path.relative_to(PROJECT_ROOT)}")
        print(f"Reason: {error}")


def main() -> None:
    print("Phase 5 Week 2 Day 2: Inspecting existing Phase 4 output arrays")
    print("=" * 80)

    files = []

    for search_dir in SEARCH_DIRS:
        if not search_dir.exists():
            continue

        files.extend(search_dir.rglob("*.npy"))
        files.extend(search_dir.rglob("*.npz"))

    files = sorted(set(files))

    if not files:
        print("No .npy or .npz files found in results/ or data/.")
        return

    print(f"Found {len(files)} array file(s).")

    for path in files:
        describe_array(path)

    print("\nInspection complete.")
    print("Look for arrays shaped like:")
    print("  positions:  (frames, platelets, 3) or (platelets, 3)")
    print("  activation: (frames, platelets) or (platelets,)")
    print("  shear:      (frames, platelets) or (platelets,)")


if __name__ == "__main__":
    main()