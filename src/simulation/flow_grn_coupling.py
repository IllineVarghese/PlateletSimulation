from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_grn_shear_input_table(
    normalized_shear: np.ndarray,
) -> pd.DataFrame:
    """
    Convert normalized shear array into GRN input table.

    Output column:
        InShearStress

    Shape expected:
        normalized_shear: frames x agents
    """
    rows = []

    for frame_id in range(normalized_shear.shape[0]):
        for agent_id in range(normalized_shear.shape[1]):
            rows.append(
                {
                    "frame": frame_id,
                    "agent_id": agent_id,
                    "InShearStress": float(normalized_shear[frame_id, agent_id]),
                }
            )

    return pd.DataFrame(rows)


def export_grn_shear_inputs(
    base_dir: str | Path = "results/phase4/week1_flow_validation",
) -> None:
    base_dir = Path(base_dir)

    normalized_shear_path = base_dir / "normalized_shear.npy"
    normalized_shear = np.load(normalized_shear_path)

    df = build_grn_shear_input_table(normalized_shear)

    out_csv = base_dir / "phase4_grn_shear_inputs.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved GRN shear input table: {out_csv}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"InShearStress range: {df['InShearStress'].min():.4f} to {df['InShearStress'].max():.4f}")


if __name__ == "__main__":
    export_grn_shear_inputs()