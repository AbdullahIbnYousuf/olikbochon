from __future__ import annotations

import pandas as pd

from olikbochon.v13_public_pool import add_keys
from olikbochon.v13_template_transfer import FrozenTemplateTransfer, THRESHOLDS


def _rows(count: int) -> pd.DataFrame:
    return add_keys(
        pd.DataFrame(
            [
                {
                    "context": "[NULL]",
                    "prompt_bn": f"প্রশ্ন {index}",
                    "response_bn": f"উত্তর {index}",
                    "label": index % 2,
                }
                for index in range(count)
            ]
        ),
        workers=6,
    )


def test_frozen_transfer_uses_exact_grid_and_seven_neighbours() -> None:
    train = _rows(80)
    validation = _rows(40).assign(prompt_bn=lambda value: value.prompt_bn + " নতুন")
    validation = add_keys(validation.iloc[:, :4], workers=6)
    model = FrozenTemplateTransfer(train)
    calibration = model.calibrate(validation)
    assert [row["threshold"] for row in calibration["nearest_neighbour_thresholds"]] == list(THRESHOLDS)
    output = model.predict(validation)
    assert len(output.predictions) == len(validation)
    assert set(output.predictions).issubset({0, 1})
