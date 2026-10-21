"""Configurations of the reported runs.

Each line-level preset is a pair (model_params, training_params) of values that
replace the VAN configuration in main_line_ctc.py. The paragraph models in
main_pg_va.py take the model_params of the line preset their encoder comes from.
"""
import copy

from basic.scaled_optimizer import ScaledAdam, Eden


# optimizer, schedule and batch size of all ZipVAN line-level runs
LINE_RECIPE = {
    "batch_size": 4,
    "max_nb_epochs": 500,
    "manual_seed": 1111,
    "optimizer": {
        "class": ScaledAdam,
        "args": {"lr": 0.03, "clipping_scale": 2.0},
    },
    "lr_scheduler": {
        "type": Eden,
        "args": {"lr_batches": 15000, "lr_epochs": 30, "warmup_batches": 500},
    },
    "ctc_reduction": "mean",
    "ctc_zero_infinity": True,
}

ZIPFORMER_BLOCKS = {"activation": "swoosh", "bypass": True}

LINE_PRESETS = {
    "van": ({}, {}),
    "L1": ({}, LINE_RECIPE),
    "L2": (ZIPFORMER_BLOCKS, LINE_RECIPE),
}


def apply_preset(params, model_params, training_params):
    params["model_params"].update(copy.deepcopy(model_params))
    params["training_params"].update(copy.deepcopy(training_params))
