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

CR_CTC = dict(LINE_RECIPE, cr_alpha=0.2, train_metrics=["loss_ctc", "loss_cr", "cer", "wer"])

ZIPFORMER_BLOCKS = {"activation": "swoosh", "bypass": True}

LINE_PRESETS = {
    "van": ({}, {}),
    "L1": ({}, LINE_RECIPE),
    "L2": (ZIPFORMER_BLOCKS, LINE_RECIPE),
    "L3": (dict(ZIPFORMER_BLOCKS, ds_factor_w=2), LINE_RECIPE),
    "L4": (dict(ZIPFORMER_BLOCKS, ds_factor_w=2, norm_dsc="bias"), LINE_RECIPE),
    "L4_no_ds": (dict(ZIPFORMER_BLOCKS, ds_factor_w=1, norm_dsc="bias"), LINE_RECIPE),
    "L5": (dict(ZIPFORMER_BLOCKS, ds_factor_w=2, norm_dsc="bias"), CR_CTC),
    "L6": (dict(ZIPFORMER_BLOCKS, ds_factor_w=2, norm_dsc="bias"), dict(CR_CTC, cr_mask_frac=0.3, cr_mask_max_px=64)),
}

# paragraph-level training of ZipVAN (the encoder and decoder come from a line-level run)
PARAGRAPH_RECIPE = {
    "batch_size": 2,
    "max_nb_epochs": 1000,
    "manual_seed": 1111,
    "use_amp": False,
    "optimizer": {
        "class": ScaledAdam,
        "args": {"lr": 0.002, "clipping_scale": 2.0},
    },
    "lr_scheduler": {
        "type": Eden,
        "args": {"lr_batches": 5000, "lr_epochs": 15, "warmup_batches": 500},
    },
    "ctc_reduction": "mean",
    "ctc_zero_infinity": True,
}


def apply_preset(params, model_params, training_params):
    params["model_params"].update(copy.deepcopy(model_params))
    params["training_params"].update(copy.deepcopy(training_params))
