Developer instructions for xrsdkit models
=========================================

Training
--------

The model training is consolidated to xrsdkit.tools.modeling_tools.
To train or re-train xrsdkit models, use that module as follows:

    from xrsdkit.tools.modeling_tools import train_models
    train_models()

To search for optimal hyperparameters during training:

    train_models(True)

To examine a new set of models:

    # TODO: train without saving, print training summaries

To decide to save the new models:

    # TODO: call functions to save the new models




