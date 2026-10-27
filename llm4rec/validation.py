"""Experiment invariants shared by training entry points."""


def validate_training_splits(train_split: str, val_split: str) -> None:
    if (train_split, val_split) != ("train", "val"):
        raise ValueError(
            "Ranker training requires train_split=train and val_split=val. "
            "Build temporal train candidates first; test is reserved for final evaluation."
        )
