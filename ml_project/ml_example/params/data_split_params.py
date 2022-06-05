from dataclasses import dataclass


@dataclass()
class SplittingParams:
    split_size: float
    random_state: int
    shuffle: bool
