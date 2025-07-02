import torch
from torch import Tensor, LongTensor
from einops import rearrange
import torch.nn.functional as F

class BaseBucketizer:
    """
    Interface for bucketizer to convert continuous / discrete variables
    Subclasses should override self._boundaries and self._centers
    """

    def __init__(self, n_boundaries: int = 128) -> None:
        self._n_boundaries = n_boundaries

        # below should be overriden
        self._boundaries = torch.tensor([])
        self._centers = torch.tensor([])

    def __call__(self, data: Tensor) -> Tensor:
        data = torch.clamp(data, min=0.0, max=1.0)
        return torch.bucketize(data, self._boundaries)

    def encode(self, data: Tensor) -> LongTensor:
        return self(data)  # type: ignore

    def decode(self, index: LongTensor) -> Tensor:
        index = torch.clamp(index, min=0, max=len(self._centers) - 1)  # type: ignore
        return F.embedding(index, self._centers)[..., 0]

    @property
    def boundaries(self) -> Tensor:
        return self._boundaries

    @property
    def centers(self) -> Tensor:
        return self._centers


class LinearBucketizer(BaseBucketizer):
    """
    Uniform bucketization between 0.0 to 1.0
    """

    def __init__(self, n_boundaries: int = 128) -> None:
        super().__init__(n_boundaries)
        arr = torch.arange(self._n_boundaries + 1) / self._n_boundaries
        starts, ends = arr[:-1], arr[1:]
        self._boundaries = ends
        self._centers = rearrange((starts + ends) / 2.0, "n -> n 1")
        
def test():
    bucketizer = LinearBucketizer(128)
    data = torch.rand(100)
    encoded = bucketizer(data)
    print(encoded)
    print(data)
    decoded = bucketizer.decode(encoded)
    print(decoded)
    print(torch.eq(data, decoded))
    
if __name__ == "__main__":
    test()