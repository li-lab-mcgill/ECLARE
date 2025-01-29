import importlib
import numpy as np

# Import the original annloader module
original_annloader = importlib.import_module("anndata.experimental.pytorch._annloader")

# Import everything from original _annloader.py
globals().update({name: getattr(original_annloader, name) for name in dir(original_annloader) if not name.startswith("__")})


class CustomBatchIndexSampler(original_annloader.BatchIndexSampler):
    """Custom version of BatchIndexSampler that supports explicit indices."""

    def __init__(self, n_obs, batch_size, indices=None, shuffle=False, drop_last=False):
        super().__init__(n_obs, batch_size, shuffle, drop_last)
        self.indices = indices if indices is not None else None

    def __iter__(self):
        indices = self.indices if self.indices is not None else (
            np.random.permutation(self.n_obs).tolist() if self.shuffle else list(range(self.n_obs))
        )
        for i in range(0, self.n_obs, self.batch_size):
            batch = indices[i: min(i + self.batch_size, self.n_obs)]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch


class CustomAnnLoader(original_annloader.AnnLoader):
    """Custom version of AnnLoader that supports explicit indices."""

    def __init__(self, *args, indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indices = indices

        if isinstance(self.sampler, original_annloader.BatchIndexSampler):
            self.sampler = CustomBatchIndexSampler(
                len(self.dataset), self.batch_size, indices=self.indices,
                shuffle=self.sampler.shuffle, drop_last=self.sampler.drop_last
            )


# Override specific functions/classes
globals().update({
    "BatchIndexSampler": CustomBatchIndexSampler,
    "AnnLoader": CustomAnnLoader,
})


def get_custom_ann_loader(*args, **kwargs):
    """Wrapper function to instantiate the CustomAnnLoader."""
    return CustomAnnLoader(*args, **kwargs)
