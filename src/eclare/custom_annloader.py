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
    """Custom AnnLoader that allows specifying explicit indices."""

    def __init__(self, *args, indices=None, **kwargs):
        self.indices = indices

        # Check if indices are provided
        if indices is not None:
            batch_size = kwargs.get("batch_size", 1)
            shuffle = kwargs.get("shuffle", False)
            drop_last = kwargs.get("drop_last", False)

            # Use custom sampler before calling super().__init__()
            kwargs["sampler"] = CustomBatchIndexSampler(
                len(args[0]), batch_size=batch_size, indices=indices, shuffle=shuffle, drop_last=drop_last
            )
            kwargs["batch_size"] = None  # Ensures batch_size is handled by sampler

        # Now call the parent class constructor with updated kwargs
        super().__init__(*args, **kwargs)



# Override specific functions/classes
globals().update({
    "BatchIndexSampler": CustomBatchIndexSampler,
    "AnnLoader": CustomAnnLoader,
})


def get_custom_ann_loader(*args, **kwargs):
    """Wrapper function to instantiate the CustomAnnLoader."""
    return CustomAnnLoader(*args, **kwargs)
