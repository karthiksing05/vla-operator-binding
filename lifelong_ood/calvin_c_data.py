"""CALVIN-C training data: FLOWER's CALVIN language dataset (flower_vla_calvin's ExtendedDiskDataset, unchanged) restricted to
the annotation windows a split seed keeps (experiments/c_calvin_split.py `windows`: held-out cells dropped, and every window
overlapping a held-out window by more than 16 frames).

Importable only where flower_vla_calvin is on sys.path (the CALVIN venv; experiments/c_train_flower.py puts it there).  The
filter touches the TRAINING directory only; the validation set (scene D, logged loss only, never used for selection) is left as
the recipe builds it.
"""
from __future__ import annotations

import logging

import numpy as np
from flower.datasets.disk_dataset import ExtendedDiskDataset

logger = logging.getLogger(__name__)


class CalvinCDiskDataset(ExtendedDiskDataset):
    def __init__(self, *args, keep_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_file = keep_file
        self.n_before = int(len(self.episode_lookup))
        if keep_file is None or self.validation:
            return
        z = np.load(keep_file)
        keep, indx = np.asarray(z["keep"], bool), np.asarray(z["indx"])
        lang = np.load(self.abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        assert np.array_equal(np.asarray(lang["info"]["indx"]), indx), "keep file is not aligned with the annotations"
        assert len(keep) == len(self.lang_text), (len(keep), len(self.lang_text))
        ll = np.asarray(self.lang_lookup)
        m = keep[ll]
        self.episode_lookup = self.episode_lookup[m]
        self.lang_lookup = ll[m].tolist()
        self.n_windows_kept = int(keep.sum())
        msg = (f"[calvin_c] {keep_file}: kept {int(keep.sum())} / {len(keep)} windows, "
               f"{len(self.episode_lookup)} / {self.n_before} sample starts")
        logger.info(msg)
        print(msg, flush=True)
