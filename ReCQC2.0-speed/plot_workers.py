"""Render separate result molecules without sharing Matplotlib state."""

import __main__
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path


def _init_plot_worker():
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")


def render_molecules(draw, results, output_dir):
    rows = list(results)
    workers = min(os.cpu_count() or 1, 4)
    main_file = getattr(__main__, "__file__", "")
    if (
        workers > 1
        and len(rows) >= 12
        and len({row[0] for row in rows}) == len(rows)
        and main_file
        and Path(main_file).is_file()
        and not getattr(sys, "frozen", False)
    ):
        groups = [rows[i::workers] for i in range(workers)]
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"), initializer=_init_plot_worker
            ) as pool:
                for _ in pool.map(draw, groups, repeat(output_dir)):
                    pass
            return
        except (BrokenProcessPool, OSError):
            # Complete the original output set if worker startup is unavailable.
            pass
    draw(rows, output_dir)
