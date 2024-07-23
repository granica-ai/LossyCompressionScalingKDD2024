"""Script to run an experiment to find the empirical excess error the theory plot."""

import os
import simplejson as json
import datetime
import numpy as np
import dataclasses
from theory.empirical import ExptGrid, run_expt
from utils.consts import THEORY_PATH


def run_empirical_expt():
    """Find the empirical excess error for a range of n and m."""
    grid = ExptGrid(
        p=2.01,
        q=2,
        r=0.99,
        tau=1,
        ms=np.arange(2, 10),
        ns=np.array(10 ** np.linspace(2, 5, 10), dtype=int),
        extra_m=5,
        niters=50,
        alpha_options=10 ** np.linspace(-2, 2, 20),
    )
    output_dir = os.path.join(
        THEORY_PATH,
        "excess_err",
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(output_dir)
    d = dataclasses.asdict(grid)
    d["ns"] = d["ns"].tolist()
    d["ms"] = d["ms"].tolist()
    d["alpha_options"] = d["alpha_options"].tolist()
    with open(f"{output_dir}/grid.json", "w") as f:
        json.dump(d, f, indent=4)

    empirical, _ = run_expt(grid)
    empirical["excess_err"] = empirical["test_error"] - grid.tau**2
    empirical.to_csv(os.path.join(output_dir, "empirical.csv"), index=False)


if __name__ == "__main__":
    run_empirical_expt()
