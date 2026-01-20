# haar measure
# normalized?
# layer opti



from optimize.optimize_jax import *

from ansatz.ansatz_jax import *

from utils import rand
from utils.job_data import JobData
from utils.circuit import apply_clifford, make_clifford_circuit, stitch_circuits
from utils.random_stabilizer import random_stabilizer_toggles_ag
from utils.process_io import get_shots_and_xeb_scores




import numpy as np
import time
from datetime import datetime
import pathlib
import logging
from concurrent.futures import ProcessPoolExecutor


import matplotlib.pyplot as plt
# This is necssary for platforms that run os.fork() by default in the ProcessPoolExectuor
# It is not necssasry on macOS, but it is on certain Linux platforms.
# import multiprocessing



def random_haar_state(n, rng):
    """
    Generate a single exactly normalized Haar-random n-qubit state.
    """
    real_part = rng.normal(size=([2] * n))
    imag_part = rng.normal(size=([2] * n))
    state = real_part + 1j * imag_part
    state /= np.linalg.norm(state)
    return state


def run_optimization(job):
    i = job["i"]
    s = job["seed"]
    target_state = job["target_state"]
    n = job["n"]
    depth = job["depth"]
    noisy = job["noisy"]

    # Ensure initial parameters are chosen consistently pseudorandomly
    rng = np.random.default_rng(s)

    start = time.time()

    opt = optimize(target_state, depth, noisy=noisy, rng=rng)
    opt_params = opt.x

    out_state = output_state(n, opt_params)
    noiseless_fidelity = -loss(opt_params, target_state)
    noise_fidelity = fidelity_from_noise(n, opt_params)

    return {
        "i": i,
        "opt_params": opt_params,
        "output_state": out_state,
        "noiseless_fidelity": noiseless_fidelity,
        "noise_fidelity": noise_fidelity,
        "time": time.time() - start,
    }

# I believe this is necessary on macs for ProcessPoolExecutor
if __name__ == "__main__":

    seed = 1234
    n = 12
    noisy = True

    depths = list(range(10, 110, 10))
    noiseless_fidelities = []
    noise_fidelities = []
    fidelities = []

    # (Optional, but recommended) keep the same target state for all depths
    # so the plot reflects "depth effect" rather than "different random targets".
    rng_target = np.random.default_rng(seed)
    target_state = random_haar_state(n, rng=rng_target)

    for depth in depths:
        i = 1  # job index (not important here)
        job = {
            "i": i,
            "seed": seed,
            "target_state": target_state,
            "n": n,
            "depth": depth,
            "noisy": noisy,
        }

        result = run_optimization(job)

        nf = float(np.array(result["noiseless_fidelity"]))
        zf = float(np.array(result["noise_fidelity"]))
        f=nf*zf

        noiseless_fidelities.append(nf)
        noise_fidelities.append(zf)
        fidelities.append(f)

        print(f"depth={depth:2d} | "f"noiseless fidelity={nf:.6f} | "f"noise fidelity={zf:.6f} | "f"overall fidelity={f:.6f}")


    # ---------- Plot ----------
    plt.figure()

    plt.plot(depths, noiseless_fidelities, marker="o", label="Noiseless fidelity")
    plt.plot(depths, noise_fidelities, marker="s", label="Noise fidelity factor")
    plt.plot(depths, fidelities, marker="^", label="Overall fidelity")

    plt.xlabel("Circuit depth")
    plt.ylabel("Fidelity")
    plt.title("Fidelity vs depth")
    plt.legend()
    plt.grid(True)

    plt.show()






    # This is necssary for platforms that run os.fork() by default in the ProcessPoolExectuor
    # It is not necssasry on macOS, but it is on certain Linux platforms.
    # multiprocessing.set_start_method("spawn")

    # n = 3#12
    # depth = 2#86
    # noisy = True
    # device_name = "H1-1LE" # Change to H1-1 for actual experiment
    # detect_leakage = False

    # submit_job = False
    # n_stitches = 2
    # n_parallel = 1
    # n_shots = 1
    # start_seed = 0
    # n_seeds = 1 # Change to 10000 for actual experiment

    # assert n_parallel % n_stitches == 0

    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_path = f"logs/{device_name}/n_{n}_depth_{depth}"
    # pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    # log_filename = (
    #     f"{log_path}/seeds_{start_seed}-{start_seed+n_seeds-1}_{timestamp}.txt"
    # )
    # logging.basicConfig(
    #     filename=log_filename,
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    # logging.info("-" * 30)
    # logging.info(f"n               : {n}")
    # logging.info(f"depth           : {depth}")
    # logging.info(f"noisy           : {noisy}")
    # logging.info(f"device_name     : {device_name}")
    # logging.info(f"detect_leakage  : {detect_leakage}")
    # logging.info(f"submit_job      : {submit_job}")
    # logging.info(f"n_stitches      : {n_stitches}")
    # logging.info(f"n_parallel      : {n_parallel}")
    # logging.info(f"n_shots         : {n_shots}")
    # logging.info("")
    # logging.info(f"start_seed      : {start_seed}")
    # logging.info(f"n_seeds         : {n_seeds}")
    # logging.info(
    #     f"n_submissions   : {n_seeds // n_stitches + bool(n_seeds % n_stitches)}"
    # )
    # logging.info("-" * 30)

    # for seed in range(start_seed, start_seed + n_seeds, n_parallel):
    #     batch = list(range(seed, min(seed + n_parallel, start_seed + n_seeds)))
    #     logging.info(f"processing seeds: {batch}")
    #     random_bits_list = [rand.read_chunk(s) for s in batch]
    #     rand_gens = [rand.TrueRandom(random_bits) for random_bits in random_bits_list]

    #     # First do all of the randomness generation
    #     target_states_r = [rand_gen.normal(size=([2] * n)) for rand_gen in rand_gens]
    #     target_states_i = [rand_gen.normal(size=([2] * n)) for rand_gen in rand_gens]
    #     target_states = [
    #         (target_state_r + 1j * target_state_i) / 2 ** ((n + 1) / 2)
    #         for target_state_r, target_state_i in zip(target_states_r, target_states_i)
    #     ]

    #     ag_toggle_lists = [
    #         random_stabilizer_toggles_ag(n, rand_gen) for rand_gen in rand_gens
    #     ]
    #     reversed_ag_toggle_lists = [list(reversed(lst)) for lst in ag_toggle_lists]

    #     # Optimize
    #     jobs = []
    #     for i, s in enumerate(batch):
    #         jobs.append(
    #             {
    #                 "i": i,
    #                 "seed": s,
    #                 "target_state": target_states[i],
    #                 "n": n,
    #                 "depth": depth,
    #                 "noisy": noisy,
    #             }
    #         run_optimization(jobs[i])
    #         )
        
      















