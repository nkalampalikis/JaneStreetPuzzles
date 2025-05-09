# Puzzle: Sum One, Somewhere

# For a fixed p, independently label the nodes of an infinite complete binary tree 0 with probability p,
# and 1 otherwise. For what p is there exactly a 1/2 probability that there exists an infinite path down
# the tree that sums to at most 1 (that is, all nodes visited, with the possible exception of one, will be labeled 0).
# Find this value of p accurate to 10 decimal places.

# Here is a solution to the April 2025 puzzle:

import multiprocessing
import numpy as np

# Parameters
TREE_DEPTH = 100
TRIALS_PER_P = 100000
EPSILON = 1e-6

def single_trial_dfs(args):
    """Perform a single trial using depth-first search."""
    p, seed = args
    rng = np.random.default_rng(seed)
    stack = [(0, 0 if rng.random() < p else 1)]  # (depth, ones_used)

    while stack:
        depth, ones_used = stack.pop()
        if depth >= TREE_DEPTH:
            return 1  # Reached depth with at most one 1

        # Generate left and right child
        for _ in range(2):
            label = 0 if rng.random() < p else 1
            if ones_used + label <= 1: # If there is more that one 1 and there is no point exploring this path further
                stack.append((depth + 1, ones_used + label))

    return 0  # No valid path found

def run_parallel_trials(p):
    """Run multiple paralel trials and return the success rate."""
    rng = np.random.default_rng()
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=TRIALS_PER_P, dtype=np.uint32)
    args = [(p, seed) for seed in seeds]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(single_trial_dfs, args)

    stderr = np.std(results, ddof=1) / np.sqrt(len(results))
    mean = np.mean(results)
    return [mean, stderr]  # Return mean and standard error


def find_critical_p():
    """Use binary search to find the critical p value"""
    low, high = 0.5305, 0.5307  # Initial bounds for p
    while high - low > EPSILON:
        mid = low + (high - low) / 2
        print(".", end="", flush=True)
        p = run_parallel_trials(mid) # Probability of success should be 0.5
        if p[0] < 0.5:
            low = mid
        else:
            high = mid
    return [low + (high - low) / 2, p[1]]  # Return the critical p value and its standard error

def main():
    """Run the algorithm to find the critical p value."""
    print("##### Finding critical p value #####")
    print(f"Using {TRIALS_PER_P} trials per p value")
    print(f"Tree depth set to {TREE_DEPTH}, and epsilon set to {EPSILON}")
    print("Starting the search.", end="", flush=True)
    result = find_critical_p()
    print(f" => Estimated critical p ≈ {result[0]:.6f}, stderr ≈ {result[1]:.10f}")

if __name__ == "__main__":
    main()