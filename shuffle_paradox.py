#!/usr/bin/env python3
"""shuffle_paradox.py"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

playlist_size = 500
total_sessions = 10_000
max_plays = 80

# Same paradox with different "calendars" (number of equally likely values)
calendars = {
    "Artists (50)": 50,
    "Phone digits (100)": 100,
    "Birthdays (365)": 365,
    "Songs (500)": 500,
}

magic = np.sqrt(2 * np.log(2))  # hidden constant (about 1.1774)

rng = np.random.default_rng(2026)  # fixed seed so the slide numbers stay the same

chart_dir = Path(__file__).parent / "charts"


def repeat_probability(n, plays):
    """Exact probability of at least one repeat in plays random picks from n items"""
    if plays > n:  # pigeonhole principle
        return 1.0
    i = np.arange(plays)
    return 1.0 - np.prod((n - i) / n)


def plays_needed(n, target):
    """Smallest number of plays where the repeat probability reaches target"""
    plays = 1
    while repeat_probability(n, plays) < target:
        plays += 1
    return plays


def first_repeat(n):
    """Play random songs until one repeats. Returns the play number of the repeat."""
    heard = set()
    play = 0
    while True:
        play += 1
        song = rng.integers(0, n)
        if song in heard:
            return play
        heard.add(song)


def run_sessions():
    """Returns an array with the first repeat play number for every session"""
    return np.array([first_repeat(playlist_size) for _ in range(total_sessions)])


def print_stats(results):
    mean = np.mean(results)
    median = np.median(results)
    mode = np.argmax(np.bincount(results))
    std = np.std(results, ddof=1)  # sample std (Bessel's correction)

    print(f"Mean   = {mean:.2f}")
    print(f"Median = {median:.1f}")
    print(f"Mode   = {mode}")
    print(f"Std    = {std:.2f}")
    print(f"Range  = {results.min()} to {results.max()}")
    if mean > median:
        print("Mean > Median, so the distribution is right skewed")
    else:
        print("Mean <= Median, so the distribution is not right skewed")
    return mean, median


def plot_curve(ax, results, k50, k90):
    k = np.arange(1, max_plays + 1)
    sim = [np.mean(results <= i) for i in k]
    exact = [repeat_probability(playlist_size, i) for i in k]

    ax.step(k, sim, color="black", linewidth=3, label="Estimated")
    ax.plot(k, exact, color="blue", label="Actual")
    for x, p in [(k50, 0.5), (k90, 0.9)]:
        ax.vlines(x, 0, p, color="red")
        ax.annotate(f"{p:.0%} by song {x}", xy=(x, p), xytext=(x + 5, p - 0.1),
                    arrowprops={"facecolor": "black"})
    ax.set_title(f"Shuffle Paradox ({playlist_size} songs, {total_sessions:,} sessions)")
    ax.set_xlabel("Songs Played")
    ax.set_ylabel("Probability of a Repeat")
    ax.legend(loc="upper left")


def plot_histogram(ax, results, mean, median):
    bins = np.arange(1, results.max() + 2) - 0.5
    ax.hist(results, bins=bins, color="blue", edgecolor="white")
    ax.axvline(median, color="red", label=f"Median = {median:.1f}")
    ax.axvline(mean, color="black", label=f"Mean = {mean:.1f}")
    ax.set_title(f"First Repeat ({total_sessions:,} sessions)")
    ax.set_xlabel("Song Number of First Repeat")
    ax.set_ylabel("Number of Sessions")
    ax.set_xlim(0, 90)
    ax.legend(loc="upper right")


def plot_calendars(ax):
    k = np.arange(1, max_plays + 1)
    for name, n in calendars.items():
        y = [repeat_probability(n, i) for i in k]
        k50 = plays_needed(n, 0.5)
        ax.plot(k, y, label=f"{name}: 50% at {k50}")
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_title("Same Paradox, Different Calendars")
    ax.set_xlabel("Number of Picks")
    ax.set_ylabel("Probability of a Repeat")
    ax.legend(loc="lower right")


def plot_magic(ax, sizes, ratios):
    ax.plot(sizes, ratios, marker="o", label="Plays for 50% / sqrt(n)")
    ax.axhline(magic, color="red", linestyle="--", label=f"sqrt(2 ln 2) = {magic:.4f}")
    ax.set_xscale("log")
    ax.set_title("Magic Number")
    ax.set_xlabel("Playlist Size (log scale)")
    ax.set_ylabel("Plays for 50% / sqrt(n)")
    ax.legend(loc="upper right")


def main():
    chart_dir.mkdir(exist_ok=True)

    # Exact answer
    k50 = plays_needed(playlist_size, 0.5)
    k90 = plays_needed(playlist_size, 0.9)
    print(f"Playlist size = {playlist_size}")
    print(f"50% chance of a repeat by song {k50}")
    print(f"90% chance of a repeat by song {k90}")
    print()

    # Simulation
    results = run_sessions()
    print(f"First repeat over {total_sessions:,} sessions")
    mean, median = print_stats(results)
    print()

    # Different calendars
    print("Calendar\t\tSize\t50% at\t90% at")
    for name, n in calendars.items():
        print(f"{name}\t{n}\t{plays_needed(n, 0.5)}\t{plays_needed(n, 0.9)}")
    print()

    # Magic number
    sizes = [10, 20, 50, 100, 200, 365, 500, 1000, 2000, 5000, 10_000, 50_000, 100_000]
    ratios = []
    print("Size\t50% at\tRatio")
    for n in sizes:
        k = plays_needed(n, 0.5)
        ratios.append(k / np.sqrt(n))
        print(f"{n}\t{k}\t{ratios[-1]:.4f}")
    print(f"Magic number = sqrt(2 ln 2) = {magic:.4f}")
    print()

    # Live demo table
    print("People\tBirthday\tPhone")
    for people in range(10, 41, 5):
        print(f"{people}\t{repeat_probability(365, people):.1%}"
              f"\t\t{repeat_probability(100, people):.1%}")
    if len(sys.argv) > 1:
        people = int(sys.argv[1])
        print(f"Room of {people}: birthday {repeat_probability(365, people):.1%}, "
              f"phone {repeat_probability(100, people):.1%}")

    # All 4 plots in one window (2 rows x 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), num=Path(__file__).name)
    plot_curve(axs[0, 0], results, k50, k90)
    plot_histogram(axs[0, 1], results, mean, median)
    plot_calendars(axs[1, 0])
    plot_magic(axs[1, 1], sizes, ratios)
    fig.tight_layout()
    fig.savefig(chart_dir / "shuffle_paradox_all.png")
    plt.show()


if __name__ == "__main__":
    main()
