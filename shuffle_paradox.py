#!/usr/bin/env python3
"""shuffle_paradox.py"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

playlist_size = 500  # number of songs in the playlist
total_sessions = 10_000  # number of listening sessions to simulate
max_plays = 80  # x-axis limit for the probability curves

# Same paradox with different "calendars" (number of equally likely values)
calendars = {
    "Artists (50)": 50,
    "Phone digits (100)": 100,
    "Birthdays (365)": 365,
    "Songs (500)": 500,
}

magic = np.sqrt(2 * np.log(2))  # hidden constant (about 1.1774)

rng = np.random.default_rng(2026)  # fixed seed so the slide numbers stay the same

chart_dir = Path(__file__).parent / "charts"  # folder where the plot image is saved


def repeat_probability(n, plays):
    """Exact probability of at least one repeat in plays random picks from n items"""
    # More picks than items means a repeat is guaranteed (pigeonhole principle)
    # Example: 501 songs played from a 500-song playlist must repeat at least one
    if plays > n:
        return 1.0
    # Logic: it is easier to find the chance of NO repeat, then subtract from 1
    # Pick 1 can be any song             -> n/n     = 500/500
    # Pick 2 must avoid the 1 song heard -> (n-1)/n = 499/500
    # Pick 3 must avoid the 2 songs heard -> (n-2)/n = 498/500
    # All picks must avoid a repeat, so we MULTIPLY these chances together
    i = np.arange(plays)  # 0, 1, 2, ..., plays-1 (number of songs already heard)
    no_repeat = np.prod((n - i) / n)  # n/n * (n-1)/n * ... * (n-plays+1)/n
    # Repeat and no repeat are opposites, so they add up to 1
    return 1.0 - no_repeat


def plays_needed(n, target):
    """Smallest number of plays where the repeat probability reaches target"""
    # Start at 1 play and keep adding a play until the chance reaches the target
    # The chance only goes up with each extra play, so the first play count
    # that reaches the target is the answer (e.g. 27 for 50% with 500 songs)
    plays = 1
    while repeat_probability(n, plays) < target:
        plays += 1
    return plays


def first_repeat(n):
    """Play random songs until one repeats. Returns the play number of the repeat."""
    # Logic: act like a music player that picks every song at random
    # (the same song can be picked again, just like a real repeat)
    # A set is used because checking "is this song in the set?" is very fast
    heard = set()  # songs we have already heard this session
    play = 0  # how many songs have played so far
    # Loop with no fixed end: a repeat always happens by play n+1 (pigeonhole)
    while True:
        play += 1
        song = rng.integers(0, n)  # pick a random song number from 0 to n-1
        if song in heard:  # already heard it, so this play is the first repeat
            return play
        heard.add(song)  # new song, remember it and keep playing


def run_sessions():
    """Returns an array with the first repeat play number for every session"""
    # Simulate total_sessions sessions and store when each one first repeated
    # Example result: [31, 12, 45, 27, ...] -> session 1 repeated on song 31, etc.
    # Many sessions are needed because one session alone is just luck;
    # 10,000 of them show the true pattern (Monte Carlo simulation)
    return np.array([first_repeat(playlist_size) for _ in range(total_sessions)])


def print_stats(results):
    """Prints mean, median, mode, std and range of the results. Returns mean and median."""
    mean = np.mean(results)  # average play number of the first repeat
    median = np.median(results)  # middle value when the results are sorted
    # Mode: bincount makes a list where position v holds how many times v appeared,
    # so argmax (the position of the biggest count) is the most common value
    mode = np.argmax(np.bincount(results))
    # Std: how spread out the results are. ddof=1 divides by (N-1) instead of N
    # because our 10,000 sessions are a sample, not every possible session
    # (Bessel's correction)
    std = np.std(results, ddof=1)

    print(f"Mean   = {mean:.2f}")
    print(f"Median = {median:.1f}")
    print(f"Mode   = {mode}")
    print(f"Std    = {std:.2f}")
    print(f"Range  = {results.min()} to {results.max()}")
    # If the mean is bigger than the median, a long right tail is pulling it up
    # (a few lucky sessions with no repeat for 70+ songs raise the average,
    # but they do not move the middle value much)
    if mean > median:
        print("Mean > Median, so the distribution is right skewed")
    else:
        print("Mean <= Median, so the distribution is not right skewed")
    return mean, median


def plot_curve(ax, results, k50, k90):
    """Plots the simulated vs exact chance of a repeat after each song"""
    k = np.arange(1, max_plays + 1)  # songs played: 1, 2, ..., max_plays
    # Simulated: fraction of sessions that already had a repeat by song i
    # results <= i gives True/False for every session; the mean of True/False
    # is the fraction of True values (True counts as 1, False as 0)
    sim = [np.mean(results <= i) for i in k]
    # Exact: the formula from repeat_probability
    exact = [repeat_probability(playlist_size, i) for i in k]

    # If the simulation is correct, the black steps sit right on the blue line
    ax.step(k, sim, color="black", linewidth=3, label="Estimated")
    ax.plot(k, exact, color="blue", label="Actual")
    # Red lines and arrows mark where the chance reaches 50% and 90%
    for x, p in [(k50, 0.5), (k90, 0.9)]:
        ax.vlines(x, 0, p, color="red")
        ax.annotate(f"{p:.0%} by song {x}", xy=(x, p), xytext=(x + 5, p - 0.1),
                    arrowprops={"facecolor": "black"})
    ax.set_title(f"Shuffle Paradox ({playlist_size} songs, {total_sessions:,} sessions)")
    ax.set_xlabel("Songs Played")
    ax.set_ylabel("Probability of a Repeat")
    ax.legend(loc="upper left")


def plot_histogram(ax, results, mean, median):
    """Plots a histogram of when the first repeat happened, with mean and median lines"""
    # One bar per song number (bin edges at 0.5, 1.5, 2.5, ... so bars are centered)
    bins = np.arange(1, results.max() + 2) - 0.5
    ax.hist(results, bins=bins, color="blue", edgecolor="white")
    # Vertical lines show the median and mean (mean > median means right skewed)
    ax.axvline(median, color="red", label=f"Median = {median:.1f}")
    ax.axvline(mean, color="black", label=f"Mean = {mean:.1f}")
    ax.set_title(f"First Repeat ({total_sessions:,} sessions)")
    ax.set_xlabel("Song Number of First Repeat")
    ax.set_ylabel("Number of Sessions")
    ax.set_xlim(0, 90)
    ax.legend(loc="upper right")


def plot_calendars(ax):
    """Plots the exact repeat curve for each calendar size on the same axes"""
    k = np.arange(1, max_plays + 1)  # number of picks: 1, 2, ..., max_plays
    # Logic: the same formula works for anything with n equally likely values
    # Smaller n means fewer values to pick from, so a repeat comes sooner
    for name, n in calendars.items():
        y = [repeat_probability(n, i) for i in k]  # exact curve for this calendar
        k50 = plays_needed(n, 0.5)  # where this calendar reaches 50%
        ax.plot(k, y, label=f"{name}: 50% at {k50}")
    ax.axhline(0.5, color="gray", linestyle="--")  # dashed line at 50%
    ax.set_title("Same Paradox, Different Calendars")
    ax.set_xlabel("Number of Picks")
    ax.set_ylabel("Probability of a Repeat")
    ax.legend(loc="lower right")


def plot_magic(ax, sizes, ratios):
    """Plots (plays for 50%) / sqrt(n) for many sizes, approaching sqrt(2 ln 2)"""
    ax.plot(sizes, ratios, marker="o", label="Plays for 50% / sqrt(n)")
    # Red dashed line is the magic number the ratios approach as n gets bigger
    ax.axhline(magic, color="red", linestyle="--", label=f"sqrt(2 ln 2) = {magic:.4f}")
    ax.set_xscale("log")  # log scale so sizes from 10 to 100,000 all fit
    ax.set_title("Magic Number")
    ax.set_xlabel("Playlist Size (log scale)")
    ax.set_ylabel("Plays for 50% / sqrt(n)")
    ax.legend(loc="upper right")


def main():
    chart_dir.mkdir(exist_ok=True)  # create the charts folder if it is missing

    # Exact answer: how many songs until a 50% and 90% chance of a repeat
    k50 = plays_needed(playlist_size, 0.5)
    k90 = plays_needed(playlist_size, 0.9)
    print(f"Playlist size = {playlist_size}")
    print(f"50% chance of a repeat by song {k50}")
    print(f"90% chance of a repeat by song {k90}")
    print()

    # Simulation: run all the sessions and print their statistics
    results = run_sessions()
    print(f"First repeat over {total_sessions:,} sessions")
    mean, median = print_stats(results)
    print()

    # Different calendars: same paradox for different numbers of values
    print("Calendar\t\tSize\t50% at\t90% at")
    for name, n in calendars.items():
        print(f"{name}\t{n}\t{plays_needed(n, 0.5)}\t{plays_needed(n, 0.9)}")
    print()

    # Magic number: (plays for 50%) / sqrt(n) gets closer to sqrt(2 ln 2) as n grows
    # Logic: for large n, chance of no repeat is about e^(-k^2 / 2n)
    # Set it equal to 1/2 and solve for k:  k = sqrt(2 ln 2) * sqrt(n)
    # So dividing k by sqrt(n) should leave only the magic number (~1.1774)
    sizes = [10, 20, 50, 100, 200, 365, 500, 1000, 2000, 5000, 10_000, 50_000, 100_000]
    ratios = []
    print("Size\t50% at\tRatio")
    for n in sizes:
        k = plays_needed(n, 0.5)
        ratios.append(k / np.sqrt(n))
        # ratios[-1] is the ratio just added (index -1 means the last item)
        print(f"{n}\t{k}\t{ratios[-1]:.4f}")
    print(f"Magic number = sqrt(2 ln 2) = {magic:.4f}")
    print()

    # Live demo table: chance two people in a room share a birthday or last 2 phone digits
    # People are the "picks", and the 365 days or 100 digit pairs are the "songs"
    print("People\tBirthday\tPhone")
    for people in range(10, 41, 5):
        print(f"{people}\t{repeat_probability(365, people):.1%}"
              f"\t\t{repeat_probability(100, people):.1%}")
    # Optional room size from the command line, e.g. python shuffle_paradox.py 24
    # sys.argv[0] is the file name, so sys.argv[1] is the first number typed after it
    if len(sys.argv) > 1:
        people = int(sys.argv[1])
        print(f"Room of {people}: birthday {repeat_probability(365, people):.1%}, "
              f"phone {repeat_probability(100, people):.1%}")

    # All 4 plots in one window (2 rows x 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), num=Path(__file__).name)
    plot_curve(axs[0, 0], results, k50, k90)  # top left
    plot_histogram(axs[0, 1], results, mean, median)  # top right
    plot_calendars(axs[1, 0])  # bottom left
    plot_magic(axs[1, 1], sizes, ratios)  # bottom right
    fig.tight_layout()  # keep titles and labels from overlapping
    fig.savefig(chart_dir / "shuffle_paradox_all.png")
    plt.show()


if __name__ == "__main__":
    main()
