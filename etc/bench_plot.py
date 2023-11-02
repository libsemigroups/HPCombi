#!/usr/bin/env python3

import os
import re
import statistics as stats
import sys
from math import isqrt

import matplotlib
import numpy as np
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt

# This file should be from libsemigroups/etc

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

color = [
    (238 / 255, 20 / 255, 135 / 255),
    (0 / 255, 221 / 255, 164 / 255),
    (86 / 255, 151 / 255, 209 / 255),
    (249 / 255, 185 / 255, 131 / 255),
    (150 / 255, 114 / 255, 196 / 255),
]

# Filenames should be: name.something.xml -> name.png


def normalize_xml(xml_fnam):
    with open(xml_fnam, "r") as f:
        xml = f.read()
        xml = re.sub("&lt;", "<", xml)
    with open(xml_fnam, "w") as f:
        f.write(xml)


def xml_stdout_get(xml, name):
    try:
        return xml.find("StdOut").find(name)["value"]
    except (KeyError, TypeError, AttributeError):
        return None


def time_unit(Y):
    time_units = ("microseconds", "milliseconds", "seconds")
    index = 0

    while all(y > 1000 for y in Y) and index < len(time_units):
        index += 1
        Y = [y / 1000 for y in Y]
    return time_units[index], Y


def add_plot(xml_fnam, num_bars=4):
    global color
    current_bar = 0
    Y = []
    Y_for_comparison = None
    labels = []

    xml = BeautifulSoup(open(xml_fnam, "r"), "xml")
    total_cols = 0
    xticks_label = []
    xticks_pos = []
    for x, test_case in enumerate(xml.find_all("TestCase")):
        results = test_case.find_all("BenchmarkResults")
        Y = (
            np.array([float(x.find("mean")["value"]) for x in results]) / 1
        )  # times in nanoseconds
        X = np.arange(total_cols + 1, total_cols + len(Y) + 1, 1)
        xticks_label.append(("\n" * (x % 2)) + test_case["name"])
        xticks_pos.append(total_cols + 1 + (len(Y) / 2) - 0.5)
        bars = plt.bar(
            X,
            Y,
            1,
            align="center",
            color=color[: len(Y)],
        )
        total_cols += len(Y) + 1
    plt.yscale("log", nonpositive="clip")
    plt.ylabel("Time in ns")
    plt.xticks(xticks_pos, xticks_label)
    # plt.legend(loc="upper right")

    # print(Y)
    # width = 1

    # plt.axhline(
    #     stats.mean(Y),
    #     color=color[current_bar],
    #     linestyle="--",
    #     lw=1,
    #     xmin=0.01,
    #     xmax=0.99,
    # )

    # current_bar += 1
    # if current_bar == num_bars - 1:
    #     Ys = zip(*sorted(zip(*Ys)))
    #     for i, Y in enumerate(Ys):
    #         X = np.arange(i, num_bars * len(Y), num_bars)
    #         bars = plt.bar(
    #             X,
    #             Y,
    #             width,
    #             align="center",
    #             color=color[i],
    #             label=labels[i],
    #         )
    #     plt.xticks(
    #         np.arange(1, num_bars * (len(X) + 1), num_bars * 20),
    #         np.arange(0, len(X) + num_bars - 1, 20),
    #     )
    #     plt.xlabel("Test case")
    #     plt.ylabel("Time (relative)")
    #     plt.legend(loc="upper left")


def determine_subplot_layout(nr_plots: int) -> tuple[int, int]:
    """Determine the number of rows and columns from number of plots."""
    nr_plot_rows = isqrt(nr_plots)
    nr_plot_cols = nr_plot_rows
    if nr_plot_rows * nr_plot_cols < nr_plots:
        nr_plot_cols += 1
    while nr_plot_rows * nr_plot_cols < nr_plots:
        nr_plot_rows += 1
    return nr_plot_rows, nr_plot_cols


def process_result(result_soup) -> tuple[str, float]:
    """Extract data from a single xml result entry.

    Returns
    -------
    result_name: str
        The test case name
    result_time: float
        The test case time in nanoseconds
    """
    result_name = result_soup["name"]
    if "name" not in result_soup.attrs:
        raise ValueError(
            f"Malformed benchmark file, result record does not contain 'name': {result_soup}"
        )
    result_mean_soup = result_soup.find("mean")
    if result_mean_soup is None:
        raise ValueError(
            f"Malformed benchmark file, result record does not contain 'mean': {result_soup}"
        )
    if "value" not in result_mean_soup.attrs:
        raise ValueError(
            f"Malformed benchmark file, result 'mean' record does not contain 'value': {result_mean_soup}"
        )
    result_time = float(result_mean_soup["value"]) / 1  # time in nanoseconds
    return result_name, result_time


def make_ax(ax, test_case_soup):
    if "name" not in test_case_soup.attrs:
        raise ValueError(
            f"Malformed benchmark file, test_case record does not contain 'name': {test_case_soup}"
        )
    results = test_case_soup.find_all("BenchmarkResults")
    result_names, result_times = zip(*map(process_result, reversed(results)))
    bars = ax.barh(
        result_names,
        result_times,
        align="center",
        color=color[: len(result_times)],
    )
    test_name = test_case_soup["name"]
    ax.set_title(f'Benchmark "{test_name}" runtime')
    ax.set_xlabel(f"ns")
    return ax


def make_fig(benchmark_soup):
    test_cases = benchmark_soup.find_all("TestCase")
    nr_plots = len(test_cases)
    nr_plot_rows, nr_plot_cols = determine_subplot_layout(nr_plots)
    fig, axs = plt.subplots(nr_plot_rows, nr_plot_cols)
    for test_case_soup, ax in zip(test_cases, axs.flat):
        ax = make_ax(ax, test_case_soup)
    return fig


def check_filename(xml_fnam):
    if len(xml_fnam.split(".")) < 2:
        raise ValueError(f"expected filename of form x.xml found {xml_fnam}")


if __name__ == "__main__":
    args = sys.argv[1:]

    for x in args:
        check_filename(x)
        # TODO more arg checks

    for x in args:
        with open(x, "r") as in_file:
            xml_text = in_file.read()
        soup = BeautifulSoup(xml_text, "xml")
        fig = make_fig(soup)
        plt.show()

    xml_fnam = args[0]
    png_fnam = "".join(xml_fnam.split(".")[:-1]) + ".png"
    print("Writing {} . . .".format(png_fnam))
    plt.savefig(png_fnam, format="png", dpi=300)
    sys.exit(0)
