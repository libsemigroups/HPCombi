#!/usr/bin/env python3

import os
import re
import statistics as stats
import sys

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
    global color;
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
            color=color[:len(Y)],
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

def check_filename(xml_fnam):
    if len(xml_fnam.split(".")) < 2:
        raise ValueError(
            f"expected filename of form x.xml found {xml_fnam}"
        )


from sys import argv

args = sys.argv[1:]

for x in args:
    check_filename(x)
    # TODO more arg checks
for x in args:
    add_plot(x)
xml_fnam = args[0]
png_fnam = "".join(xml_fnam.split(".")[:-1]) + ".png"
print("Writing {} . . .".format(png_fnam))
plt.savefig(png_fnam, format="png", dpi=300)
sys.exit(0)
