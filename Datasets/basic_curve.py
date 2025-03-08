"""
Virtual Power Plant Forecast Data Generation
"""


import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from docplex.mp.context import Context
from pylab import mpl
from matplotlib.font_manager import FontProperties

mpl.rcParams['font.serif'] = ['Times New Roman']  # Set default font
simsun = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue
mpl.rcParams['font.size'] = 12  # Set global font size
mpl.rcParams['axes.titlesize'] = 15  # Set title font size
mpl.rcParams['axes.labelsize'] = 15  # Set axis label font size
mpl.rcParams['xtick.labelsize'] = 15  # Set x-axis tick font size
mpl.rcParams['ytick.labelsize'] = 15  # Set y-axis tick font size
mpl.rcParams['legend.fontsize'] = 8  # Set legend font size
mpl.rcParams['image.interpolation'] = 'nearest'  # Set interpolation style


# TODO: Reserve an interface for average replacement of predictions


price = [16.07, 14.96, 14.28, 13.26, 13.21, 14.08, 15, 16.93, 18.6, 20.41, 22.11, 24.5, 26.57, 28.07,
         29.97, 30.66, 35.5, 35.58, 29.54, 26.73, 23.91, 22.16, 19.93, 17.93]

price = [x * 2 for x in price]

# Active power load demand data
power_load = [
    3.84, 3.92, 3.76, 3.44, 3.28, 3.36, 3.6, 4.64, 5, 6, 5.76,
    5.84, 5.76, 5.6, 5.44, 5.2, 4.96, 4.8, 4.72, 4.72, 4.8, 4.96,
    5.2, 4.8
]

# Wind energy data
wind_power = [
    1.0, 1.0, 1.0, 1.0, 0.98, 0.95, 0.9, 0.78, 0.75, 0.75, 0.77,
    0.6, 0.6, 0.62, 0.35, 0.35, 0.55, 0.5, 0.45, 0.45, 0.65,
    0.8, 0.9, 1.0
]

# Photovoltaic data
pv = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06825511, 0.53766546, 0.89338147, 1.5766065, 1.60779783,
    2.0, 1.93636582, 1.60038508, 1.31850782, 0.69622142, 0.46180505, 0.17915764, 0.01482551,
    0.0, 0.0, 0.0, 0.0, 0.0
]

wind_power = [1.00, 1.20, 0.98, 0.95, 0.98, 0.95, 0.90, 0.78, 0.75, 0.78, 0.80, 0.60, 0.35, 0.30, 0.55, 0.50, 0.45, 0.50, 0.60, 0.70, 0.80, 0.85, 0.95, 1.00]

temperature = [
    16, 15, 15, 14, 13, 15, 17, 18, 19, 22, 23, 24, 25, 26, 26,
    26, 26, 24, 23, 22, 21, 21, 20, 19
]


def f_draw():
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    colors = ['#FF9999', '#66B2FF', '#99FF98', '#FFCC99']

    axs[0, 0].plot(price, marker='o', color=colors[0], linewidth=2)
    axs[0, 0].set_title('Energy Price Forecast', fontsize=12, fontproperties=simsun)
    axs[0, 0].set_xlabel('Time (hours)', fontsize=10, fontproperties=simsun)
    axs[0, 0].set_ylabel('Price', fontsize=10, fontproperties=simsun)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    axs[0, 1].plot(power_load, marker='s', color=colors[1], linewidth=2)
    axs[0, 1].set_title('Active Power Load Demand', fontsize=12, fontproperties=simsun)
    axs[0, 1].set_xlabel('Time (hours)', fontsize=10, fontproperties=simsun)
    axs[0, 1].set_ylabel('Load (MW)', fontsize=10, fontproperties=simsun)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    axs[1, 0].plot(wind_power, marker='^', color=colors[2], linewidth=2)
    axs[1, 0].set_title('Wind Power Generation', fontsize=12, fontproperties=simsun)
    axs[1, 0].set_xlabel('Time (hours)', fontsize=10, fontproperties=simsun)
    axs[1, 0].set_ylabel('Generation (MW)', fontsize=10, fontproperties=simsun)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    axs[1, 1].plot(pv, marker='D', color=colors[3], linewidth=2)
    axs[1, 1].set_title('Solar Power Generation', fontsize=12, fontproperties=simsun)
    axs[1, 1].set_xlabel('Time (hours)', fontsize=10, fontproperties=simsun)
    axs[1, 1].set_ylabel('Generation (MW)', fontsize=10, fontproperties=simsun)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # plt.savefig('forecast.jpg', dpi=350, bbox_inches='tight')
    plt.show()
    plt.close(fig)


f_draw()