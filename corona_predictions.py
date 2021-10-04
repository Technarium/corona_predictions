#!/usr/bin/env python

from datetime import date, timedelta
from matplotlib.pyplot import (
    plot,
    subplots,
    grid,
    figure,
    show,
    title,
    xlabel,
    ylabel,
    legend,
)
from numpy import array, arange, exp, log
from scipy.optimize import curve_fit

import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# NOTE: first case was detected on 2020-02-27: announced the night of
# 27/28; recorded on 2020-02-28
START_DATE = date(2020, 2, 24)
# confirmed on day-end prior to Apr 6 (Mon); "number at morning briefing" afterwards
DAILY_CASES = [
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 3, 1, 5,
    4, 10, 8, 14, 21, 30, 44,
    36, 30, 65, 25, 59, 36, 66,
    31, 46, 68, 44, 75, 40, 32,
    47, 37, 32, 43, 44, 27, 27,
    9, 8, 21, 37, 21, 90, 59,
    28, 24, 20, 28, 12, 16, 12,
    11, -105, 31, 10, 14, 7, 4,
    9, 4, 5, 5, 3, 8, 35,
    6, 6, 14, 6, 12, 11, 7,
    6, 15, 15, 16, 11, 12, 7,
    12, 4, 8, 9, 6, 8, 5,
    3, 4, 2, 3, 7, 11, 9,
    6, 7, 6, 19, 4, 7, 5,
    5, 3, 2, 6, 8, 3, 3,
    3, 2, 1, 2, 2, 5, 2,
    1, 1, 1, 7, 3, 3, 5,
    5, 3, 10, 3, 4, 4, 4,
    5, 1, 7, 20, 6, 7, 17,
    15, 2, 2, 9, 26, 15, 7,
    11, 8, 16, 19, 13, 18, 17,
    10, 17, 10, 24, 23, 37, 21,
    13, 18, 26, 21, 22, 34, 30,
    20, 38, 22, 32, 36, 30, 41,
    38, 21, 32, 36, 48, 29, 35,
    32, 23, 29, 20, 26, 36, 43,
    17, 31, 32, 36, 44, 53, 39,
    51, 11, 45, 62, 61, 99, 80,
    70, 45, 73, 138, 114, 111, 90,
    105, 88, 115, 91, 172, 125, 104,
    100, 81, 117, 142, 133, 205, 160,
    125, 118, 139, 255, 281, 228, 252,
    205, 202, 311, 424, 441, 474, 606,
    765, 413, 776, 950, 735, 1001, 895,
    837, 897, 639, 999, 1656, 1972, 1980,
    1056, 1086, 1421, 1550, 2066, 1531, 1372,
    1130, 965, 1934, 1682, 2265, 1983, 2307,
    1179, 1167, 2289, 2077, 2339, 2121, 1976,
    1136, 1187, 2109, 2450, 2514, 2848, 2219,
    1386, 1388, 3128, 3330, 3067, 3303, 2849,
    1919, 1432, 3418, 3159, 3181, 3219, 2927,
    2130, 1871, 3737, 3799, 2691, 2350, 1668,
    1773, 1921, 3934, 2360, 3322, 1493, 1239,
    1361, 1510, 2717, 2331, 1993, 1641, 1490,
    775, 902, 1694, 1371, 1149, 1122, 836,
    476, 717, 1232, 1274, 1032, 1001, 773,
    541, 686, 1278, 953, 860, 718, 803,
    355, 546, 760, 721, 654, 676, 491,
    270, 387, 569, 487, 468, 510, 375,
    203, 300, 376, 365, 663, 561, 463,
    263, 482, 631, 538, 624, 626, 356,
    243, 426, 503, 517, 434, 450, 390,
    213, 462, 460, 596, 355, 589, 397,
    249, 628, 570, 552, 534, 575, 321,
    318, 855, 768, 769, 816, 724, 412,
    418, 831, 853, 869, 810, 985, 833,
    600, 686, 1116, 1067, 1155, 995, 839,
    597, 1088, 1284, 1168, 1088, 1193, 726,
    592, 1138, 1113, 1344, 1278, 1212, 729,
    514, 1031, 1585, 1327, 1306, 1335, 949,
    655, 1043, 1235, 1293, 1244, 1486, 883,
    786, 1226, 1230, 1136, 1125, 1141, 740,
    504, 685, 911, 865, 664, 822, 459,
    269, 537, 523, 562, 503, 522, 285,
    181, 375, 415, 339, 365, 334, 168,
    84, 215, 241, 212, 155, 221, 117,
    66, 129, 126, 104, 72, 75, 50,
    28, 58, 54, 61, 18, 44, 17,
    15, 35, 40, 25, 30, 35, 20,
    18, 40, 33, 34, 63, 74, 25,
    22, 86, 80, 102, 82, 89, 75,
    60, 101, 176, 193, 233, 231, 189,
    96, 211, 337, 380, 346, 397, 184,
    212, 394, 503, 462, 528, 496, 367,
    287, 517, 576, 599, 548, 585, 434,
    356, 560, 576, 506, 537, 580, 407,
    314, 600, 649, 703, 545, 634, 396,
    353, 646, 695, 807, 819, 833, 581,
    346, 705, 854, 925, 922, 1061, 810,
    478, 889, 1298, 1300, 1197, 1351, 848,
    745, 1155, 1402, 1486, 1464, 1609, 1251,
    1043, 1225, 1847, 1806, 1970, 2066, 1576,
    1114,
]
# previously, this calculation was done manually; an artifact of better days
cumulative = 0
CUMULATIVE_CASES = []
for day_cases in DAILY_CASES:
    cumulative += day_cases
    CUMULATIVE_CASES.append(cumulative)

LAST_PERFECT_EXP_DATAPOINT = 143
SECOND_WAVE_START_DAY = 135 # somewhat arbitrary
DAYS_TO_PREDICT = 7

def linear(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
    #print(len(x), a, b, c)
    return a * exp(b * x) + c

def logarithmic(x, a, b, c):
    return a * log(b * x + 1e-10) + c

def sigmoidal(x, y0, x0, c, k):
    return c / (1 + exp((x - x0) / k)) + y0

def fit(func, x, y, start_date=START_DATE, days_to_predict=DAYS_TO_PREDICT):
    popt, pcov = curve_fit(func, x, y, maxfev=10000)
    # "full": current (fitted) and future
    full_x = arange(len(y) + days_to_predict)
    full_x_as_dates = [start_date + timedelta(days=int(i)) for i in full_x]
    return popt, pcov, full_x, full_x_as_dates


#myFmt = mdates.DateFormatter('%d')
days = mdates.DayLocator()

# actual cases
all_cases = array(CUMULATIVE_CASES)
all_x = arange(len(all_cases))
all_xdates = [START_DATE + timedelta(days=int(i)) for i in all_x]

# best-effort to fit an exponential function, using all data
all_popt, _, all_future_x, all_future_xdates = fit(exponential, all_x, all_cases)
all_future_ycases = exponential(all_future_x, *all_popt)

# sigmoidal fit on first wave data
sigmoidal_x_offset = 14 # skip first weeks of low cases
sigmoidal_start_date = START_DATE + timedelta(days=sigmoidal_x_offset)
sigmoidal_cases = array(CUMULATIVE_CASES[sigmoidal_x_offset:sigmoidal_x_offset+85])
sigmoidal_x = arange(len(sigmoidal_cases))
sig_popt, sig_pcov, sig_future_x, sig_future_xdates = fit(
    sigmoidal,
    sigmoidal_x,
    sigmoidal_cases,
    start_date=sigmoidal_start_date,
)
sig_future_ycases = sigmoidal(sig_future_x, *sig_popt)

# limited exponent fitting when it was still perfect:
# first, take a slice of all the cases...
len_perf_cases = CUMULATIVE_CASES.index(LAST_PERFECT_EXP_DATAPOINT) + 1
perf_cases = array(CUMULATIVE_CASES[:len_perf_cases])
perf_x = arange(len(perf_cases))
# ... then fit a curve to that
perf_popt, _, perf_future_x, perf_future_xdates = fit(
    exponential,
    perf_x,
    perf_cases,
    days_to_predict=5,
)
perf_future_ycases = exponential(perf_future_x, *perf_popt)

# linear fit after initial exponential run
linear_x_offset = len_perf_cases - 1
linear_start_date = START_DATE + timedelta(days=linear_x_offset)
linear_cases = array(CUMULATIVE_CASES[linear_x_offset:])
linear_x = arange(len(linear_cases))
linear_popt, _, linear_future_x, linear_future_xdates = fit(
    linear,
    linear_x,
    linear_cases,
    start_date=linear_start_date,
)
linear_future_ycases = linear(linear_future_x, *linear_popt)

# exponential fit across same data
exp_popt, _, exp_future_x, exp_future_xdates = fit(
    exponential,
    linear_x,
    linear_cases,
    start_date=linear_start_date,
)
exp_future_ycases = exponential(exp_future_x, *exp_popt)

# logarithmic fit across same data
log_popt, _, log_future_x, log_future_xdates = fit(
    logarithmic,
    linear_x,
    linear_cases,
    start_date=linear_start_date,
)
log_future_ycases = logarithmic(log_future_x, *log_popt)

# logarithmic fit on first wave data
log1w_x_offset = len_perf_cases - 1
log1w_start_date = START_DATE + timedelta(days=log1w_x_offset)
log1w_cases = array(CUMULATIVE_CASES[log1w_x_offset:log1w_x_offset+125])
log1w_x = arange(len(log1w_cases))
log1w_popt, log1w_pcov, log1w_future_x, log1w_future_xdates = fit(
    logarithmic,
    log1w_x,
    log1w_cases,
    start_date=log1w_start_date,
)
log1w_future_ycases = logarithmic(log1w_future_x, *log1w_popt)

# exponential fit + difference between exponential and linear
diffe_future_xdates = exp_future_xdates
diffe_future_ycases = [2 * ex - li
                       for ex, li in zip(exp_future_ycases,
                                         linear_future_ycases)]

# linear fit - difference between exponential and linear
diffl_future_xdates = exp_future_xdates
diffl_future_ycases = [2 * li - ex
                       for ex, li in zip(exp_future_ycases,
                                         linear_future_ycases)]

# exponential fit on second wave start
exp2_x_offset = SECOND_WAVE_START_DAY
exp2_start_date = START_DATE + timedelta(days=exp2_x_offset)
exp2_cases = array(CUMULATIVE_CASES[exp2_x_offset:exp2_x_offset+44])
exp2_x = arange(len(exp2_cases))
exp2_popt, _, exp2_future_x, exp2_future_xdates = fit(
    exponential,
    exp2_x,
    exp2_cases,
    start_date=exp2_start_date,
    days_to_predict=140,
)
exp2_future_ycases = exponential(exp2_future_x, *exp2_popt)

# exponential fit over last 42 days
expl42_x_offset = len(CUMULATIVE_CASES)-42
expl42_start_date = START_DATE + timedelta(days=expl42_x_offset)
expl42_cases = array(CUMULATIVE_CASES[expl42_x_offset:])
expl42_x = arange(len(expl42_cases))
expl42_popt, _, expl42_future_x, expl42_future_xdates = fit(
    exponential,
    expl42_x,
    expl42_cases,
    start_date=expl42_start_date,
)
expl42_future_ycases = exponential(expl42_future_x, *expl42_popt)

# exponential fit over last 15 days
expl15_x_offset = len(CUMULATIVE_CASES)-15
expl15_start_date = START_DATE + timedelta(days=expl15_x_offset)
expl15_cases = array(CUMULATIVE_CASES[expl15_x_offset:])
expl15_x = arange(len(expl15_cases))
expl15_popt, _, expl15_future_x, expl15_future_xdates = fit(
    exponential,
    expl15_x,
    expl15_cases,
    start_date=expl15_start_date,
)
expl15_future_ycases = exponential(expl15_future_x, *expl15_popt)


# figure()
f, ax = subplots()

ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(n=7))
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
#ax.minorticks_on()

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# plot fits first, so they get layered on the bottom
# plot(all_future_xdates, all_future_ycases,
#      'x--r', label="Exponential fit across all data")
plot(perf_future_xdates, perf_future_ycases,
     'x--', color='grey', label="Exponential fit during initial run")
plot(sig_future_xdates, sig_future_ycases,
     'x--', color='#446644', label="Sigmoidal fit across first wave data")
plot(log1w_future_xdates, log1w_future_ycases,
     'x--', color='#66cc44', label="Logarithmic fit across first wave data")
# plot(exp_future_xdates, exp_future_ycases,
#      'x--', color='#66ccee', label="Exponential fit after initial run")
# plot(log_future_xdates, log_future_ycases,
#      'x--', color='#66cc44', label="Logarithmic fit after initial run")
# plot(linear_future_xdates, linear_future_ycases,
#      'x--', color='#ee66aa', label="Linear fit after initial run")
plot(exp2_future_xdates, exp2_future_ycases,
     'x--', color='#ee9955', label="Exponential fit after second wave start")
plot(expl42_future_xdates, expl42_future_ycases,
     'x--', color='#aa3311', label="Exponential fit over last 42 days")
plot(expl15_future_xdates, expl15_future_ycases,
     'x--', color='#dd2200', label="Exponential fit over last 15 days")

# supplemental: neither raw data nor actual fits
# last_n = DAYS_TO_PREDICT + 3
# plot(diffe_future_xdates[-last_n:], diffe_future_ycases[-last_n:],
#      '--', color='#66ccee', alpha=0.5,
#      label="(Difference between exp and lin, added)",
# )
# plot(diffl_future_xdates[-last_n:], diffl_future_ycases[-last_n:],
#      '--', color='#ee66aa',
#      alpha=0.5, label="(Difference between exp and lin, removed)",
# )

# plot actual data on top of everything, so it doesn't get "hidden"
plot(all_xdates, all_cases, '-b', alpha=0.7, label="Confirmed cases")

title("SARS-CoV-2 case predictions in Lithuania")
xlabel("Date")
ylabel("Number of cases")
legend()
grid(which='major')
grid(which='minor', linestyle='--')
show()
