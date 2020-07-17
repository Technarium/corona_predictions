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
# confirmed on day-end
CONFIRMED_CASES = [
    0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    2, 3, 3, 3, 6, 7, 12,
    # quarantine declared on Mar 16 (Mon, day that ended with 16 cases);
    # initial stretch where an exponential function gives good
    # fit is up to Mar 22 (day with 143 cases)
    16, 26, 34, 48, 69, 99, 143,
    # no longer exponential on Mar 23 (one week after quarantine start)
    179, 209, 274, 299, 358, 394, 460,
    491, 537, 605, 649, 724, 764, 796,
    # starting week of Apr 6 (Mon), data is no longer entered as "the
    # number of cases at day-end", but instead "the number of cases at
    # morning briefing"; this is the same data most other sources will
    # present, but they'll have it shifted a day back
    843, 880, 912, 955, 999, 1026, 1053,
    1062, 1070, 1091, 1128, 1149, 1239, 1298,
    1326, 1350, 1370, 1398, 1410, 1426, 1438,
    1449, 1344, 1375, 1385, 1399, 1406, 1410,
    1419, 1423, 1428, 1433, 1436, 1444, 1479,
    1485, 1491, 1505, 1511, 1523, 1534, 1541,
    1547, 1562, 1577, 1593, 1604, 1616, 1623,
    1635, 1639, 1647, 1656, 1662, 1670, 1675,
    1678, 1682, 1684, 1687, 1694, 1705, 1714,
    1720, 1727, 1733, 1752, 1756, 1763, 1768,
    1773, 1776, 1778, 1784, 1792, 1795, 1798,
    1801, 1803, 1804, 1806, 1808, 1813, 1815,
    1816, 1817, 1818, 1825, 1828, 1831, 1836,
    1841, 1844, 1854, 1857, 1861, 1865, 1869,
    1874, 1875, 1882, 1902, 1908,
]
LAST_PERFECT_EXP_DATAPOINT = 143
DAYS_TO_PREDICT = 7


def linear(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
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
all_cases = array(CONFIRMED_CASES)
all_x = arange(len(all_cases))
all_xdates = [START_DATE + timedelta(days=int(i)) for i in all_x]

# best-effort to fit an exponential function, using all data
all_popt, _, all_future_x, all_future_xdates = fit(exponential, all_x, all_cases)
all_future_ycases = exponential(all_future_x, *all_popt)

# sigmoidal fit across all data
sig_popt, sig_pcov, sig_future_x, sig_future_xdates = fit(
    sigmoidal,
    all_x,
    all_cases,
)
sig_future_ycases = sigmoidal(sig_future_x, *sig_popt)

# limited exponent fitting when it was still perfect:
# first, take a slice of all the cases...
len_perf_cases = CONFIRMED_CASES.index(LAST_PERFECT_EXP_DATAPOINT) + 1
perf_cases = array(CONFIRMED_CASES[:len_perf_cases])
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
linear_cases = array(CONFIRMED_CASES[linear_x_offset:])
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
plot(sig_future_xdates, sig_future_ycases,
     'x--', color='#446644', label="Sigmoidal fit across all data")
plot(perf_future_xdates, perf_future_ycases,
     'x--', color='grey', label="Exponential fit during initial run")
# plot(exp_future_xdates, exp_future_ycases,
#      'x--', color='#66ccee', label="Exponential fit after initial run")
plot(log_future_xdates, log_future_ycases,
     'x--', color='#66cc44', label="Logarithmic fit after initial run")
# plot(linear_future_xdates, linear_future_ycases,
#      'x--', color='#ee66aa', label="Linear fit after initial run")

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
