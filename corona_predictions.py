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
from numpy import array, arange, exp
from scipy.optimize import curve_fit

import matplotlib.dates as mdates

# NOTE: first case was detected on 2020-02-27/28: announced that night
START_DATE = date(2020, 3, 8)
# confirmed on day-end
CONFIRMED_CASES = [
    # initial stretch where an exponential function gives good fit:
    # Feb 28 to Mar 22; however, for now the first 9 days are omitted
    # so the graph isn't too stretched
    #1, 1, 1,
    #1, 1, 1, 1, 1, 1,
    1,
    2, 3, 3, 3, 6, 7, 12,
    # quarantine declared on Mar 16 (Mon, day that ended with 16 cases)
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
    1449,
]
LAST_PERFECT_EXP_DATAPOINT = 143
DAYS_TO_PREDICT = 7


def linear(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
    return a * exp(b * x) + c

def sigmoid(x, y0, x0, c, k):
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
cases = array(CONFIRMED_CASES)
x = arange(len(cases))
xdates = [START_DATE + timedelta(days=int(i)) for i in x]

# best-effort to fit an exponential function, using all data
all_popt, _, all_future_x, all_future_xdates = fit(exponential, x, cases)

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

# sigmoidal fit across same data
sig_popt, _, sig_future_x, sig_future_xdates = fit(
    sigmoid,
    linear_x,
    linear_cases,
    start_date=linear_start_date,
)
sig_future_ycases = sigmoid(sig_future_x, *sig_popt)

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
#ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.format_xdata = mdates.DateFormatter('%m-%d')
ax.minorticks_on()

f.autofmt_xdate()

# plot fits first, so they get layered on the bottom
# plot(all_future_xdates, exponential(all_future_x, *all_popt),
#      'x--r', label="Exponential fit across all data")
plot(perf_future_xdates, perf_future_ycases,
     'x--', color='grey', label="Exponential fit during initial run")
plot(sig_future_xdates, sig_future_ycases,
     'x--', color='#66ee66', label="Sigmoidal fit after initial run")
plot(exp_future_xdates, exp_future_ycases,
     'x--', color='#66ccee', label="Exponential fit after initial run")
plot(linear_future_xdates, linear_future_ycases,
     'x--', color='#ee66aa', label="Linear fit after initial run")
# supplemental: neither raw data nor actual fits
last_n = DAYS_TO_PREDICT + 3
plot(diffe_future_xdates[-last_n:], diffe_future_ycases[-last_n:],
     '--', color='#66ccee', alpha=0.5,
     label="(Difference between the two, added)",
)
plot(diffl_future_xdates[-last_n:], diffl_future_ycases[-last_n:],
     '--', color='#ee66aa',
     alpha=0.5, label="(Difference between the two, removed)",
)
# plot actual data on top of everything, so it doesn't get "hidden"
plot(xdates, cases, '-b', alpha=0.7, label="Confirmed cases")

title("SARS-CoV-2 case predictions in Lithuania")
xlabel("Date")
ylabel("Number of cases")
legend()
grid(which='major')
grid(which='minor', linestyle='--')
show()
