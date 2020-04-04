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
    491, 537, 605, 649, 724,
]
LAST_PERFECT_EXP_DATAPOINT = 143
DAYS_TO_PREDICT = 3


def linear(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
    return a * exp(b * x) + c

def fit(func, x, y, start_date=START_DATE):
    popt, pcov = curve_fit(func, x, y)
    # "full": current (fitted) and future
    full_x = arange(len(y) + DAYS_TO_PREDICT)
    full_x_as_dates = [start_date + timedelta(days=int(i)) for i in full_x]
    return popt, pcov, full_x, full_x_as_dates


#myFmt = mdates.DateFormatter('%d')
days = mdates.DayLocator()

# actual cases
cases = array(CONFIRMED_CASES)
x = arange(len(cases))
xdates = [START_DATE + timedelta(days=int(i)) for i in x]

# best-effort to fit an exponent, using all data
popt, _, future_x, future_xdates = fit(exponential, x, cases)

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
)

# linear fit after initial exponential run
linear_x_offset = len_perf_cases - 1
linear_start_date = START_DATE + timedelta(days=linear_x_offset)
linear_cases = array(CONFIRMED_CASES[linear_x_offset:])
linear_x = arange(len(linear_cases))
linear_popt, _, linear_future_x, linear_future_xdates = fit(
    linear,
    linear_x,
    linear_cases,
    start_date=linear_start_date
)

# exponential fit across same data
exp2_popt, _, exp2_future_x, exp2_future_xdates = fit(
    exponential,
    linear_x,
    linear_cases,
    start_date=linear_start_date
)

# figure()
f, ax = subplots()
#ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.format_xdata = mdates.DateFormatter('%m-%d')
f.autofmt_xdate()

plot(xdates, cases,
     '-b', label="Confirmed cases")
# plot(future_xdates, exponential(future_x, *popt),
#      'x--r', label="Exponential fit across all data")
plot(perf_future_xdates, exponential(perf_future_x, *perf_popt),
     'x--', color='grey', label="Exponential fit during initial run")
plot(linear_future_xdates, linear(linear_future_x, *linear_popt),
     'x--', color='#33aabb', label="Linear fit after initial run")
plot(exp2_future_xdates, exponential(exp2_future_x, *exp2_popt),
     'x--', color='#ee66aa', label="Exponential fit after initial run")

title("SARS-CoV-2 case prediction in Lithuania")
xlabel("Date")
ylabel("Number of cases")
legend()
grid(True)
show()
