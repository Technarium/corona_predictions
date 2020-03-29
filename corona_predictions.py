#!/usr/bin/env python

from datetime import date, timedelta
from matplotlib.pyplot import plot, subplots, grid, figure, show, title, xlabel, ylabel, legend
from numpy import array, arange, exp
from scipy.optimize import curve_fit

import matplotlib.dates as mdates

START_DATE = date(2020, 2, 28)
CONFIRMED_CASES = [
        # initial stretch where an exponential function gives good fit: Feb 28 to Mar 22
        # NOTE: quarantine declared on Mar 16 (day that ended with 16 cases)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 6, 7, 12, 16, 26, 34, 48, 69, 99, 143,
        # spread breaks exponential trajectory on Mar 23 (one week after quarantine start)
        179, 209, 274, 299, 358, 394
]
DAYS_TO_PREDICT = 3

def exponential(x, a, b, c):
	return a * exp(b * x) + c


#myFmt = mdates.DateFormatter('%d')
days = mdates.DayLocator()

# actual cases
cases = array(CONFIRMED_CASES)
x = arange(len(cases))
xdates = [START_DATE + timedelta(days=int(i)) for i in x]

# best-effort to fit an exponent, using all data
popt, pcov = curve_fit(exponential, x, cases)
future_x = arange(len(cases) + DAYS_TO_PREDICT)
future_xdates = [START_DATE + timedelta(days=int(i)) for i in future_x]

# figure()
f, ax = subplots()
#ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.format_xdata = mdates.DateFormatter('%m-%d')
f.autofmt_xdate()

plot(xdates, cases, '-b', label = "Confirmed cases")
plot(future_xdates, exponential(future_x, *popt), 'x-r', label="Prediction (exponential)")

title("SARS-CoV-2 case prediction in Lithuania")
xlabel("Date")
ylabel("Number of cases")
legend()
grid(True)
show()
