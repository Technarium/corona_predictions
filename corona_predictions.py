#!/usr/bin/env python

from datetime import date, timedelta
from matplotlib.pyplot import plot, subplots, grid, figure, show, title, xlabel, ylabel, legend
from numpy import array, arange, exp
from scipy.optimize import curve_fit

import matplotlib.dates as mdates

CONFIRMED_CASES = [1, 2, 3, 3, 3, 6, 7, 12, 16, 26, 34, 48, 69, 99, 143, 179, 209]
DAYS_TO_PREDICT = 3

def func(x, a, b, c):
	return a * exp(b * x) + c


#myFmt = mdates.DateFormatter('%d')
days = mdates.DayLocator()

cases = array(CONFIRMED_CASES)
x = arange(len(cases))

future_x = arange(len(cases) + DAYS_TO_PREDICT)

popt, pcov = curve_fit(func, x, cases) 

xd = [date(2020, 3, 8+i) for i in x]
future_xd = [date(2020, 3, 8) + timedelta(days=int(i)) for i in future_x]

# figure()
f, ax = subplots()
#ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.format_xdata = mdates.DateFormatter('%m-%d')
f.autofmt_xdate()

plot(future_xd, func(future_x, *popt), 'x-r', label="Prediction")
plot(xd, cases, '-b', label = "Confirmed cases")
title("SARS-CoV-2 case prediction in Lithuania")
xlabel("Date")
ylabel("Number of cases")
legend()
grid(True)
show()
