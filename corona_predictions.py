#!/usr/bin/env python

from datetime import date, timedelta
from matplotlib.pyplot import plot, subplots, grid, figure, show, title, xlabel, ylabel, legend
from numpy import array, arange, exp
from scipy.optimize import curve_fit

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d')

days = mdates.DayLocator()

def func(x, a, b , c):
	return a * exp(b * x) + c



cases = array([1, 2, 3, 3, 3, 6, 7, 12, 16, 26, 34, 48, 69, 99])
x = arange(len(cases))
future_x=arange(len(cases)+3) 

popt, pcov = curve_fit(func, x, cases) 

xd=[date(2020,3,8+i) for i in x]
future_xd = [date(2020,3,8+i) for i in future_x]

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
