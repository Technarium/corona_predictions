# corona_predictions

[CSSE@JHU data](https://github.com/CSSEGISandData/2019-nCoV) follows
case information at time and interval relevant to them. This produces
skewed results for us when trying to fit prediction curves, since the
data is not aligned to day-end.

This is a simple script that uses local data instead. It plots
known-confirmed cases, and tries to fit a curve to that (currently an
exponential curve).
