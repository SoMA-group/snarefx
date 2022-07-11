#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

/**
 * Convert amplitude to decibels.
 *
 * @param a the amplitude to convert
 */
double a2db (double a);

/**
 * Convert decibels to amplitude.
 *
 * @param db the decible value to convert
 */
double db2a (double db);

/*
 * Return the coefficient, g, for a one pole filter with the given decay time, t, measured in milliseconds.
 *
 * @param fs the sampling frequency
 * @param t the decay time in ms
 *
 * Use g as:
 *
 * y[n] = y[n-1] + g(x[n] - y[n-1])
 */
double one_pole_coeff (double fs, double t);

#endif /* UTIL_H_INCLUDED */
