#include "util.h"
#include <math.h>

double a2db (double a)
{
    if (a <= 0.0)
        return -200.0;
    else
        return 20.0 * log10 (a);
}

double db2a (double db)
{
    if (db <= -200.0)
        return 0.0;
    else
        return pow (10.0, db * 0.05);
}

double one_pole_coeff (double fs, double t)
{
    if (t <= 0.0)
        return 1.0;
    else
        return 1.0 - exp (-1000.0 / (fs * t));
}
