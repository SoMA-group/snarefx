#include "biquad.h"
#include <stdlib.h>
#include <math.h>

/*****************************************************
 * Filter state
 *****************************************************/
struct biquad
{
    /* config */
    double fs;

    /* filter coefficients */
    double b [3];
    double a [2];

    /* filter states */
    double s [2];
};

/*****************************************************
 * Struct management
 *****************************************************/
struct biquad* bq_init (double fs)
{
    struct biquad *bq = calloc (1, sizeof (*bq));

    if (!bq)
        return NULL;

    /* init config */
    bq->fs = fs;

    /* init coeffs to unity gain */
    bq->b [0] = 1.0;

    return bq;
}

void bq_free (struct biquad *bq)
{
    free (bq);
}

/*****************************************************
 * Signal processing
 *****************************************************/
double bq_process_sample (struct biquad *bq, double in)
{
    double out = bq->b [0] * in + bq->s [0];
    bq->s [0] = bq->b [1] * in - bq->a [0] * out + bq->s [1];
    bq->s [1] = bq->b [2] * in - bq->a [1] * out;

    return out;
}

void bq_process_buffer (struct biquad *bq,
                        const double *in,
                        double *out,
                        int length)
{
    for (int i = 0; i < length; ++i)
    {
        out [i] = bq_process_sample (bq, in [i]);
    }
}

void bq_reset_state (struct biquad *bq)
{
    bq->s [0] = 0.0;
    bq->s [1] = 0.0;
}

/*****************************************************
 * Filter settings
 *****************************************************/
void bq_make_bandpass (struct biquad *bq,
                       double fc,
                       double Q)
{
  double w0 = 2.0 * M_PI * fc / bq->fs;
  double cw = cos (w0);
  double sw = sin (w0);
  double alpha = sw / (2.0 * Q);

  double norm = 1.0 + alpha;

  bq->b [0] = alpha / norm;
  bq->b [1] = 0.0;
  bq->b [2] = -bq->b [0];

  bq->a [0] = -2.0 * cw / norm;
  bq->a [1] = (1 - alpha) / norm;
}
