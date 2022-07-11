/**
 * Standard boring biquad filter.
 */

#ifndef BIQUAD_H_INCLUDED
#define BIQUAD_H_INCLUDED

/** Filter state struct */
struct biquad;

/**
 * Create a filter struct.
 *
 * @param fs the sampling frequency the filter is intended to work on
 */
struct biquad* bq_init (double fs);

/**
 * Free filter struct.
 *
 * @param bq the struct to free
 */
void bq_free (struct biquad *bq);

/**
 * Process a sample of a signal with the filter.
 *
 * @param bq the filter to apply
 * @param in the input sample
 */
double bq_process_sample (struct biquad *bq, double in);

/**
 * Process a signal buffer.
 *
 * @param bq the filter to apply
 * @param in the input signal
 * @param out the output signal
 * @param length signal length
 */
void bq_process_buffer (struct biquad *bq,
                        const double *in,
                        double *out,
                        int length);

/**
 * Reset the internal state of a filter.
 *
 * @param bq the filter reset
 */
void bq_reset_state (struct biquad *bq);

/**
 * Set the filter coefficients to apply a bandpass filter.
 *
 * @param bq the filter to alter
 * @param fc the centre frequency of the band
 * @param Q the Q factor
 */
void bq_make_bandpass (struct biquad *bq,
                       double fc,
                       double Q);

#endif /* BIQUAD_H_INCLUDED */
