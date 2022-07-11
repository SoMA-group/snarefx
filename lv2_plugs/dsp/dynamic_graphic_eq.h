/**
 * A dynamic graphic equaliser.
 */

#ifndef DYNAMIC_GRAPHIC_EQ_H_INCLUDED
#define DYNAMIC_GRAPHIC_EQ_H_INCLUDED

/** Dynamic EQ struct */
struct dynamic_graphic_eq;

/**
 * Create a dynamic EQ struct.
 *
 * @param fs the sampling frequency the EQ is intended to work at
 * @param n_bands the number of EQ bands
 * @param start_freq the centre frequency of the first band
 * @param freq_ratio the multiplication factor between successive centre frequencies
 * @param frame_size the frame size used for envelope detection
 * @param hop_size the hop size used for envelope detection
 */
struct dynamic_graphic_eq* dgeq_init (double fs,
                                      int n_bands,
                                      double start_freq,
                                      double freq_ratio,
                                      int frame_size,
                                      int hop_size);

/**
 * Free dynamic EQ struct.
 *
 * @param dgeq the struct to free
 */
void dgeq_free (struct dynamic_graphic_eq *dgeq);

/**
 * Process a sample of a signal with the EQ.
 *
 * @param dgeq the EQ to apply
 * @param in the input sample
 */
double dgeq_process_sample (struct dynamic_graphic_eq *dgeq, double in);

/**
 * Process a signal buffer.
 *
 * @param dgeq the EQ to apply
 * @param in the input signal
 * @param out the output signal
 * @param length signal length
 */
void dgeq_process_buffer (struct dynamic_graphic_eq *dgeq,
                          const double *in,
                          double *out,
                          int length);

/**
 * Reset the internal state of all elements of an EQ
 *
 * @param dgeq the EQ reset
 */
void dgeq_reset_state (struct dynamic_graphic_eq *dgeq);

/**
 * Set the compression threshold for a particular EQ band.
 *
 * @param dgeq the EQ to alter
 * @param band the band to alter (must be between 0 and n_bands - 1)
 * @param threshold the new threshold in dB
 */
void dgeq_set_band_threshold (struct dynamic_graphic_eq *dgeq,
                              int band,
                              double threshold);

/**
 * Set the compression ratio for a particular EQ band.
 *
 * @param dgeq the EQ to alter
 * @param band the band to alter (must be between 0 and n_bands - 1)
 * @param ratio the new ratio (ratio:1)
 */
void dgeq_set_band_ratio (struct dynamic_graphic_eq *dgeq,
                          int band,
                          double ratio);

/**
 * Set the compression knee for a particular EQ band.
 *
 * @param dgeq the EQ to alter
 * @param band the band to alter (must be between 0 and n_bands - 1)
 * @param knee the new knee in dB
 */
void dgeq_set_band_knee (struct dynamic_graphic_eq *dgeq,
                         int band,
                         double knee);

/**
 * Set the compression attack time for a particular EQ band.
 *
 * @param dgeq the EQ to alter
 * @param band the band to alter (must be between 0 and n_bands - 1)
 * @param attack the new attack time in ms
 */
void dgeq_set_band_attack (struct dynamic_graphic_eq *dgeq,
                           int band,
                           double attack);

/**
 * Set the compression release time for a particular EQ band.
 *
 * @param dgeq the EQ to alter
 * @param band the band to alter (must be between 0 and n_bands - 1)
 * @param release the new release time in ms
 */
void dgeq_set_band_release (struct dynamic_graphic_eq *dgeq,
                            int band,
                            double release);

/**
 * Set the compression makeup gain for a particular EQ band.
 *
 * @param dgeq the EQ to alter
 * @param band the band to alter (must be between 0 and n_bands - 1)
 * @param makeup_gain the new makeup gain in dB
 */
void dgeq_set_band_makeup_gain (struct dynamic_graphic_eq *dgeq,
                                int band,
                                double makeup_gain);

/**
 * Set the output gain of the EQ
 *
 * @param dgeq the EQ to alter
 * @param output_gain the new output gain
 */
void dgeq_set_output_gain (struct dynamic_graphic_eq *dgeq,
                           double output_gain);

#endif /* DYNAMIC_GRAPHIC_EQ_H_INCLUDED */
