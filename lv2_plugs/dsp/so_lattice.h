#ifndef DYNAMIC_SO_LATTICE_H_INCLUDED
#define DYNAMIC_SO_LATTICE_H_INCLUDED

/**
 * A second order lattice-ladder filter.
 */
struct so_lattice;

/**
 * Create a filter struct.
 *
 * @param fs the sampling frequency the filter is intended to work on
 */
struct so_lattice* sol_init (double fs);

/**
 * Free filter struct.
 *
 * @param sol the struct to free
 */
void sol_free (struct so_lattice *sol);

/**
 * Process a sample of a signal with the filter.
 *
 * @param sol the filter to apply
 * @param in the input sample
 */
double sol_process_sample (struct so_lattice *sol, double in);

/**
 * Process a signal buffer.
 *
 * @param sol the filter to apply
 * @param in the input signal
 * @param out the output signal
 * @param length signal length
 */
void sol_process_buffer (struct so_lattice *sol,
                         const double *in,
                         double *out,
                         int length);

/**
 * Reset the internal state of a filter.
 *
 * @param sol the filter reset
 */
void sol_reset_state (struct so_lattice *sol);

/**
 * Set the filter coefficients so the filter is just a gain stage.
 *
 * @param sol the filter to alter
 * @param gain the gain
 */
void sol_make_gain_stage (struct so_lattice *sol,
                          double gain);

/**
 * Set the filter coefficients to apply a peaking filter (band shelf).
 *
 * @param sol the filter to alter
 * @param fc the centre frequency of the peak
 * @param Q the Q factor
 * @param gain the gain of the peak in dB
 */
void sol_make_peak (struct so_lattice *sol,
                    double fc,
                    double Q,
                    double gain);

/**
 * Set the gain of the filter.
 * 
 * @param sol the filter to alter
 * @param gain the new gain in dB
 */
void sol_set_gain (struct so_lattice *sol, double gain);

/**
 * A second order IIR filter with gain attack and release ballistics.
 * 
 * The filter uses a lattice-ladder structure so that the coefficients can be interpolated between settings without
 * everything blowing up. At least that's the idea.
 */
struct dynamic_so_lattice;

/**
 * Dynamics modes the filter can operate in.
 *
 * These determine what is considered attack and release of the system.
 * In compressor mode a decrease in gain is considered attack while an increase in gain is a release.
 * For expander mode the opposite is true.
 */
enum dynamics_mode
{
    COMPRESSOR,
    EXPANDER
};

/**
 * Create a filter struct.
 *
 * @param fs the sampling frequency the filter is intended to work on
 * @param mode the mode the filter should start in
 * @param attack the initial attack time in ms
 * @param release the initial release time in ms
 */
struct dynamic_so_lattice* dsol_init (double fs,
                                      enum dynamics_mode mode,
                                      double attack,
                                      double release);

/**
 * Free filter struct.
 *
 * @param dsol the struct to free
 */
void dsol_free (struct dynamic_so_lattice *dsol);

/**
 * Process a sample of a signal with the filter.
 *
 * @param dsol the filter to apply
 * @param in the input sample
 */
double dsol_process_sample (struct dynamic_so_lattice *dsol, double in);

/**
 * Process a signal buffer.
 *
 * @param dsol the filter to apply
 * @param in the input signal
 * @param out the output signal
 * @param length signal length
 */
void dsol_process_buffer (struct dynamic_so_lattice *dsol,
                          const double *in,
                          double *out,
                          int length);

/**
 * Reset the internal state of a filter and set the gain back to unity.
 *
 * @param dsol the filter reset
 */
void dsol_reset_state (struct dynamic_so_lattice *dsol);

/**
 * Change the mode of a filter.
 *
 * @param dsol the filter to alter
 * @param mode the new mode
 */
void dsol_set_mode (struct dynamic_so_lattice *dsol,
                    enum dynamics_mode mode);

/**
 * Set the filter coefficients so the filter is just a gain stage.
 *
 * Changes to the filter are made immediately, no attack or release.
 *
 * @param dsol the filter to alter
 * @param gain the gain
 */
void dsol_make_gain_stage (struct dynamic_so_lattice *dsol,
                           double gain);

/**
 * Set the filter coefficients to apply a peaking filter (band shelf).
 *
 * Changes to the filter are made immediately, no attack or release.
 *
 * @param dsol the filter to alter
 * @param fc the centre frequency of the peak
 * @param Q the Q factor
 * @param gain the gain of the peak in dB
 */
void dsol_make_peak (struct dynamic_so_lattice *dsol,
                     double fc,
                     double Q,
                     double gain);

/**
 * Set the gain of the filter.
 * 
 * @param dsol the filter to alter
 * @param gain the target gain in dB
 * @param immediate a boolean value: if 0 the filter will approach the target gain according to the attack or release time,
 *                                   if any other value the gain change will take effect immediately
 */
void dsol_set_gain (struct dynamic_so_lattice *dsol, double gain, int immediate);

/**
 * Set the attack time of the filter.
 * 
 * @param dsol the filter to alter
 * @param attack the new attack time in ms
 */
void dsol_set_attack (struct dynamic_so_lattice *dsol, double attack);

/**
 * Set the release time of the filter.
 * 
 * @param dsol the filter to alter
 * @param release the new release time in ms
 */
void dsol_set_release (struct dynamic_so_lattice *dsol, double release);

#endif /* DYNAMIC_SO_LATTICE_H_INCLUDED */
