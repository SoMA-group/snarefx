#include "so_lattice.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*****************************************************
 * Filter functionality
 *****************************************************/
enum filter_type
{
    SOL_GAIN,
    SOL_PEAK
};

static void set_gain_coeffs (double gain,
                             double *k,
                             double *v)
{
    v [0] = db2a (gain);

    for (int i = 0; i < 2; ++i)
    {
        k [i] = 0.0;
        v [i + 1] = 0.0;
    }
}

static void set_peak_coeffs (double fs,
                             double fc,
                             double Q,
                             double gain,
                             double *k,
                             double *v)
{
  double A = pow (10.0, gain / 40.0);
  double w0 = 2.0 * M_PI * fc / fs;
  double cw = cos (w0);
  double sw = sin (w0);
  double alpha = sw / (2.0 * Q);

  double norm = 1.0 + alpha / A;

  double a1 = (-2.0 * cw) / norm;
  double b0 = (1.0 + alpha * A) / norm;

  k [1] = (A - alpha) / (A + alpha);
  k [0] = -cw;

  v [2] = (1.0 - alpha * A) / norm;
  v [1] = (1.0 - v [2]) * a1;
  v [0] = b0 - k [0] * a1 + (a1 * a1 / (k [1] + 1.0) - k [1]) * v [2];
}

static double process_sample (double in,
                              double *k,
                              double *v,
                              double *g)
{
    double f1 = in - k [1] * g [1];
    double f0 = f1 - k [0] * g [0];

    g [2] = k [1] * f1 + g [1];
    g [1] = k [0] * f0 + g [0];
    g [0] = f0;

    double out = 0.0;

    for (int i = 0; i < 3; ++i)
    {
        out += v [i] * g [i];
    }

    return out;
}

static void reset_state (double *g)
{
    for (int i = 0; i < 3; ++i)
    {
        g [i] = 0.0;
    }
}

/*****************************************************
 * Standard filter
 *****************************************************/
struct so_lattice
{
    /* config */
    enum filter_type type;
    double fs;
    double fc;
    double Q;
    double gain;

    /* filter coefficients */
    double k [2];
    double v [3];

    /* filter states */
    double g [3];
};

struct so_lattice* sol_init (double fs)
{
    struct so_lattice *sol = calloc (1, sizeof (*sol));

    if (!sol)
        return NULL;

    /* init config */
    sol->type = SOL_GAIN;
    sol->fs = fs;

    /* init coeffs to unity gain */
    sol_make_gain_stage (sol, 0.0);

    return sol;
}

void sol_free (struct so_lattice *sol)
{
    free (sol);
}

double sol_process_sample (struct so_lattice *sol, double in)
{
    return process_sample (in,
                           sol->k,
                           sol->v,
                           sol->g);
}

void sol_process_buffer (struct so_lattice *sol,
                         const double *in,
                         double *out,
                         int length)
{
    for (int i = 0; i < length; ++i)
    {
        out [i] = sol_process_sample (sol, in [i]);
    }
}

void sol_reset_state (struct so_lattice *sol)
{
    reset_state (sol->g);
}

void sol_make_gain_stage (struct so_lattice *sol,
                          double gain)
{
    /* update params */
    sol->type = SOL_GAIN;
    sol->fc = 0.0;
    sol->Q = 0.0;
    sol->gain = gain;

    /* set coefficients */
    set_gain_coeffs (gain,
                     sol->k,
                     sol->v);
}

void sol_make_peak (struct so_lattice *sol,
                    double fc,
                    double Q,
                    double gain)
{
    /* update params */
    sol->type = SOL_PEAK;
    sol->fc = fc;
    sol->Q = Q;
    sol->gain = gain;

    /* set coefficients */
    set_peak_coeffs (sol->fs,
                     fc,
                     Q,
                     gain,
                     sol->k,
                     sol->v);
}

void sol_set_gain (struct so_lattice *sol, double gain)
{
    /* set gain */
    sol->gain = gain;

    /* update filter coefficients */
    switch (sol->type)
    {
        case SOL_GAIN:
            set_gain_coeffs (sol->gain,
                             sol->k,
                             sol->v);
            break;
        case SOL_PEAK:
            set_peak_coeffs (sol->fs,
                             sol->fc,
                             sol->Q,
                             sol->gain,
                             sol->k,
                             sol->v);
            break;
    }
}

/*****************************************************
 * Dynamic filter
 *****************************************************/
struct dynamic_so_lattice
{
    /* config */
    enum filter_type type;
    double fs;
    double fc;
    double Q;
    double gain;
    enum dynamics_mode mode;

    /* filter coefficients */
    double k [2];
    double v [3];

    /* target coefficients */
    double gain_target;
    double k_target [2];
    double v_target [3];

    /* coefficients errors */
    /*double k_error [2];
    double v_error [3];*/

    /* filter states */
    double g [3];

    /* ballistics */
    double attack, attack_coeff;
    double release, release_coeff;
    double *control_coeff;
};

/*****************************************************
 * Ballistics update functions
 *****************************************************/
static void compute_attack (struct dynamic_so_lattice *dsol)
{
    dsol->attack_coeff = one_pole_coeff (dsol->fs, dsol->attack);
}

static void compute_release (struct dynamic_so_lattice *dsol)
{
    dsol->release_coeff = one_pole_coeff (dsol->fs, dsol->release);
}

static void compute_ballistics (struct dynamic_so_lattice *dsol)
{
    compute_attack (dsol);
    compute_release (dsol);
}

/*****************************************************
 * Struct management
 *****************************************************/
struct dynamic_so_lattice* dsol_init (double fs,
                                      enum dynamics_mode mode,
                                      double attack,
                                      double release)
{
    struct dynamic_so_lattice *dsol = calloc (1, sizeof (*dsol));

    if (!dsol)
        return NULL;

    /* init config */
    dsol->type = SOL_GAIN;
    dsol->fs = fs;
    dsol->mode = mode;

    /* init coeffs to unity gain */
    dsol_make_gain_stage (dsol, 0.0);

    /* init ballistics */
    dsol->attack = attack;
    dsol->release = release;
    compute_ballistics (dsol);
    dsol->control_coeff = &dsol->attack_coeff;

    return dsol;
}

void dsol_free (struct dynamic_so_lattice *dsol)
{
    free (dsol);
}

/*****************************************************
 * Filter coefficient interpolation
 *****************************************************/
static void update_coeffs (struct dynamic_so_lattice *dsol)
{
    /* Keep a rough track of the current gain of the filter.
     * This won't be very accurate but will probably do the job.
     */
    dsol->gain += *(dsol->control_coeff) * (dsol->gain_target - dsol->gain);

    /* lattice coefficients */
    for (int i = 0; i < 2; ++i)
    {
        dsol->k [i] += *(dsol->control_coeff) * (dsol->k_target [i] - dsol->k [i]);
    }

    /* ladder coefficients */
    for (int i = 0; i < 3; ++i)
    {
        dsol->v [i] += *(dsol->control_coeff) * (dsol->v_target [i] - dsol->v [i]);
    }
}

/*
 * When working with floats there was some serious rounding problems going on here.
 * This kahan summing / fraction saving type approach worked quite well. 
 * Switching over to doubles improved things as well. There are probably still problems,
 * but I can't be bothered to chase that goose right now. 
 */
/*static void update_coeffs (struct dynamic_so_lattice *dsol)
{
    dsol->gain += *(dsol->control_coeff) * (dsol->gain_target - dsol->gain);

    for (int i = 0; i < 2; ++i)
    {
        double inc = *(dsol->control_coeff) * (dsol->k_target [i] - dsol->k [i]) + dsol->k_error [i];
        double old = dsol->k [i];

        dsol->k [i] += inc;
        dsol->k_error [i] = inc - (dsol->k [i] - old);
    }

    for (int i = 0; i < 3; ++i)
    {
        double inc = *(dsol->control_coeff) * (dsol->v_target [i] - dsol->v [i]) + dsol->v_error [i];
        double old = dsol->v [i];

        dsol->v [i] += inc;
        dsol->v_error [i] = inc - (dsol->v [i] - old);
    }
}*/

/*****************************************************
 * Signal processing
 *****************************************************/
double dsol_process_sample (struct dynamic_so_lattice *dsol, double in)
{
    /* apply ballistics */
    update_coeffs (dsol);

    /* filter input */
    return process_sample (in,
                           dsol->k,
                           dsol->v,
                           dsol->g);
}

void dsol_process_buffer (struct dynamic_so_lattice *dsol,
                          const double *in,
                          double *out,
                          int length)
{
    for (int i = 0; i < length; ++i)
    {
        out [i] = dsol_process_sample (dsol, in [i]);
    }
}

static void skip_ballistics (struct dynamic_so_lattice *dsol)
{
    dsol->gain = dsol->gain_target;
    memcpy (dsol->k, dsol->k_target, sizeof (dsol->k));
    memcpy (dsol->v, dsol->v_target, sizeof (dsol->v));
}

void dsol_reset_state (struct dynamic_so_lattice *dsol)
{
    reset_state (dsol->g);
    dsol_set_gain (dsol, 0.0, 1);
}

/*****************************************************
 * Filter settings
 *****************************************************/
void dsol_set_mode (struct dynamic_so_lattice *dsol,
                    enum dynamics_mode mode)
{
    /* do nothing if mode isn't changing */
    if (dsol->mode == mode)
        return;

    /* update mode and swap control coefficient */
    dsol->mode = mode;
    dsol->control_coeff = dsol->control_coeff == &dsol->attack_coeff ? &dsol->release_coeff : &dsol->attack_coeff;
}

void dsol_make_gain_stage (struct dynamic_so_lattice *dsol,
                           double gain)
{
    /* update params */
    dsol->type = SOL_GAIN;
    dsol->fc = 0.0;
    dsol->Q = 0.0;
    dsol->gain_target = gain;

    /* set coefficients */
    set_gain_coeffs (gain,
                     dsol->k_target,
                     dsol->v_target);
    skip_ballistics (dsol);
}

void dsol_make_peak (struct dynamic_so_lattice *dsol,
                     double fc,
                     double Q,
                     double gain)
{
    /* update params */
    dsol->type = SOL_PEAK;
    dsol->fc = fc;
    dsol->Q = Q;
    dsol->gain_target = gain;

    /* set coefficients */
    set_peak_coeffs (dsol->fs,
                     fc,
                     Q,
                     gain,
                     dsol->k_target,
                     dsol->v_target);
    skip_ballistics (dsol);
}

void dsol_set_gain (struct dynamic_so_lattice *dsol, double gain, int immediate)
{
    /* work out whether we are in attack or release */
    if (gain < dsol->gain)
        dsol->control_coeff = dsol->mode == COMPRESSOR ? &dsol->attack_coeff : &dsol->release_coeff;
    else
        dsol->control_coeff = dsol->mode == COMPRESSOR ? &dsol->release_coeff : &dsol->attack_coeff;

    /* set gain */
    dsol->gain_target = gain;

    /* update filter coefficients */
    switch (dsol->type)
    {
        case SOL_GAIN:
            set_gain_coeffs (dsol->gain_target,
                             dsol->k_target,
                             dsol->v_target);
            break;
        case SOL_PEAK:
            set_peak_coeffs (dsol->fs,
                             dsol->fc,
                             dsol->Q,
                             dsol->gain_target,
                             dsol->k_target,
                             dsol->v_target);
            break;
    }

    if (immediate != 0)
        skip_ballistics (dsol);
}

void dsol_set_attack (struct dynamic_so_lattice *dsol, double attack)
{
    dsol->attack = attack;
    compute_attack (dsol);
}

void dsol_set_release (struct dynamic_so_lattice *dsol, double release)
{
    dsol->release = release;
    compute_release (dsol);
}
