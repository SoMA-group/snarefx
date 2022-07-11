#include "dynamic_graphic_eq.h"
#include "frame_envelope_follower.h"
#include "biquad.h"
#include "util.h"
#include "so_lattice.h"
#include <stdlib.h>
#include <math.h>

struct dynamic_graphic_eq
{
    /* config */
    double fs;
    int n_bands;
    double start_freq, freq_ratio;
    int frame_size, hop_size;

    /* dsp elements */
    struct biquad **bpfs;
    struct frame_envelope_follower **efs;
    struct dynamic_so_lattice **bands;
    struct so_lattice **mgs;

    /* parameters */
    double *thresholds;
    double *ratios;
    double *knees;
    double output_gain;
};

static int init_dsp_elements (struct dynamic_graphic_eq *dgeq)
{
    const double initial_attack = 10.0;
    const double initial_release = 100.0;

    for (int i = 0; i < dgeq->n_bands; ++i)
    {
        struct biquad *bpf = bq_init (dgeq->fs);
        struct frame_envelope_follower *fef = fef_init (PEAK_DETECTION,
                                                        dgeq->frame_size,
                                                        dgeq->hop_size);
        struct dynamic_so_lattice *dsol = dsol_init (dgeq->fs,
                                                     COMPRESSOR,
                                                     initial_attack,
                                                     initial_release);
        struct so_lattice *sol = sol_init (dgeq->fs);

        if (!(bpf && fef && dsol && sol))
            return -1;

        dgeq->bpfs [i] = bpf;
        dgeq->efs [i] = fef;
        dgeq->bands [i] = dsol;
        dgeq->mgs [i] = sol;
    }

    return 0;
}

static void prepare_filters (struct dynamic_graphic_eq *dgeq)
{
    double f = dgeq->start_freq;
    double k = dgeq->freq_ratio;
    double Q = sqrt (k) / (k - 1);

    for (int i = 0; i < dgeq->n_bands; ++i)
    {
        bq_make_bandpass (dgeq->bpfs [i], f, Q);
        dsol_make_peak (dgeq->bands [i], f, Q, 0.0);
        sol_make_peak (dgeq->mgs [i], f, Q, 0.0);
        f *= k;
    }
}

static void set_initial_parameters (struct dynamic_graphic_eq *dgeq)
{
    const double initial_threshold = 0.0;
    const double initial_ratio = 1.0;
    const double initial_knee = 0.0;
    const double initial_output_gain = 1.0;

    for (int i = 0; i < dgeq->n_bands; ++i)
    {
        dgeq->thresholds [i] = initial_threshold;
        dgeq->ratios [i] = initial_ratio;
        dgeq->knees [i] = initial_knee;
    }

    dgeq->output_gain = initial_output_gain;
}

struct dynamic_graphic_eq* dgeq_init (double fs,
                                      int n_bands,
                                      double start_freq,
                                      double freq_ratio,
                                      int frame_size,
                                      int hop_size)
{
    struct dynamic_graphic_eq *dgeq = calloc (1, sizeof (*dgeq));

    if (!dgeq)
        goto failed;

    dgeq->fs = fs;
    dgeq->n_bands = n_bands;
    dgeq->start_freq = start_freq;
    dgeq->freq_ratio = freq_ratio;
    dgeq->frame_size = frame_size;
    dgeq->hop_size = hop_size;

    /* allocate dsp element arrays */
    dgeq->bpfs = calloc (n_bands, sizeof (*(dgeq->bpfs)));
    dgeq->efs = calloc (n_bands, sizeof (*(dgeq->efs)));
    dgeq->bands = calloc (n_bands, sizeof (*(dgeq->bands)));
    dgeq->mgs = calloc (n_bands, sizeof (*(dgeq->mgs)));

    if (!(dgeq->bpfs && dgeq->efs && dgeq->bands && dgeq->mgs))
        goto failed;

    /* initialise dsp elements */
    if (init_dsp_elements (dgeq) < 0)
        goto failed;

    prepare_filters (dgeq);

    /* allocate band parameters */
    dgeq->thresholds = malloc (n_bands * sizeof (*(dgeq->thresholds)));
    dgeq->ratios = malloc (n_bands * sizeof (*(dgeq->ratios)));
    dgeq->knees = malloc (n_bands * sizeof (*(dgeq->knees)));

    if (!(dgeq->thresholds && dgeq->ratios && dgeq->knees))
        goto failed;

    /* set default compression parameters */
    set_initial_parameters (dgeq);

    return dgeq;

failed:
    dgeq_free (dgeq);
    return NULL;
}

void dgeq_free (struct dynamic_graphic_eq *dgeq)
{
    if (dgeq)
    {
        free (dgeq->knees);
        free (dgeq->ratios);
        free (dgeq->thresholds);

        for (int i = 0; i < dgeq->n_bands; ++i)
        {
            sol_free (dgeq->mgs [i]);
            dsol_free (dgeq->bands [i]);
            fef_free (dgeq->efs [i]);
            bq_free (dgeq->bpfs [i]);
        }

        free (dgeq->mgs);
        free (dgeq->bands);
        free (dgeq->efs);
        free (dgeq->bpfs);
        free (dgeq);
    }
}

static double compute_gain (double in,
                            double threshold,
                            double ratio,
                            double knee)
{
    double over = in - threshold;
    double knee_on_2 = knee / 2.0;
    double out = in;

    if (over > knee_on_2)
    {
        out = threshold + over / ratio;
    }
    else if (over > -knee_on_2)
    {
        double quad = over + knee_on_2;
        out = in + (1.0 / ratio - 1.0) * quad * quad / (2.0 * knee);
    }

    return out - in;
}

static void band_sidechain_process_sample (struct dynamic_graphic_eq *dgeq,
                                           int band,
                                           double in)
{
    struct biquad *bpf = dgeq->bpfs [band];
    struct frame_envelope_follower *ef = dgeq->efs [band];
    double filtered = bq_process_sample (bpf, in);
    double level = 0.0;

    if (fef_process_sample (ef, filtered, &level))
    {
        double t = dgeq->thresholds [band];
        double r = dgeq->ratios [band];
        double k = dgeq->knees [band];
        double g = compute_gain (a2db (level), t, r, k);

        dsol_set_gain (dgeq->bands [band], g, 0);
    }
}

double dgeq_process_sample (struct dynamic_graphic_eq *dgeq, double in)
{
    double out = in;

    for (int i = 0; i < dgeq->n_bands; ++i)
    {
        band_sidechain_process_sample (dgeq, i, in);
        out = dsol_process_sample (dgeq->bands [i], out);
        out = sol_process_sample (dgeq->mgs [i], out);
    }

    return dgeq->output_gain * out;
}

void dgeq_process_buffer (struct dynamic_graphic_eq *dgeq,
                          const double *in,
                          double *out,
                          int length)
{
    for (int i = 0; i < length; ++i)
    {
        out [i] = dgeq_process_sample (dgeq, in [i]);
    }
}

void dgeq_reset_state (struct dynamic_graphic_eq *dgeq)
{
    for (int i = 0; i < dgeq->n_bands; ++i)
    {
        bq_reset_state (dgeq->bpfs [i]);
        fef_reset_state (dgeq->efs [i]);
        dsol_reset_state (dgeq->bands [i]);
        sol_reset_state (dgeq->mgs [i]);
    }
}

void dgeq_set_band_threshold (struct dynamic_graphic_eq *dgeq,
                              int band,
                              double threshold)
{
    dgeq->thresholds [band] = threshold;
}

void dgeq_set_band_ratio (struct dynamic_graphic_eq *dgeq,
                          int band,
                          double ratio)
{
    dgeq->ratios [band] = ratio;

    if (ratio < 1.0)
        dsol_set_mode (dgeq->bands [band], EXPANDER);
    else
        dsol_set_mode (dgeq->bands [band], COMPRESSOR);
}

void dgeq_set_band_knee (struct dynamic_graphic_eq *dgeq,
                         int band,
                         double knee)
{
    dgeq->knees [band] = knee;
}

void dgeq_set_band_attack (struct dynamic_graphic_eq *dgeq,
                           int band,
                           double attack)
{
    dsol_set_attack (dgeq->bands [band], attack);
}

void dgeq_set_band_release (struct dynamic_graphic_eq *dgeq,
                            int band,
                            double release)
{
    dsol_set_release (dgeq->bands [band], release);
}

void dgeq_set_band_makeup_gain (struct dynamic_graphic_eq *dgeq,
                                int band,
                                double makeup_gain)
{
    sol_set_gain (dgeq->mgs [band], makeup_gain);
}

void dgeq_set_output_gain (struct dynamic_graphic_eq *dgeq,
                           double output_gain)
{
    dgeq->output_gain = db2a (output_gain);
}
