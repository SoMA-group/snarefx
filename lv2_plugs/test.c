#include <dynamic_graphic_eq.h>
#include <biquad.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sndfile.h>

static void print_octave_array (const char *name, double *buffer, int length)
{
    /* create octave data */
    printf ("%s = [", name);

    for (int i = 0; i < length - 1; ++i)
    {
        printf ("%e,", buffer [i]);
    }

    printf ("%e];\n", buffer [length - 1]);
}

int main(int argc, const char **argv)
{
    if (argc <= 2)
        return 1;

    const char *in_file = argv [1];
    const char *out_file = argv [2];

    /* Open audio files */
    SF_INFO in_info;
    in_info.format = 0;
    SNDFILE *in_snd = sf_open (in_file, SFM_READ, &in_info);

    SF_INFO out_info;
    out_info.samplerate = in_info.samplerate;
    out_info.channels = 1;
    out_info.format = in_info.format;
    SNDFILE *out_snd = sf_open (out_file, SFM_WRITE, &out_info);

    /* process audio */
    int buff_size = 10000;
    double *in_buff = malloc (buff_size * in_info.channels * sizeof (*in_buff));
    double *out_buff = malloc (buff_size * sizeof (*out_buff));
    int n_frames_left = in_info.frames;

    int frame = in_info.samplerate * 0.0125;
    int hop = frame;
    int n_bands = 30;
    struct dynamic_graphic_eq *dgeq = dgeq_init (in_info.samplerate, n_bands, 25.0, pow (2.0, 1.0 / 3.0), frame, hop);

    const double thresholds[] = {0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0};
    const double ratios[] = {1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0};
    const double knees[] = {0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0};
    const double attacks[] = {10.0, 10.0, 10.0, 10.0, 10.0,
                              10.0, 10.0, 10.0, 10.0, 10.0,
                              10.0, 10.0, 10.0, 10.0, 10.0,
                              10.0, 10.0, 10.0, 10.0, 10.0,
                              10.0, 10.0, 10.0, 10.0, 10.0,
                              10.0, 10.0, 10.0, 10.0, 10.0};
    const double releases[] = {100.0, 100.0, 100.0, 100.0, 100.0,
                               100.0, 100.0, 100.0, 100.0, 100.0,
                               100.0, 100.0, 100.0, 100.0, 100.0,
                               100.0, 100.0, 100.0, 100.0, 100.0,
                               100.0, 100.0, 100.0, 100.0, 100.0,
                               100.0, 100.0, 100.0, 100.0, 100.0};
    const double makeups[] = {0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0};

    for (int b = 0; b < n_bands; ++b)
    {
        dgeq_set_band_threshold (dgeq, b, thresholds [b]);
        dgeq_set_band_ratio (dgeq, b, ratios [b]);
        dgeq_set_band_knee (dgeq, b, knees [b]);
        dgeq_set_band_attack (dgeq, b, attacks [b]);
        dgeq_set_band_release (dgeq, b, releases [b]);
        dgeq_set_band_makeup_gain (dgeq, b, makeups [b]);
    }

    dgeq_set_output_gain (dgeq, -6.0);

    while (n_frames_left > 0)
    {
        int n = buff_size > n_frames_left ? n_frames_left : buff_size;

        sf_readf_double (in_snd, in_buff, n);

        for (int i = 0; i < n; ++i)
        {
            out_buff [i] = in_buff [i * in_info.channels];
        }

        dgeq_process_buffer (dgeq, out_buff, out_buff, n);

        sf_writef_double (out_snd, out_buff, n);

        n_frames_left -= n;
    }

    sf_write_sync (out_snd);
    
    dgeq_free (dgeq);
    free (out_buff);
    free (in_buff);
    sf_close (out_snd);
    sf_close (in_snd);

    return 0;
}
