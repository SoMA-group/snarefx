#include "frame_envelope_follower.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double calculate_peak (const double *frame, int length)
{
    double peak = 0.0;

    for (int i = 0; i < length; ++i)
    {
        double m = fabs (frame [i]);

        if (m > peak)
            peak = m;
    }

    return peak;
}

static double calculate_rms (const double *frame, int length)
{
    double rms = 0.0;

    for (int i = 0; i < length; ++i)
    {
        rms += frame [i] * frame [i];
    }

    return sqrt (rms / length);
}

struct frame_envelope_follower
{
    double *frame;
    int frame_size, hop_size;
    int tap;
    int hop_count;
    double (*calculate_level) (const double*, int);
};

struct frame_envelope_follower* fef_init (enum detection_method method,
                                          int frame_size,
                                          int hop_size)
{
    struct frame_envelope_follower *fef = calloc (1, sizeof (*fef));

    if (!fef)
        goto failed;

    /* allocate analysis frame */
    fef->frame = calloc (frame_size, sizeof (*(fef->frame)));

    if (!fef->frame)
        goto failed;

    fef->frame_size = frame_size;
    fef->hop_size = hop_size;

    /* set analysis function */
    switch (method)
    {
        case PEAK_DETECTION:
            fef->calculate_level = calculate_peak;
            break;

        case RMS_DETECTION:
        default:
            fef->calculate_level = calculate_rms;

    }

    return fef;

failed:
    fef_free (fef);
    return NULL;
}

void fef_free (struct frame_envelope_follower *fef)
{
    if (fef)
    {
        free (fef->frame);
        free (fef);
    }
}

int fef_process_sample (struct frame_envelope_follower *fef,
                        double in,
                        double *level)
{
    /* put input sample in frame */
    fef->frame [fef->tap] = in;
    fef->tap = (fef->tap + 1) % fef->frame_size;

    /* do we need to calculate the fef? */
    if (++fef->hop_count == fef->hop_size)
    {
        *level = fef->calculate_level (fef->frame, fef->frame_size);
        fef->hop_count = 0;
        return 1;
    }

    return 0;
}

void fef_reset_state (struct frame_envelope_follower *fef)
{
    memset (fef->frame, 0, fef->frame_size * sizeof (*(fef->frame)));
    fef->tap = 0;
    fef->hop_count = 0;
}
