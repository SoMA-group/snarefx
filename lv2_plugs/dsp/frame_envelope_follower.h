/**
 * A frame based amplitude follower.
 */

#ifndef FRAME_ENVELOPE_FOLLOWER_H_INCLUDED
#define FRAME_ENVELOPE_FOLLOWER_H_INCLUDED

/**
 * Envelope detection modes.
 */
enum detection_method
{
    PEAK_DETECTION, /**< Use the highest magnitude value in the frame as the signal amplitude. */
    RMS_DETECTION /**< Calculate the RMS amplitude of the values in the frame. */
};

/** Envelope follower struct */
struct frame_envelope_follower;

/**
 * Create an envelope follower struct.
 *
 * @param method the amplitude detection method to use
 * @param frame_size the analysis frame size
 * @param hop_size the analysis hop size
 */
struct frame_envelope_follower* fef_init (enum detection_method method,
                                          int frame_size,
                                          int hop_size);

/**
 * Free envelope follower struct.
 *
 * @param fef the struct to free
 */
void fef_free (struct frame_envelope_follower *fef);

/**
 * Process a single sample.
 *
 * Call this function repeatedly to add samples to the follower's analysis frame. Every hop_size samples which are
 * processed, a new level value will be calculated.
 *
 * @param fef the follower to use
 * @param in the input sample
 * @param level if the function reruns 1 the value pointed to by level will be updated to the new level value,
 *              otherwise level is left unchanged
 */
int fef_process_sample (struct frame_envelope_follower *fef,
                        double in,
                        double *level);

/**
 * Reset envelope follower analysis frame to zeros.
 *
 * @param fef the follower reset
 */
void fef_reset_state (struct frame_envelope_follower *fef);

#endif /* FRAME_ENVELOPE_FOLLOWER_H_INCLUDED */
