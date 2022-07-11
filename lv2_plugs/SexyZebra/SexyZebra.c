#include <lv2.h>
#include <dynamic_graphic_eq.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <plugin_info.h>

#define SZ10_URI "https://gitlab.com/dmt-soma/lv2_plugs#SexyZebra10"
#define SZ30_URI "https://gitlab.com/dmt-soma/lv2_plugs#SexyZebra30"

#define SZ10_N_BANDS 10
#define SZ30_N_BANDS 30

enum plugin_id
{
    SZ10_ID,
    SZ30_ID,
    NUM_PLUGINS
};

enum band_parameter
{
    BAND_THRESHOLD,
    BAND_RATIO,
    BAND_KNEE,
    BAND_ATTACK,
    BAND_RELEASE,
    BAND_MAKEUP_GAIN,
    N_BAND_PARAMS
};

enum non_band_port
{
    OUTPUT_GAIN,
    AUDIO_IN,
    AUDIO_OUT,
    N_NON_BAND_PORTS
};

struct SexyZebra
{
    /* input and output buffers */
    const float *in;
    float *out;

    /* the EQ */
    int n_bands;
    struct dynamic_graphic_eq *dgeq;

    /* parameters */
    const float **thresholds;
    const float **ratios;
    const float **knees;
    const float **attacks;
    const float **releases;
    const float **makeup_gains;
    const float *output_gain;
};

static void cleanup (LV2_Handle instance)
{
    struct SexyZebra *sz = (struct SexyZebra*) instance;

    if (sz)
    {
        free (sz->makeup_gains);
        free (sz->releases);
        free (sz->attacks);
        free (sz->knees);
        free (sz->ratios);
        free (sz->thresholds);

        dgeq_free (sz->dgeq);
        free (sz);
    }
}

static LV2_Handle instantiate (const LV2_Descriptor *descriptor,
                               double rate,
                               const char *bundle_path,
                               const LV2_Feature* const *features,
                               int n_bands,
                               double start_freq,
                               double freq_ratio)
{
    /* allocate plugin struct */
    struct SexyZebra *sz = calloc (1, sizeof (*sz));

    if (!sz)
        goto failed;

    /* initialise dsp */
    int frame_size = rate * 0.0125;
    int hop_size = frame_size;

    sz->n_bands = n_bands;
    sz->dgeq = dgeq_init (rate, n_bands, start_freq, freq_ratio, frame_size, hop_size);

    if (!sz->dgeq)
        goto failed;

    /* allocate parameter pointers */
    sz->thresholds = calloc (n_bands, sizeof (*(sz->thresholds)));
    sz->ratios = calloc (n_bands, sizeof (*(sz->ratios)));
    sz->knees = calloc (n_bands, sizeof (*(sz->knees)));
    sz->attacks = calloc (n_bands, sizeof (*(sz->attacks)));
    sz->releases = calloc (n_bands, sizeof (*(sz->releases)));
    sz->makeup_gains = calloc (n_bands, sizeof (*(sz->makeup_gains)));

    if (!(sz->thresholds &&
          sz->ratios &&
          sz->knees &&
          sz->attacks &&
          sz->releases &&
          sz->makeup_gains))
        goto failed;

    return (LV2_Handle) sz;

failed:
    cleanup ((LV2_Handle) sz);
    return NULL;
}

static LV2_Handle instantiate_sz10 (const LV2_Descriptor *descriptor,
                                    double rate,
                                    const char *bundle_path,
                                    const LV2_Feature* const *features)
{
    int n_bands = SZ10_N_BANDS;
    double start_freq = 25.0;
    double freq_ratio = 2.0;

    return instantiate (descriptor,
                        rate,
                        bundle_path,
                        features,
                        n_bands,
                        start_freq,
                        freq_ratio);
}

static LV2_Handle instantiate_sz30 (const LV2_Descriptor *descriptor,
                                    double rate,
                                    const char *bundle_path,
                                    const LV2_Feature* const *features)
{
    int n_bands = SZ30_N_BANDS;
    double start_freq = 25.0;
    double freq_ratio = pow (2.0, 1.0 / 3.0);

    return instantiate (descriptor,
                        rate,
                        bundle_path,
                        features,
                        n_bands,
                        start_freq,
                        freq_ratio);
}

static void port_to_band_param (uint32_t port,
                                uint32_t *band,
                                enum band_parameter *param)
{
    *band = port / N_BAND_PARAMS;
    *param = port % N_BAND_PARAMS;
}

static uint32_t band_param_to_port (uint32_t band,
                                    enum band_parameter param)
{
    return band * N_BAND_PARAMS + param;
}

static void connect_band_port (struct SexyZebra *sz,
                               uint32_t band,
                               enum band_parameter param,
                               void *data)
{
    if (band >= sz->n_bands)
        return;

    switch (param)
    {
        case BAND_THRESHOLD:
            sz->thresholds [band] = (const float*) data;
            break;
        case BAND_RATIO:
            sz->ratios [band] = (const float*) data;
            break;
        case BAND_KNEE:
            sz->knees [band] = (const float*) data;
            break;
        case BAND_ATTACK:
            sz->attacks [band] = (const float*) data;
            break;
        case BAND_RELEASE:
            sz->releases [band] = (const float*) data;
            break;
        case BAND_MAKEUP_GAIN:
            sz->makeup_gains [band] = (const float*) data;
            break;
        default:
            break;
    }
}

static void connect_ports (LV2_Handle instance,
                           uint32_t port,
                           void *data,
                           uint32_t first_non_band_port)
{
    struct SexyZebra *sz = (struct SexyZebra*) instance;

    if (port < first_non_band_port)
    {
        uint32_t band = 0;
        enum band_parameter param = BAND_THRESHOLD;

        port_to_band_param (port, &band, &param);
        connect_band_port (sz, band, param, data);
    }
    else
    {
        switch ((enum non_band_port) (port - first_non_band_port))
        {
            case OUTPUT_GAIN:
                sz->output_gain = (const float*) data;
                break;
            case AUDIO_IN:
                sz->in = (const float*) data;
                break;
            case AUDIO_OUT:
                sz->out = (float*) data;
                break;
            default:
                break;
        }
    }
}

static uint32_t output_gain_ports[] = {SZ10_N_BANDS * N_BAND_PARAMS,
                                       SZ30_N_BANDS * N_BAND_PARAMS};

static void connect_port_sz10 (LV2_Handle instance,
                               uint32_t port,
                               void *data)
{
    connect_ports (instance, port, data, output_gain_ports [SZ10_ID]);
}

static void connect_port_sz30 (LV2_Handle instance,
                               uint32_t port,
                               void *data)
{
    connect_ports (instance, port, data, output_gain_ports [SZ30_ID]);
}

static void activate (LV2_Handle instance)
{
    struct SexyZebra *sz = (struct SexyZebra*) instance;

    dgeq_reset_state (sz->dgeq);
}

static void run (LV2_Handle instance,
                 uint32_t n_samples)
{
    struct SexyZebra *sz = (struct SexyZebra*) instance;

    /* update parameters */
    for (int b = 0; b < sz->n_bands; ++b)
    {
        dgeq_set_band_threshold (sz->dgeq, b, *(sz->thresholds [b]));
        dgeq_set_band_ratio (sz->dgeq, b, *(sz->ratios [b]));
        dgeq_set_band_knee (sz->dgeq, b, *(sz->knees [b]));
        dgeq_set_band_attack (sz->dgeq, b, *(sz->attacks [b]));
        dgeq_set_band_release (sz->dgeq, b, *(sz->releases [b]));
        dgeq_set_band_makeup_gain (sz->dgeq, b, *(sz->makeup_gains [b]));
    }

    dgeq_set_output_gain (sz->dgeq, *(sz->output_gain));

    /* process buffer */
    for (int s = 0; s < n_samples; ++s)
    {
        sz->out [s] = dgeq_process_sample (sz->dgeq, sz->in [s]);
    }
}

static void deactivate (LV2_Handle instance)
{
    /* nothing to do here */
}

static const void* extension_data (const char *uri)
{
    return NULL;
}

static const LV2_Descriptor sz10_descriptor = {SZ10_URI,
                                               instantiate_sz10,
                                               connect_port_sz10,
                                               activate,
                                               run,
                                               deactivate,
                                               cleanup,
                                               extension_data};

static const LV2_Descriptor sz30_descriptor = {SZ30_URI,
                                               instantiate_sz30,
                                               connect_port_sz30,
                                               activate,
                                               run,
                                               deactivate,
                                               cleanup,
                                               extension_data};

LV2_SYMBOL_EXPORT
const LV2_Descriptor* lv2_descriptor (uint32_t index)
{
    switch (index)
    {
        case SZ10_ID:
            return &sz10_descriptor;

        case SZ30_ID:
            return &sz30_descriptor;

        default:
            return NULL;
    }
}

/**************************************
 * ttl generation stuff
 **************************************/
static const char * plugin_uris[] = {SZ10_URI, SZ30_URI};
static const char * plugin_names[] = {"SexyZebra10", "SexyZebra30"};
static const char * plugin_type = "lv2:DynamicsPlugin";
static const char * plugin_project = "https://gitlab.com/dmt-soma/lv2_plugs";
static const char * plugin_license = "https://opensource.org/licenses/MIT";

uint32_t num_plugins()
{
    return NUM_PLUGINS;
}

int get_plugin_info (uint32_t index, struct plugin_info *info)
{
    if (index >= NUM_PLUGINS)
        return 0;

    info->uri = plugin_uris [index];
    info->name = plugin_names [index];
    info->type = plugin_type;
    info->minor_version = 0;
    info->micro_version = 0;
    info->project = plugin_project;
    info->license = plugin_license;

    return 1;
}

static uint32_t plugin_n_bands[] = {SZ10_N_BANDS, SZ30_N_BANDS};

int num_ports (uint32_t plugin_index, uint32_t *n_ports)
{
    if (plugin_index >= NUM_PLUGINS)
        return 0;

    *n_ports = plugin_n_bands [plugin_index] * N_BAND_PARAMS + N_NON_BAND_PORTS;

    return 1;
}

int get_port_indexes (uint32_t plugin_index,
                      uint32_t *indexes)
{
    if (plugin_index >= NUM_PLUGINS)
        return 0;

    int i = 0;

    for (int b = 0; b < plugin_n_bands [plugin_index]; ++b)
    {
        for (int p = 0; p < N_BAND_PARAMS; ++p)
        {
            indexes [i++] = band_param_to_port (b, p);
        }
    }

    uint32_t output_gain = output_gain_ports [plugin_index];

    for (int p = output_gain; p < output_gain + N_NON_BAND_PORTS; ++p)
    {
        indexes [i++] = p;
    }

    return 1;
}

static void set_band_param_name (struct port_info *info,
                                 uint32_t band,
                                 const char *symbol_param,
                                 const char *name_param)
{
    const char *symbol_fmt = "band_%" PRIu32 "_%s";
    const char *name_fmt = "Band %" PRIu32 " %s";

    snprintf (info->symbol,
              PORT_NAME_BUFF_SIZE,
              symbol_fmt,
              band,
              symbol_param);

    snprintf (info->name,
              PORT_NAME_BUFF_SIZE,
              name_fmt,
              band,
              name_param);
}
                            

static int band_param_port_info (uint32_t band,
                                 enum band_parameter param,
                                 struct port_info *info)
{
    switch (param)
    {
        case BAND_THRESHOLD:
            set_band_param_name (info, band, "threshold", "Threshold");
            info->def = 0.0f;
            info->min = -200.0f;
            info->max = 0.0f;
            break;

        case BAND_RATIO:
            set_band_param_name (info, band, "ratio", "Ratio");
            info->def = 1.0f;
            info->min = 0.1f;
            info->max = 100.0f;
            break;

        case BAND_KNEE:
            set_band_param_name (info, band, "knee", "Knee");
            info->def = 0.0f;
            info->min = 0.0f;
            info->max = 20.0f;
            break;

        case BAND_ATTACK:
            set_band_param_name (info, band, "attack", "Attack");
            info->def = 10.0f;
            info->min = 10.0f;
            info->max = 500.0f;
            break;

        case BAND_RELEASE:
            set_band_param_name (info, band, "release", "Release");
            info->def = 100.0f;
            info->min = 10.0f;
            info->max = 2000.0f;
            break;

        case BAND_MAKEUP_GAIN:
            set_band_param_name (info, band, "makeup_gain", "Makeup Gain");
            info->def = 0.0f;
            info->min = -20.0f;
            info->max = 20.0f;
            break;

        default:
            return 0;
    }

    return 1;
}

static int other_param_port_info (uint32_t port,
                                  struct port_info *info)
{
    snprintf (info->symbol,
              PORT_NAME_BUFF_SIZE,
              "output_gain");
    snprintf (info->name,
              PORT_NAME_BUFF_SIZE,
              "Output Gain");
    info->def = 0.0f;
    info->min = -20.0f;
    info->max = 20.0f;

    return 1;
}

static int control_port_info (uint32_t port,
                              uint32_t first_non_band_port,
                              struct port_info *info)
{
    info->direction = INPUT_PORT;
    info->type = CONTROL_PORT;

    if (port < first_non_band_port)
    {
        uint32_t band = 0;
        enum band_parameter param = BAND_THRESHOLD;

        port_to_band_param (port, &band, &param);

        return band_param_port_info (band, param, info);
    }
    else
    {
        return other_param_port_info (port, info);
    }
}

static int audio_port_info (uint32_t plugin_index,
                            uint32_t port,
                            struct port_info *info)
{
    info->type = AUDIO_PORT;

    switch ((enum non_band_port) (port - output_gain_ports [plugin_index]))
    {
        case AUDIO_IN:
            info->direction = INPUT_PORT;
            snprintf (info->symbol,
                      PORT_NAME_BUFF_SIZE,
                      "in");
            snprintf (info->name,
                      PORT_NAME_BUFF_SIZE,
                      "Input");
            break;

        case AUDIO_OUT:
            info->direction = OUTPUT_PORT;
            snprintf (info->symbol,
                      PORT_NAME_BUFF_SIZE,
                      "out");
            snprintf (info->name,
                      PORT_NAME_BUFF_SIZE,
                      "Output");
            break;

        default:
            return 0;
    }

    return 1;
}

int get_port_info (uint32_t plugin_index,
                   uint32_t port_index,
                   struct port_info *info)
{
    if (plugin_index >= NUM_PLUGINS)
        return 0;

    uint32_t output_gain = output_gain_ports [plugin_index];

    if (port_index <= output_gain)
        return control_port_info (port_index, output_gain, info);
    else
        return audio_port_info (plugin_index, port_index, info);
}
