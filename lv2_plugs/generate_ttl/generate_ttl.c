#include "plugin_info.h"
#include <stdlib.h>
#include <stdio.h>
#include <serd-0/serd/serd.h>

#define TTL_FILENAME_BUFF_SIZE 20
#define USTR(s) ((const uint8_t*)(s))

/**************************************************************************
 * a bunch of nodes from whatever ontologies
 **************************************************************************/
static SerdNode rdf_uri = {0};
static SerdNode rdf = {0};
static SerdNode rdf_type = {0};

static SerdNode rdfs_uri = {0};
static SerdNode rdfs = {0};
static SerdNode rdfs_seeAlso = {0};

static SerdNode lv2_uri = {0};
static SerdNode lv2 = {0};
static SerdNode lv2_Plugin = {0};
static SerdNode lv2_minorVersion = {0};
static SerdNode lv2_microVersion = {0};
static SerdNode lv2_binary = {0};
static SerdNode lv2_project = {0};
static SerdNode lv2_optionalFeature = {0};
static SerdNode lv2_hardRTCapable = {0};
static SerdNode lv2_port = {0};
static SerdNode lv2_index = {0};
static SerdNode lv2_symbol = {0};
static SerdNode lv2_name = {0};
static SerdNode lv2_default = {0};
static SerdNode lv2_minimum = {0};
static SerdNode lv2_maximum = {0};
static SerdNode lv2_InputPort = {0};
static SerdNode lv2_OutputPort = {0};
static SerdNode lv2_AudioPort = {0};
static SerdNode lv2_ControlPort = {0};

static SerdNode doap_uri = {0};
static SerdNode doap = {0};
static SerdNode doap_name = {0};
static SerdNode doap_license = {0};

static void init_static_nodes()
{
    rdf_uri = serd_node_from_string (SERD_URI, USTR ("http://www.w3.org/1999/02/22-rdf-syntax-ns#"));
    rdf = serd_node_from_string (SERD_URI, USTR ("rdf"));
    rdf_type = serd_node_from_string (SERD_CURIE, USTR ("a"));

    rdfs_uri = serd_node_from_string (SERD_URI, USTR ("http://www.w3.org/2000/01/rdf-schema#"));
    rdfs = serd_node_from_string (SERD_URI, USTR ("rdfs"));
    rdfs_seeAlso = serd_node_from_string (SERD_CURIE, USTR ("rdfs:seeAlso"));

    lv2_uri = serd_node_from_string (SERD_URI, USTR ("http://lv2plug.in/ns/lv2core#"));
    lv2 = serd_node_from_string (SERD_URI, USTR ("lv2"));
    lv2_Plugin = serd_node_from_string (SERD_CURIE, USTR ("lv2:Plugin"));
    lv2_minorVersion = serd_node_from_string (SERD_CURIE, USTR ("lv2:minorVersion"));
    lv2_microVersion = serd_node_from_string (SERD_CURIE, USTR ("lv2:microVersion"));
    lv2_binary = serd_node_from_string (SERD_CURIE, USTR ("lv2:binary"));
    lv2_project = serd_node_from_string (SERD_CURIE, USTR ("lv2:project"));
    lv2_optionalFeature = serd_node_from_string (SERD_CURIE, USTR ("lv2:optionalFeature"));
    lv2_hardRTCapable = serd_node_from_string (SERD_CURIE, USTR ("lv2:hardRTCapable"));
    lv2_port = serd_node_from_string (SERD_CURIE, USTR ("lv2:port"));
    lv2_index = serd_node_from_string (SERD_CURIE, USTR ("lv2:index"));
    lv2_symbol = serd_node_from_string (SERD_CURIE, USTR ("lv2:symbol"));
    lv2_name = serd_node_from_string (SERD_CURIE, USTR ("lv2:name"));
    lv2_default = serd_node_from_string (SERD_CURIE, USTR ("lv2:default"));
    lv2_minimum = serd_node_from_string (SERD_CURIE, USTR ("lv2:minimum"));
    lv2_maximum = serd_node_from_string (SERD_CURIE, USTR ("lv2:maximum"));
    lv2_InputPort = serd_node_from_string (SERD_CURIE, USTR ("lv2:InputPort"));
    lv2_OutputPort = serd_node_from_string (SERD_CURIE, USTR ("lv2:OutputPort"));
    lv2_AudioPort = serd_node_from_string (SERD_CURIE, USTR ("lv2:AudioPort"));
    lv2_ControlPort = serd_node_from_string (SERD_CURIE, USTR ("lv2:ControlPort"));

    doap_uri = serd_node_from_string (SERD_URI, USTR ("http://usefulinc.com/ns/doap#"));
    doap = serd_node_from_string (SERD_URI, USTR ("doap"));
    doap_name = serd_node_from_string (SERD_CURIE, USTR ("doap:name"));
    doap_license = serd_node_from_string (SERD_CURIE, USTR ("doap:license"));
}

/**************************************************************************
 * ttl file writers
 **************************************************************************/
struct ttl_writer
{
    FILE *fd;
    SerdEnv *env;
    SerdWriter *writer;
};

static void free_ttl_writer (struct ttl_writer *w)
{
    if (w)
    {
        serd_writer_free (w->writer);
        serd_env_free (w->env);
        fclose (w->fd);
        free (w);
    }
}

static struct ttl_writer* init_ttl_writer (const char *filename)
{
    struct ttl_writer *w = calloc (1, sizeof (*w));

    if (!w)
        goto failed;

    w->fd = fopen (filename, "wb");

    if (!w->fd)
        goto failed;

    w->env = serd_env_new (NULL);

    if (!w->env)
        goto failed;

    w->writer = serd_writer_new (SERD_TURTLE,
                                 SERD_STYLE_ABBREVIATED,
                                 w->env,
                                 NULL,
                                 serd_file_sink,
                                 w->fd);

    if (!w->writer)
        goto failed;

    return w;

failed:
    free_ttl_writer (w);
    return NULL;
}

static SerdStatus ttl_writer_add_prefix (struct ttl_writer *w,
                                         SerdNode *name,
                                         SerdNode *uri)
{
    return serd_writer_set_prefix (w->writer, name, uri);
}

static SerdStatus ttl_writer_add_statement (struct ttl_writer *w,
                                            SerdNode *subject,
                                            SerdNode *predicate,
                                            SerdNode *object)
{
    return serd_writer_write_statement (w->writer,
                                        0,
                                        NULL,
                                        subject,
                                        predicate,
                                        object,
                                        NULL,
                                        NULL);
}

static SerdStatus ttl_writer_start_anon_object (struct ttl_writer *w,
                                                SerdNode *subject,
                                                SerdNode *predicate,
                                                SerdNode *object)
{
    return serd_writer_write_statement (w->writer,
                                        SERD_ANON_O_BEGIN,
                                        NULL,
                                        subject,
                                        predicate,
                                        object,
                                        NULL,
                                        NULL);
}

static SerdStatus ttl_writer_continue_anon_object (struct ttl_writer *w,
                                                   SerdNode *subject,
                                                   SerdNode *predicate,
                                                   SerdNode *object)
{
    return serd_writer_write_statement (w->writer,
                                        SERD_ANON_CONT,
                                        NULL,
                                        subject,
                                        predicate,
                                        object,
                                        NULL,
                                        NULL);
}

static SerdStatus ttl_writer_end_anon_object (struct ttl_writer *w,
                                              SerdNode *object)
{
    return serd_writer_end_anon (w->writer, object);
}

/**************************************************************************
 * process plugin info
 **************************************************************************/
static SerdStatus add_plugin_to_manifest (struct ttl_writer *manifest_writer,
                                          SerdNode *plugin_uri,
                                          SerdNode *lib_bin,
                                          const char *ttl_file)
{
    SerdNode file_uri = serd_node_from_string (SERD_URI, USTR (ttl_file));

    SerdStatus status = ttl_writer_add_statement (manifest_writer, plugin_uri, &rdf_type, &lv2_Plugin);
    status = ttl_writer_add_statement (manifest_writer, plugin_uri, &lv2_binary, lib_bin);
    status = ttl_writer_add_statement (manifest_writer, plugin_uri, &rdfs_seeAlso, &file_uri);

    return status;
}

static SerdStatus add_port (struct ttl_writer *writer,
                            SerdNode *plugin_uri,
                            uint32_t port_index,
                            struct port_info *info)
{
    SerdNode port = serd_node_from_string (SERD_BLANK, USTR ("port"));
    SerdNode index = serd_node_new_integer (port_index);
    SerdNode symbol = serd_node_from_string (SERD_LITERAL, USTR (info->symbol));
    SerdNode name = serd_node_from_string (SERD_LITERAL, USTR (info->name));
    SerdNode def = serd_node_new_decimal (info->def, 1);
    SerdNode min = serd_node_new_decimal (info->min, 1);
    SerdNode max = serd_node_new_decimal (info->max, 1);

    SerdStatus status = ttl_writer_start_anon_object (writer, plugin_uri, &lv2_port, &port);

    if (info->direction == INPUT_PORT)
        status = ttl_writer_continue_anon_object (writer, &port, &rdf_type, &lv2_InputPort);
    else
        status = ttl_writer_continue_anon_object (writer, &port, &rdf_type, &lv2_OutputPort);

    if (info->type == AUDIO_PORT)
        status = ttl_writer_continue_anon_object (writer, &port, &rdf_type, &lv2_AudioPort);
    else
        status = ttl_writer_continue_anon_object (writer, &port, &rdf_type, &lv2_ControlPort);

    status = ttl_writer_continue_anon_object (writer, &port, &lv2_index, &index);
    status = ttl_writer_continue_anon_object (writer, &port, &lv2_symbol, &symbol);
    status = ttl_writer_continue_anon_object (writer, &port, &lv2_name, &name);

    if (info->type == CONTROL_PORT)
    {
        status = ttl_writer_continue_anon_object (writer, &port, &lv2_default, &def);
        status = ttl_writer_continue_anon_object (writer, &port, &lv2_minimum, &min);
        status = ttl_writer_continue_anon_object (writer, &port, &lv2_maximum, &max);
    }

    status = ttl_writer_end_anon_object (writer, &port);

    serd_node_free (&max);
    serd_node_free (&min);
    serd_node_free (&def);
    serd_node_free (&index);

    return status;
}

static SerdStatus add_plugin_ports (struct ttl_writer *writer,
                                    uint32_t plugin_index,
                                    SerdNode *plugin_uri)
{
    SerdStatus status = SERD_SUCCESS;
    uint32_t n_ports = 0;

    if (!num_ports (plugin_index, &n_ports))
        return SERD_FAILURE;

    uint32_t *ports = calloc (n_ports, sizeof (*ports));

    if (!ports)
    {
        status = SERD_FAILURE;
        goto failed;
    }

    if (!get_port_indexes (plugin_index, ports))
    {
        status = SERD_FAILURE;
        goto failed;
    }

    for (int p = 0; p < n_ports; ++p)
    {
        uint32_t port_index = ports [p];
        struct port_info port_info = {0};

        if (!get_port_info (plugin_index, port_index, &port_info))
            continue;

        status = add_port (writer, plugin_uri, port_index, &port_info);
    }

failed:
    free (ports);
    return status;
}

static SerdStatus generate_plugin_file (uint32_t plugin_index,
                                        const struct plugin_info *info,
                                        struct ttl_writer *manifest_writer,
                                        SerdNode *lib_bin)
{
    /* create nodes for this plugin */
    SerdNode plugin_uri = serd_node_from_string (SERD_URI, USTR (info->uri));
    SerdNode plugin_name = serd_node_from_string (SERD_LITERAL, USTR (info->name));
    SerdNode plugin_type = serd_node_from_string (SERD_CURIE, USTR (info->type));
    SerdNode plugin_minor = serd_node_new_integer (info->minor_version);
    SerdNode plugin_micro = serd_node_new_integer (info->micro_version);
    SerdNode plugin_project = serd_node_from_string (SERD_URI, USTR (info->project));
    SerdNode plugin_license = serd_node_from_string (SERD_URI, USTR (info->license));

    /* work out ttl file name for it */
    char filename [TTL_FILENAME_BUFF_SIZE] = {0};
    snprintf (filename, TTL_FILENAME_BUFF_SIZE, "%s.ttl", info->name);

    /* add it to the bundle manifest */
    SerdStatus status = add_plugin_to_manifest (manifest_writer, &plugin_uri, lib_bin, filename);

    /* write its own ttl file */
    struct ttl_writer *writer = init_ttl_writer (filename);

    /* prefixes */
    status = ttl_writer_add_prefix (writer, &rdf, &rdf_uri);
    status = ttl_writer_add_prefix (writer, &lv2, &lv2_uri);
    status = ttl_writer_add_prefix (writer, &doap, &doap_uri);

    /* plugin details */
    status = ttl_writer_add_statement (writer, &plugin_uri, &rdf_type, &lv2_Plugin);
    status = ttl_writer_add_statement (writer, &plugin_uri, &rdf_type, &plugin_type);
    status = ttl_writer_add_statement (writer, &plugin_uri, &lv2_project, &plugin_project);
    status = ttl_writer_add_statement (writer, &plugin_uri, &doap_name, &plugin_name);
    status = ttl_writer_add_statement (writer, &plugin_uri, &lv2_minorVersion, &plugin_minor);
    status = ttl_writer_add_statement (writer, &plugin_uri, &lv2_microVersion, &plugin_micro);
    status = ttl_writer_add_statement (writer, &plugin_uri, &doap_license, &plugin_license);
    status = ttl_writer_add_statement (writer, &plugin_uri, &lv2_optionalFeature, &lv2_hardRTCapable);

    /* port details */
    status = add_plugin_ports (writer, plugin_index, &plugin_uri);

    free_ttl_writer (writer);
    serd_node_free (&plugin_micro);
    serd_node_free (&plugin_minor);

    return status;
}

/**************************************************************************
 * do everything
 **************************************************************************/
int main (int argc, char **argv)
{
    /* check input */
    if (argc < 2)
    {
        fprintf (stderr, "No library name provided.\n");
        return 1;
    }

    SerdNode lib_bin = serd_node_from_string (SERD_URI, USTR (argv [1]));

    /* initialise static variables */
    init_static_nodes();

    /* create manifest ttl */
    struct ttl_writer *manifest_writer = init_ttl_writer ("manifest.ttl");

    ttl_writer_add_prefix (manifest_writer, &rdf, &rdf_uri);
    ttl_writer_add_prefix (manifest_writer, &rdfs, &rdfs_uri);
    ttl_writer_add_prefix (manifest_writer, &lv2, &lv2_uri);

    /* create plugin ttls */
    uint32_t n_plugs = num_plugins();

    for (uint32_t i = 0; i < n_plugs; ++i)
    {
        struct plugin_info info = {0};
        
        if (!get_plugin_info (i, &info))
            continue;

        generate_plugin_file (i, &info, manifest_writer, &lib_bin);
    }

    free_ttl_writer (manifest_writer);

    return 0;
}
