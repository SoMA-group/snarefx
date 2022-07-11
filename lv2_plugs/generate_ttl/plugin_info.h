#ifndef TTL_GENERATION_H_INCLUDED
#define TTL_GENERATION_H_INCLUDED

#include <stdint.h>
#include <inttypes.h>

#define PORT_NAME_BUFF_SIZE 40

struct plugin_info
{
    const char *uri;
    const char *name;
    const char *type;
    unsigned int minor_version;
    unsigned int micro_version;
    const char *project;
    const char *license;
};

uint32_t num_plugins();
int get_plugin_info (uint32_t index, struct plugin_info *info);

enum port_direction
{
    INPUT_PORT,
    OUTPUT_PORT
};

enum port_type
{
    AUDIO_PORT,
    CONTROL_PORT
};

struct port_info
{
    enum port_direction direction;
    enum port_type type;
    char symbol [PORT_NAME_BUFF_SIZE];
    char name [PORT_NAME_BUFF_SIZE];
    float def;
    float min;
    float max;
};

int num_ports (uint32_t plugin_index, uint32_t *n_ports);
int get_port_indexes (uint32_t plugin_index,
                      uint32_t *indexes);
int get_port_info (uint32_t plugin_index,
                   uint32_t port_index,
                   struct port_info *info);


#endif /* TTL_GENERATION_H_INCLUDED */
