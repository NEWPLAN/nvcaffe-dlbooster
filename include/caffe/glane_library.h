#ifndef GLANE_LIBRARY_H
#define GLANE_LIBRARY_H
#include <stdint.h>
#if defined (__cplusplus)
extern "C" {
#endif
struct glane_entry
{
    uint8_t req_type;
    uint8_t req_flag;
    uint16_t dst_devid;
    uint32_t dst_ipv4;
    uint64_t param[4];
};

/**
 * Initialize the glane driver
 * @param chardev_name the name of character device for FPGA (e.g. "/dev/glane_kernel0") 
 * @return status code (<0 means failed)
 */
int init_driver(char *chardev_name);

/**
 * Free the glane driver
 */
void free_driver();

/**
 * Register a dma buffer
 * @param id the id of the dma buffer
 * @param size the size of the dma buffer
 * @param cpu_id the dma buffer should be close to the NUMA node of cpu_id cpu 
 * @return status code (<0 means failed)
 */
int register_glane_buf(uint32_t id, uint32_t size, uint8_t cpu_id);

/**
 * Deregister a dma buffer
 * @param id the id of the dma buffer
 * @param size the size of the dma buffer
 * @param cpu_id the dma buffer should be close to the NUMA node of cpu_id cpu 
 * @return status code (<0 means failed)
 */
int deregister_glane_buf(uint32_t id);

/**
 * Request a dma buffer
 * @param id the id of the dma buffer
 * @param addr returned address will be filled in this field
 * @param size returned size will be filled in this field
 * @return status code (<0 means failed)
 */
int request_glane_buf(uint32_t id, void **addr, uint32_t *size);

/**
 * Submit the submission queue entries
 * @param cmd desired entries to insert
 * @param count how many entries desired
 * @return successful submitted entry count (<0 means failed)
 */
int submit_admin_cmd(struct glane_entry *cmd, int count);

/**
 * Submit the submission queue entries
 * @param qid the id of desired submission queue (0 for admin queue)
 * @param cmd desired entries to insert
 * @param count how many entries desired
 * @return successful submitted entry count (<0 means failed)
 */
int submit_cmd(int qid, struct glane_entry *cmd, int count);

/**
 * Read the admin completion queue entries
 * @param cmd where fetched entries stores
 * @param count how many entries desired
 * @return successful fetched entry count (<0 means failed)
 */
int read_admin_completion(struct glane_entry *cmd, int count);

/**
 * Read the completion queue entries
 * @param qid the id of desired completion queue (0 for admin queue)
 * @param cmd where fetched entries stores
 * @param count how many entries desired
 * @return successful fetched entry count (<0 means failed)
 */
int read_completion(int qid, struct glane_entry *cmd, int count);

// int write_target_address(uint64_t addr);
// int write_target_value(uint64_t value);
// int trigger_fpga();

int glane_main(void);
#if defined (__cplusplus)
}
#endif
#endif