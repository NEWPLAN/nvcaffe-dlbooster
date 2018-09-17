//#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <sys/mman.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "caffe/glane_library.h"
#include <iostream>


class fpga_channel
{
  public:
    explicit fpga_channel(uint32_t _core_id);
    ~fpga_channel();
    submit_cmds(void){}

protected:
    void exit_with_status();

  private:
    int fd;
    int core_id;
    int core_num;
#define  MAX_CMD_LENGTH  95000
    struct glane_entry cmds[MAX_CMD_LENGTH];
    struct glane_entry cpls[MAX_CMD_LENGTH];

    uint32_t submitted_num= 0;
    uint32_t polled_num= 0;

    bool is_inited=false;
};



#define PFN_MASK_SIZE 8

int set_cpu_affinity(int core_id)
{
    int result;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    result = sched_setaffinity(0, sizeof(mask), &mask);
    return result;
}



fpga_channel::fpga_channel(uint32_t _core_id)
{
    this->core_num = init_driver("/dev/glaneoncpu0");
    
    if (core_num < 0)
    {
        std::cerr<<"init_driver(): error"<<std::endl;
        exit(-1);
    }
    if(_core_id<32 && _core_id>0)
        this->core_id = _core_id;
    else
        this->core_id = 19;
    if (set_cpu_affinity(this->core_id))
    {
        std::cerr<<"set_cpu_affinity(): error\n"<<std::endl;
        exit(-1);
    }

    for (int i = 0; i < MAX_CMD_LENGTH; i++)
    {
        cmds[i].req_type = 0x40; // REQ_AIPRE_READ
        cmds[i].dst_ipv4 = 123;
        cmds[i].dst_devid = 234;
        cmds[i].req_flag = 34;
    }
    is_inited = true;
}

fpga_channel::~fpga_channel()
{
    exit_with_status();
}

void fpga_channel::exit_with_status()
{
    if(is_inited)
        free_driver();
    is_inited = false;
}

inline uint64_t current_time();
uint64_t current_time()
{
    struct timespec tstart = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    return tstart.tv_nsec;
}

int glane_main(void)
{
    int fd, core_num, core_id = 19, i;
    int cmd_length = 95000, sub_length = 95000, cpl_length = 95000, sub_total_num = 0, cpl_total_num = 0;
    struct glane_entry cmds[cmd_length];
    struct glane_entry cpls[cmd_length];

    int launch = 0;
    int defaults = 0;
    printf("your seted launch time: ", launch);
    scanf("%d", &launch);
    defaults = launch;

    core_num = init_driver("/dev/glaneoncpu0");
    if (core_num < 0)
    {
        fprintf(stderr, "init_driver(): error\n");
        goto failed;
    }
    printf("core number is: %d\n", core_num);

    if (set_cpu_affinity(core_id))
    {
        fprintf(stderr, "set_cpu_affinity(): error\n");
        goto free_driver;
    }

    for (i = 0; i < cmd_length; i++)
    {
        cmds[i].req_type = 0x40; // REQ_AIPRE_READ
        cmds[i].dst_ipv4 = 123;
        cmds[i].dst_devid = 234;
        cmds[i].req_flag = 34;
    }

    while (launch-- > 0)
    {
        cmd_length = sub_length;
        do
        {
            sub_length = submit_cmd(core_id + 1, cmds, cmd_length);
            if (sub_length < 0)
            {
                fprintf(stderr, "submit_cmd(): error\n");
                goto free_driver;
            }
            // printf("submit: %d\n", sub_length);
            cmd_length -= sub_length;
        } while (cmd_length != 0);

        if (cmd_length != 0)
        {
            fprintf(stderr, "submit_cmd(): error\n");
            goto free_driver;
        }

        if (launch % 100 == 0)
            printf("submit success in %d\n", defaults - launch);

        cmd_length = cpl_length;
        do
        {
            cpl_length = read_completion(core_id + 1, cpls, cmd_length);
            if (cpl_length < 0)
            {
                fprintf(stderr, "read_completion(): error\n");
                goto free_driver;
            }
            // printf("complete: %d\n", cpl_length);
            cmd_length -= cpl_length;
        } while (cmd_length != 0);

        if (cmd_length != 0)
        {
            fprintf(stderr, "read_completion(): error\n");
            goto free_driver;
        }
        if (launch % 100 == 0)
            printf("complete success in %d\n", defaults - launch);

        for (i = 0; i < sub_length; i++)
            if (memcmp(&cmds[i], &cpls[i], sizeof(struct glane_entry)) != 0)
            {
                fprintf(stderr, "memcmp(): error\n");
                goto free_driver;
            }
        if (launch % 100 == 0)
            printf("memcmp success in %d\n", defaults - launch);
    }

free_driver:
    free_driver();
failed:
    return 0;
}
