#ifndef __MY_HUGE_PAGE__
#define __MY_HUGE_PAGE__

#if defined(__cplusplus)
#include <iostream>
#include <string>
#include <vector>
extern "C"
{
#endif

#define HUGEPAGE_FILE "/dev/hugepages1G/random"
#define LENGTH (1024UL * 1024 * 1024)   // 1M
#define PROTECTION (PROT_READ | PROT_WRITE)
#define ADDR (void *)(0x0UL)
#define FLAGS (MAP_SHARED | MAP_POPULATE)
#define BAD_PHYS_ADDR 0
#define PFN_MASK_SIZE 8


class _hugepage_
{
  public:
	explicit _hugepage_(std::string dev_info, uint32_t batch_size, uint32_t M_size);
	~_hugepage_();

	

	struct _batch_mem_
	{
		uint32_t index=0;
		void * virt_addr = 0;
		uint64_t phy_addr = 0;
		uint32_t size_in_m = 0;
	};

	//protected:
	uint64_t mem_virt2phy(const void *virtaddr);
	void display();
	void write_byt(char *addr, char c);
	void print_byt(char *addr);

  private:
	int MAX_LRNGTH = 1024 * 1024; //default is 1M

	std::string dev_info;
	int hugepage_fd = -1;

	void *addr = nullptr;
	uint64_t paddr = 0;

	std::vector<_batch_mem_> batch_vect;
};

int hugepage_test(void);

#if defined(__cplusplus)
}
#endif

#endif