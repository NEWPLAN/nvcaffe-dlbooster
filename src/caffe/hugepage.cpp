#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include "caffe/fpga/hugepage.h"

static void write_byt(char *addr, char c);
static void print_byt(char *addr);

// Get physical address of any mapped virtual address in the current process
static uint64_t mem_virt2phy(const void *virtaddr);




_hugepage_::_hugepage_(std::string dev_info, uint32_t batch_size, uint32_t M_size)
{
	this->MAX_LRNGTH = MAX_LRNGTH * M_size;
	this->dev_info = dev_info;

	if (dev_info.length() == 0 || batch_size == 0 || M_size == 0)
	{
		std::cerr << "error in init hugepage, out!" << std::endl;
		exit(-1);
	}

	hugepage_fd = open(this->dev_info.c_str(), O_CREAT | O_RDWR, 0755);
	if (hugepage_fd < 0)
	{
		std::cerr << "Fail to open hugepage memory file" << std::endl;
		exit(-1);
	}

	addr = mmap(ADDR, LENGTH, PROTECTION, FLAGS, hugepage_fd, 0);
	if (addr == MAP_FAILED)
	{
		std::cerr << "mmap error" << std::endl;
		unlink(this->dev_info.c_str());
		exit(-1);
	}

	paddr = this->mem_virt2phy(addr);

	for(uint32_t index=0;index<MAX_LRNGTH/(batch_size);index++)
	{
		_batch_mem_ a;
		a.index=index;
		a.size_in_m=batch_size;
		a.phy_addr=paddr+batch_size*index;
		a.virt_addr = addr+batch_size*index;
		batch_vect.push_back(a);
	}
}

_hugepage_::~_hugepage_()
{
	std::cout << "destroy variable" << std::endl;
	if (addr != nullptr)
		munmap(addr, MAX_LRNGTH);
	if (hugepage_fd >= 0)
	{
		close(hugepage_fd);
		unlink(this->dev_info.c_str());
	}
}

uint64_t _hugepage_::mem_virt2phy(const void *virtaddr)
{
	int fd, retval;
	uint64_t page, physaddr;
	unsigned long virt_pfn; // virtual page frame number
	int page_size;
	off_t offset;

	/* standard page size */
	page_size = getpagesize();
	fd = open("/proc/self/pagemap", O_RDONLY);
	if (fd < 0)
	{
		fprintf(stderr, "%s(): cannot open /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	}
	virt_pfn = (unsigned long)virtaddr / page_size;
	//printf("Virtual page frame number is %llu\n", virt_pfn);

	offset = sizeof(uint64_t) * virt_pfn;
	if (lseek(fd, offset, SEEK_SET) == (off_t)-1)
	{
		fprintf(stderr, "%s(): seek error in /proc/self/pagemap: %s\n", __func__, strerror(errno));
		close(fd);
		return BAD_PHYS_ADDR;
	}
	retval = read(fd, &page, PFN_MASK_SIZE);
	close(fd);
	if (retval < 0)
	{
		fprintf(stderr, "%s(): cannot read /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	}
	else if (retval != PFN_MASK_SIZE)
	{
		fprintf(stderr, "%s(): read %d bytes from /proc/self/pagemap but expected %d:\n",
				__func__, retval, PFN_MASK_SIZE);
		return BAD_PHYS_ADDR;
	}

	/*
	 * the pfn (page frame number) are bits 0-54 (see
	 * pagemap.txt in linux Documentation)
	 */
	if ((page & 0x7fffffffffffffULL) == 0)
	{
		fprintf(stderr, "Zero page frame number\n");
		return BAD_PHYS_ADDR;
	}
	physaddr = ((page & 0x7fffffffffffffULL) * page_size) + ((unsigned long)virtaddr % page_size);
	return physaddr;
}

void _hugepage_::display()
{
	printf("Virtual address is %p\n", addr);
	printf("Physical address is %llu\n", paddr);

	std::cout<<"batched items is: "<<batch_vect.size()<<std::endl;
	for(int index = 0; index<batch_vect.size();index++)
	{
		std::cout<<"index: "<<batch_vect[index].index
		<<" physical addr: "<<batch_vect[index].phy_addr
		<<" virt addr: "<<batch_vect[index].virt_addr
		<<" block size: "<<batch_vect[index].size_in_m
		<<std::endl;
	}
}

void _hugepage_::write_byt(char *addr, char c)
{
	if (addr)
	{
		*addr = c;
	}
	else
	{
		fprintf(stderr, "%s(): empty pointer\n", __func__);
	}
}
void _hugepage_::print_byt(char *addr)
{
	if (addr)
	{
		printf("%d\n", (int)(*addr));
	}
	else
	{
		fprintf(stderr, "%s(): empty pointer\n", __func__);
	}
}

int hugepage_main(void)
{

	uint32_t item_size = 256*256*3;
	_hugepage_ var(HUGEPAGE_FILE, 256*item_size, 1024);
	var.display();
	return 0;
}

int hugepage_main__2(void)
{
	void *addr;
	int hugepage_fd;
	uint64_t paddr;

	hugepage_fd = open(HUGEPAGE_FILE, O_CREAT | O_RDWR, 0755);
	if (hugepage_fd < 0)
	{
		perror("Fail to open hugepage memory file");
		exit(1);
	}

	addr = mmap(ADDR, LENGTH, PROTECTION, FLAGS, hugepage_fd, 0);
	if (addr == MAP_FAILED)
	{
		perror("mmap");
		unlink(HUGEPAGE_FILE);
		exit(1);
	}

	paddr = mem_virt2phy(addr);
	printf("Virtual address is %p\n", addr);
	printf("Physical address is %llu\n", paddr);

	munmap(addr, LENGTH);
	close(hugepage_fd);
	unlink(HUGEPAGE_FILE);

	return 0;
}

static void write_byt(char *addr, char c)
{
	if (addr)
	{
		*addr = c;
	}
	else
	{
		fprintf(stderr, "%s(): empty pointer\n", __func__);
	}
}

static void print_byt(char *addr)
{
	if (addr)
	{
		printf("%d\n", (int)(*addr));
	}
	else
	{
		fprintf(stderr, "%s(): empty pointer\n", __func__);
	}
}

static uint64_t mem_virt2phy(const void *virtaddr)
{
	int fd, retval;
	uint64_t page, physaddr;
	unsigned long virt_pfn; // virtual page frame number
	int page_size;
	off_t offset;

	/* standard page size */
	page_size = getpagesize();

	fd = open("/proc/self/pagemap", O_RDONLY);
	if (fd < 0)
	{
		fprintf(stderr, "%s(): cannot open /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	}

	virt_pfn = (unsigned long)virtaddr / page_size;
	//printf("Virtual page frame number is %llu\n", virt_pfn);

	offset = sizeof(uint64_t) * virt_pfn;
	if (lseek(fd, offset, SEEK_SET) == (off_t)-1)
	{
		fprintf(stderr, "%s(): seek error in /proc/self/pagemap: %s\n", __func__, strerror(errno));
		close(fd);
		return BAD_PHYS_ADDR;
	}

	retval = read(fd, &page, PFN_MASK_SIZE);
	close(fd);
	if (retval < 0)
	{
		fprintf(stderr, "%s(): cannot read /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	}
	else if (retval != PFN_MASK_SIZE)
	{
		fprintf(stderr, "%s(): read %d bytes from /proc/self/pagemap but expected %d:\n",
				__func__, retval, PFN_MASK_SIZE);
		return BAD_PHYS_ADDR;
	}

	/*
	 * the pfn (page frame number) are bits 0-54 (see
	 * pagemap.txt in linux Documentation)
	 */
	if ((page & 0x7fffffffffffffULL) == 0)
	{
		fprintf(stderr, "Zero page frame number\n");
		return BAD_PHYS_ADDR;
	}

	physaddr = ((page & 0x7fffffffffffffULL) * page_size) + ((unsigned long)virtaddr % page_size);

	return physaddr;
}
