
/*
/home/yang/project/imagenet/lmdb/ilsvrc12/train.txt
/mnt/dc_p3700/imagenet/train/
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include <sys/time.h>
using namespace std;

#include <stdio.h>
#include <sys/types.h>
#include "caffe/fpga/hdparm.h"

#define MAXCOUNT 1024


int main_weibai(string file_name)
{
	struct block_info result[MAXCOUNT];
	int count;

	my_do_filemap(file_name.c_str(), result, MAXCOUNT, &count);

	if (count > MAXCOUNT)
	{
		fprintf(stderr, "count (%d) > maxcount (%d)\n", count, MAXCOUNT);
		return -1;
	}

	 /*for (int i = 0; i < count; i++)
	 {
	 	printf("%lu %u\n", result[i].begin_lba, result[i].length);
	 }*/
	 if(count >0 &&  (result[0].begin_lba%8!=0))
	 	std::cout<<file_name.c_str()<<", first block is not aligned in 4K"<<std::endl;

	return 0;
}


static const unsigned int sector_bytes = 512;

struct command
{
	uint64_t begin_lba;     // begin logical block address (LBA)
	uint32_t length;        // segment length in bytes
	uint64_t recv_addr;     // physical memory address to receive this segment
	uint32_t available_byte; // available bytes for this chunk.
};

// Read block information of a file
vector<struct command> read_file_blk(string name);

static uint64_t current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


std::vector<string> manifest;
std::vector<string> load_manifest(const char* manifest_path)
{
	///*
	std::ifstream infile(manifest_path);
	if (!infile.good())
	{
		std::cout << "can not open file: " << manifest_path << std::endl;
		exit(0);
	}
	printf("open file: %s\n", manifest_path);
	string filename;
	int label;
	while (infile >> filename >> label)
	{
		//std::cout << filename << "," << label << std::endl;
		manifest.push_back(filename);
	}
	std::cout << "A total of " << manifest.size() << " images." << std::endl;
	infile.close();
	return manifest;
}

int hdparm_main(const char* name, const char* path)
{
	vector<struct command> blk_cmds;

	printf("Hello world, newplan...\n");
	//"/mnt/dc_p3700/imagenet/train"
	string root_path = string(path);
	//*/
	std::vector<string> _manifest;
	_manifest = load_manifest(name);
	uint64_t index = 0;
	for (auto& each_file : _manifest)
	{
		index++;
		blk_cmds.clear();
		uint64_t before = current_time();
		string fff = root_path + each_file;
		main_weibai(root_path + each_file);
		uint64_t after = current_time();
		std::set<uint64_t> v;
		for (int i = 0; i < blk_cmds.size(); i++)
		{
			/*printf("%llu %u\n", blk_cmds[i].begin_lba, blk_cmds[i].available_byte);
			for (int block_index = 0; block_index < blk_cmds[i].available_byte / sector_bytes; block_index += 1)
			{
			        //printf("%10llu in %d\n", blk_cmds[i].begin_lba + block_index * sector_bytes, sector_bytes);
			}
			if (blk_cmds[i].available_byte % sector_bytes)
			{
			        //printf("%10llu in %d\n", blk_cmds[i].begin_lba + blk_cmds[i].available_byte / sector_bytes * sector_bytes, blk_cmds[i].available_byte % sector_bytes);
			}
			*/
			if (v.find(blk_cmds[i].begin_lba) != v.end())
			{
				std::cout << "running error for " << each_file << std::endl;
				//exit(0);
			}
			v.insert(blk_cmds[i].begin_lba);
		}
		if (index % 5000 == 0)
		{
			std::cout << "finished parsing " << index << std::endl;
			std::cout << "rate : " << 1000 / ((after - before) / 1000.0) << " im/s" << std::endl;
		}
		{
			v.clear();
		}
	}
	return 0;
}

