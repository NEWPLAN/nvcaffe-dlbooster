#ifndef CAFFE_FPGA_READER_HPP_
#define CAFFE_FPGA_READER_HPP_

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <hash_map>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/thread_pool.hpp"

#define MAX_GPU_PER_MACHINE_SUPPORT 128

namespace caffe
{


class PackedData
{
public:
  size_t channel;
  size_t height;
  size_t width;
  size_t batch_size;
  int* label_;
  char* data_;
};

/**
 * @brief Reads data from a source to queues available to data layers.
 * Few reading threads are created per source, every record gets it's unique id
 * to allow deterministic ordering down the road. Data is distributed to solvers
 * in a round-robin way to keep parallel training deterministic.
 */
template <typename DatumType>
class FPGAReader : public InternalThread
{

public:
  FPGAReader(const LayerParameter& param,
             size_t solver_count,
             size_t solver_rank,
             size_t batch_size,
             bool shuffle,
             bool epoch_count_required);
  virtual ~FPGAReader();

  void start_reading() { start_reading_flag_.set();}
  bool producer_pop(DatumType* &packed_data, int bulket);
  bool producer_push(DatumType* packed_data, int bulket);
  bool consumer_pop(DatumType* &packed_data, int bulket);
  bool consumer_push(DatumType* packed_data, int bulket);

protected:
  void InternalThreadEntry() override;
  void InternalThreadEntryN(size_t thread_id) override;

  const size_t solver_count_, solver_rank_;
  size_t batch_size_, channel_, height_, width_;

private:
  Flag start_reading_flag_;
  const bool shuffle_;
  const bool epoch_count_required_;
  string manifest_path;

  hash_map<std::string,char*> _cache_vect;
  bool _cache_all;

protected:
  void images_shuffles(int shuffle_rank);
  static vector<std::pair<std::string, int>> train_manifest[2];
  static vector<std::pair<std::string, int>> val_manifest[2];

  static vector<std::shared_ptr<BlockingQueue<DatumType*>>> fpga_pixel_queue;
  static vector<std::shared_ptr<BlockingQueue<DatumType*>>> fpga_cycle_queue;


  DISABLE_COPY_MOVE_AND_ASSIGN(FPGAReader);
};

template <typename DatumType>
vector<std::pair<std::string, int>> FPGAReader<DatumType>::train_manifest[2];
template <typename DatumType>
vector<std::pair<std::string, int>> FPGAReader<DatumType>::val_manifest[2];

template <typename DatumType>
vector<std::shared_ptr<BlockingQueue<DatumType*>>> FPGAReader<DatumType>::fpga_pixel_queue;
template <typename DatumType>
vector<std::shared_ptr<BlockingQueue<DatumType*>>> FPGAReader<DatumType>::fpga_cycle_queue;

}  // namespace caffe

#endif  // CAFFE_FPGA_READER_HPP_
