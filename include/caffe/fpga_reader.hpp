#ifndef CAFFE_FPGA_READER_HPP_
#define CAFFE_FPGA_READER_HPP_

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
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

  // push back
  bool recycle_packed_data(DatumType* packed_data)
  {
    while (!FPGAReader::recycle_queue.push(packed_data))
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(11));
    }
    return true;
  }

  bool pop_packed_data(DatumType* &packed_data, int bulket = 0)
  {
    while (!FPGAReader::pixel_queue[bulket].pop(packed_data))
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(11));
    }
    return true;
  }

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

protected:
  //newplan added
  void images_shuffles(int shuffle_rank);
  static vector<boost::lockfree::queue<DatumType*, boost::lockfree::capacity<1024>>> pixel_queue;
  static boost::lockfree::queue<DatumType*, boost::lockfree::capacity<1024>> recycle_queue;
  static vector<std::pair<std::string, int>> train_manifest;
  static vector<std::pair<std::string, int>> val_manifest;

  DISABLE_COPY_MOVE_AND_ASSIGN(FPGAReader);
};

template <typename DatumType>
vector<boost::lockfree::queue<DatumType*, boost::lockfree::capacity<1024>>> FPGAReader<DatumType>::pixel_queue(MAX_GPU_PER_MACHINE_SUPPORT);
template <typename DatumType>
boost::lockfree::queue<DatumType*, boost::lockfree::capacity<1024>> FPGAReader<DatumType>::recycle_queue;
template <typename DatumType>
vector<std::pair<std::string, int>> FPGAReader<DatumType>::train_manifest;
template <typename DatumType>
vector<std::pair<std::string, int>> FPGAReader<DatumType>::val_manifest;

}  // namespace caffe

#endif  // CAFFE_FPGA_READER_HPP_