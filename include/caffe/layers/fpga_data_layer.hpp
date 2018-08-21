#ifndef CAFFE_FPGA_DATA_LAYER_HPP_
#define CAFFE_FPGA_DATA_LAYER_HPP_

#include <map>
#include <vector>
#include <atomic>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/batch_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/thread_pool.hpp"

//newplan added
#include <boost/lockfree/queue.hpp>
#include "caffe/fpga_reader.hpp"

namespace caffe
{

template <typename Ftype, typename Btype>
class FPGADataLayer : public BasePrefetchingDataLayer<Ftype, Btype>
{
public:
  FPGADataLayer(const LayerParameter& param, size_t solver_rank);
  virtual ~FPGADataLayer();
  void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  // FPGADataLayer uses DataReader instead for sharing for parallelism
  bool ShareInParallel() const override
  {
    return false;
  }
  const char* type() const override {return "FPGAData";}
  int ExactNumBottomBlobs() const override { return 0;}
  int MinTopBlobs() const override { return 1; }
  int MaxTopBlobs() const override { return 2; }
  Flag* layer_inititialized_flag() override
  {
    return this->phase_ == TRAIN ? &layer_inititialized_flag_ : nullptr;
  }
  size_t prefetch_bytes() { return this->batch_transformer_->prefetch_bytes(); }

protected:
  void InitializePrefetch() override;
  void load_batch(Batch* batch, int thread_id, size_t queue_id = 0UL) override;
  size_t queue_id(size_t thread_id) const override;

  void init_offsets();
  void start_reading() override ;

  std::vector<shared_ptr<GPUMemory::Workspace>> tmp_gpu_buffer_;

  // stored random numbers for this batch
  std::vector<shared_ptr<TBlob<unsigned int>>> random_vectors_;
  Flag layer_inititialized_flag_;
  const bool shuffle_;

  //newplan added
  static std::shared_ptr<FPGAReader<PackedData>> train_reader_, val_reader_;
  std::shared_ptr<FPGAReader<PackedData>> train_reader, val_reader;
  size_t _solver_rank;
};

//newplan added
template <typename Ftype, typename Btype>
std::shared_ptr<FPGAReader<PackedData>> FPGADataLayer<Ftype, Btype>::train_reader_;
template <typename Ftype, typename Btype>
std::shared_ptr<FPGAReader<PackedData>> FPGADataLayer<Ftype, Btype>::val_reader_;

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
