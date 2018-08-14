#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <map>
#include <vector>
#include <atomic>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/batch_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/thread_pool.hpp"

//newplan added
#include <boost/lockfree/queue.hpp>

namespace caffe {

template <typename Ftype, typename Btype>
class DataLayer : public BasePrefetchingDataLayer<Ftype, Btype> {
 public:
  DataLayer(const LayerParameter& param, size_t solver_rank);
  virtual ~DataLayer();
  void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  // DataLayer uses DataReader instead for sharing for parallelism
  bool ShareInParallel() const override {
    return false;
  }
  const char* type() const override {
    return "Data";
  }
  int ExactNumBottomBlobs() const override {
    return 0;
  }
  int MinTopBlobs() const override {
    return 1;
  }
  int MaxTopBlobs() const override {
    return 2;
  }
  Flag* layer_inititialized_flag() override {
    return this->phase_ == TRAIN ? &layer_inititialized_flag_ : nullptr;
  }
  size_t prefetch_bytes() {
    return this->batch_transformer_->prefetch_bytes();
  }

 protected:
  void InitializePrefetch() override;
  void load_batch(Batch* batch, int thread_id, size_t queue_id = 0UL) override;
  size_t queue_id(size_t thread_id) const override;

  void init_offsets();
  void start_reading() override {
    reader_->start_reading();
  }

  std::shared_ptr<DataReader<Datum>> sample_reader_, reader_;
  std::vector<shared_ptr<GPUMemory::Workspace>> tmp_gpu_buffer_;

  // stored random numbers for this batch
  std::vector<shared_ptr<TBlob<unsigned int>>> random_vectors_;
  mutable std::vector<size_t> parser_offsets_, queue_ids_;
  Flag layer_inititialized_flag_;
  std::atomic_bool sample_only_;
  const bool cache_, shuffle_;
  bool datum_encoded_;

  //newplan added
  static boost::lockfree::queue<char*, boost::lockfree::capacity<1024>> pixel_queue, cycle_queue;
	static vector<std::pair<std::string, int>> train_index;
	static vector<std::pair<std::string, int>> val_index;
  static void fpga_reader_cycle(uint32_t batch_size, uint32_t new_height, uint32_t new_width, uint32_t channel)
	{
		char* abc = nullptr;
		while (!DataLayer::pixel_queue.empty())
			DataLayer::pixel_queue.pop(abc);

		while (!DataLayer::cycle_queue.empty())
			DataLayer::cycle_queue.pop(abc);

		for (auto index = 0 ; index < 1000; index++)
		{
			char* tmp_buf = new char[batch_size * new_height * new_width * channel];
			sprintf(tmp_buf, "producer id : %u, index = %d", lwp_id(), index);
			DataLayer::pixel_queue.push(tmp_buf);
		}

		int index = 1000;
		while (true)
		{
			char* abc = nullptr;
			if (DataLayer::cycle_queue.pop(abc))
			{
				int cycles_index = 0;
				string a(abc);

				LOG(INFO) << "Received from consumer: " << a;
				sprintf(abc, "producer id : %u, index = %d", lwp_id(), index++);
				index %= 50000;

				while (!DataLayer::pixel_queue.push(abc))
				{
					if (cycles_index % 100 == 0)
					{
						LOG(WARNING) << "Something wrong in push queue.";
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(50));
				}
			}
			else
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
		}
	}
};

//newplan added
template <typename Ftype, typename Btype>
boost::lockfree::queue<char*, boost::lockfree::capacity<1024>> DataLayer<Ftype, Btype>::pixel_queue;
template <typename Ftype, typename Btype>
boost::lockfree::queue<char*, boost::lockfree::capacity<1024>> DataLayer<Ftype, Btype>::cycle_queue;
template <typename Ftype, typename Btype>
vector<std::pair<std::string, int>> DataLayer<Ftype, Btype>::train_index;
template <typename Ftype, typename Btype>
vector<std::pair<std::string, int>> DataLayer<Ftype, Btype>::val_index;

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
