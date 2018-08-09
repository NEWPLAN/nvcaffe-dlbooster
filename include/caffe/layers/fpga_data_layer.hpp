#ifndef CAFFE_FPGA_DATA_LAYER_HPP_
#define CAFFE_FPGA_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <boost/lockfree/queue.hpp>

namespace caffe
{

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Ftype, typename Btype>
class FPGADataLayer : public BasePrefetchingDataLayer<Ftype, Btype>
{
public:
  size_t im_solver;
  FPGADataLayer(const LayerParameter& param, size_t solver_rank);
  virtual ~FPGADataLayer();
  void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  bool ShareInParallel() const override
  {
    return false;
  }
  const char* type() const override { return "FPGAData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }

protected:
  void ShuffleImages();
  void load_batch(Batch* batch, int thread_id, size_t queue_id = 0UL) override;
  void start_reading() override {}
  void InitializePrefetch() override;

  bool auto_mode() const override
  {
    return false;
  }

  Flag* layer_inititialized_flag() override
  {
    return this->phase_ == TRAIN ? &layer_inititialized_flag_ : nullptr;
  }

  static size_t id(const string& ph, const string& name);
  cv::Mat next_mat(const string& root_folder, const string& filename, int height, int width,
                   bool is_color, int short_side);

  const size_t id_;  // per layer per phase
  shared_ptr<Caffe::RNG> prefetch_rng_;
  Flag layer_inititialized_flag_;
  size_t epoch_count_;
  vector<size_t> line_ids_;
  static vector<vector<std::pair<std::string, int>>> lines_;  // per id_
  static vector<unordered_map<std::string, cv::Mat>> cache_;
  static vector<std::mutex> cache_mutex_;
  static vector<bool> cached_;
  static vector<size_t> cached_num_, failed_num_;
  static vector<float> cache_progress_;

  static boost::lockfree::queue<char*, boost::lockfree::capacity<1024>> pixel_queue, cycle_queue;
	static vector<std::pair<std::string, int>> train_index;
	static vector<std::pair<std::string, int>> val_index;

	static void fpga_reader_cycle(uint32_t batch_size, uint32_t new_height, uint32_t new_width, uint32_t channel)
  {
    char* abc=nullptr;
    while(!FPGADataLayer::pixel_queue.empty())
      FPGADataLayer::pixel_queue.pop(abc);

    while(!FPGADataLayer::cycle_queue.empty())
      FPGADataLayer::cycle_queue.pop(abc);

    for(auto index =0 ;index < 1000; index++)
    {
        FPGADataLayer::pixel_queue.push(new char[batch_size * new_height * new_width * channel]);
    }

    while(true)
    {
      LOG_EVERY_N(INFO, 100) << "in read threads...-----";
        char* abc = nullptr;
        if (FPGADataLayer::cycle_queue.pop(abc))
        {
            int cycles_index = 0;
            while(!FPGADataLayer::pixel_queue.push(abc))
            {
                if(cycles_index % 100 == 0)
                {
                    LOG(WARNING) << "Something wrong in push queue.";
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
            }
        }
        else
        {
          LOG_EVERY_N(INFO, 100) << "in read threads...-----" ;
          std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
        }
    }
  }
};

#define MAX_IDL_CACHEABLE (2UL * Phase_ARRAYSIZE)

template <typename Ftype, typename Btype>
vector<vector<std::pair<std::string, int>>> FPGADataLayer<Ftype, Btype>::lines_(MAX_IDL_CACHEABLE);
template <typename Ftype, typename Btype>
vector<unordered_map<std::string, cv::Mat>> FPGADataLayer<Ftype, Btype>::cache_(MAX_IDL_CACHEABLE);
template <typename Ftype, typename Btype>
vector<bool> FPGADataLayer<Ftype, Btype>::cached_(MAX_IDL_CACHEABLE);
template <typename Ftype, typename Btype>
vector<size_t> FPGADataLayer<Ftype, Btype>::cached_num_(MAX_IDL_CACHEABLE);
template <typename Ftype, typename Btype>
vector<size_t> FPGADataLayer<Ftype, Btype>::failed_num_(MAX_IDL_CACHEABLE);
template <typename Ftype, typename Btype>
vector<std::mutex> FPGADataLayer<Ftype, Btype>::cache_mutex_(MAX_IDL_CACHEABLE);
template <typename Ftype, typename Btype>
vector<float> FPGADataLayer<Ftype, Btype>::cache_progress_(MAX_IDL_CACHEABLE);

template <typename Ftype, typename Btype>
boost::lockfree::queue<char*, boost::lockfree::capacity<1024>> FPGADataLayer<Ftype, Btype>::pixel_queue;
template <typename Ftype, typename Btype>
boost::lockfree::queue<char*, boost::lockfree::capacity<1024>> FPGADataLayer<Ftype, Btype>::cycle_queue;
template <typename Ftype, typename Btype>
vector<std::pair<std::string, int>> FPGADataLayer<Ftype, Btype>::train_index;
template <typename Ftype, typename Btype>
vector<std::pair<std::string, int>> FPGADataLayer<Ftype, Btype>::val_index;

}  // namespace caffe

#endif  // CAFFE_FPGA_DATA_LAYER_HPP_
