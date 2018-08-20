#include <boost/thread.hpp>
#include <sys/sysinfo.h>

#include "caffe/util/rng.hpp"
#include "caffe/parallel.hpp"
#include "caffe/fpga_reader.hpp"
#include <algorithm>
#include <cstdlib>
#include <future>

namespace caffe
{

template<typename DatumType>
FPGAReader<DatumType>::FPGAReader(const LayerParameter& param,
                                  size_t solver_count,
                                  size_t solver_rank,
                                  size_t batch_size,
                                  bool shuffle,
                                  bool epoch_count_required)
  : InternalThread(Caffe::current_device(), solver_rank, 1U, false),
    solver_count_(solver_count),
    solver_rank_(solver_rank),
    batch_size_(batch_size),
    shuffle_(shuffle),
    epoch_count_required_(epoch_count_required)
{

  //batch_size_ = param.data_param().batch_size();
  height_ = param.data_param().new_height();
  width_ = param.data_param().new_width();
  channel_ = param.data_param().new_channel();

  string source = param.data_param().manifest();
  // Read the file with filenames and labels
  FPGAReader::train_manifest.clear();
  
  FPGAReader::fpga_pixel_queue.resize(solver_count_);
  FPGAReader::fpga_cycle_queue.resize(solver_count_);
  for(size_t index = 0; index<solver_count_; index++)
  {
    FPGAReader::fpga_pixel_queue[index]=std::make_shared<BlockingQueue<DatumType*>>();
    FPGAReader::fpga_cycle_queue[index]=std::make_shared<BlockingQueue<DatumType*>>();
  }
  

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good()) << "File " << source;
  string filename;
  int label;
  while (infile >> filename >> label)
  {
    FPGAReader::train_manifest.emplace_back(std::make_pair(filename, label));
  }

  LOG(INFO) << " A total of " << FPGAReader::train_manifest.size() << " images.";

  for (int s_index = 0; s_index < solver_count_; s_index++)
  {
    auto& pixel_buffer = FPGAReader::fpga_pixel_queue[s_index];
    for (auto index = 0 ; index < 16 ; index++)
    {
      PackedData* tmp_buf = new PackedData;
      tmp_buf->label_ = new int[batch_size_];
      //tmp_buf->data_ = new char[batch_size_ * height_ * width_ * channel_];
      CUDA_CHECK(cudaMallocHost((void**) & (tmp_buf->data_), batch_size_ * height_ * width_ * channel_));

      tmp_buf->channel = channel_;
      tmp_buf->height = height_;
      tmp_buf->width = width_;
      tmp_buf->batch_size = batch_size_;

      sprintf(tmp_buf->data_, "producer id : %u, index = %d", lwp_id(), index);
      sprintf((char*)(tmp_buf->label_), "producer id : %u, index = %d", lwp_id(), index);
      pixel_buffer->push(tmp_buf);
    }
  }
  LOG(INFO) << "FPGAReader finished construction function, batch size is: "
            << batch_size_ << "Solver counter is : " << solver_count_;
  StartInternalThread(false, Caffe::next_seed());
}

template<typename DatumType>
FPGAReader<DatumType>::~FPGAReader()
{
  StopInternalThread();
  LOG(INFO) << "FPGAReader goodbye....";
}

template<typename DatumType>
void FPGAReader<DatumType>::images_shuffles(int shuffle_rank)
{
  CPUTimer timer;
  timer.Start();
  auto& shuffle_array = FPGAReader::train_manifest;
  std::random_shuffle ( shuffle_array.begin(), shuffle_array.end());
  timer.Stop();
  LOG(INFO) << "shuffle " << shuffle_array.size() << " Images...." << timer.MilliSeconds() << " ms";
}

template<typename DatumType>
void FPGAReader<DatumType>::InternalThreadEntry()
{
  InternalThreadEntryN(0U);
}

template<typename DatumType>
void FPGAReader<DatumType>::InternalThreadEntryN(size_t thread_id)
{
  std::srand ( unsigned ( std::time(0) ) );
  LOG(INFO) << "In FPGA Reader.....loops";
  start_reading_flag_.wait(); // waiting for running.
  LOG(INFO) << "In FPGA Reader.....after wait";
  
  int current_shuffle = 0;
  std::future<int> f1 = std::async(std::launch::async, [current_shuffle++](){
            images_shuffles(current_shuffle%2);
            return ".";
  });

  int item_nums = FPGAReader::train_manifest.size() / batch_size_;
  try
  {
    int index = 100;
    while (!must_stop(thread_id))
    {
      for (int s_index = 0; s_index < solver_count_; s_index++)
      {
        DatumType* tmp_datum = nullptr;
        if (index == 0)
        {
          
          LOG(INFO) << "After " << item_nums << " itertations" << f1.get();
          f1=std::async(std::launch::async, [current_shuffle++](){
            images_shuffles(current_shuffle%2);
            return ".";
          });
          //images_shuffles(0);
        }
        if (must_stop(thread_id)) break;
        producer_pop(tmp_datum, s_index);
        string a(tmp_datum->data_);
        LOG_EVERY_N(INFO, 100) << "Received from consumer: " << a;
        sprintf(tmp_datum->data_, "producer id : %u, index = %d", lwp_id(), index++);
        index %= item_nums;
        if (must_stop(thread_id)) break;
        producer_push(tmp_datum, s_index);
      }
    }
  }
  catch (boost::thread_interrupted&) {}
}
template<typename DatumType>
bool  FPGAReader<DatumType>::producer_pop(DatumType* &packed_data, int bulket)
{
  packed_data = FPGAReader::fpga_cycle_queue[bulket]->pop("producer pop empty");
  return true;
}
template<typename DatumType>
bool  FPGAReader<DatumType>::producer_push(DatumType* packed_data, int bulket)
{
  FPGAReader::fpga_pixel_queue[bulket]->push(packed_data);
  return true;
}
template<typename DatumType>
bool FPGAReader<DatumType>::consumer_pop(DatumType* &packed_data, int bulket)
{
  packed_data = FPGAReader::fpga_pixel_queue[bulket]->pop("consumer pop empty");
  return true;
}
template<typename DatumType>
bool  FPGAReader<DatumType>::consumer_push(DatumType* packed_data, int bulket)
{
  FPGAReader::fpga_cycle_queue[bulket]->push(packed_data);
  return true;
}

template class FPGAReader<PackedData>;

}  // namespace caffe
