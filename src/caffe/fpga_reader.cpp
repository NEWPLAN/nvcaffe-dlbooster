#include <boost/thread.hpp>
#include <sys/sysinfo.h>

#include "caffe/util/rng.hpp"
#include "caffe/parallel.hpp"
#include "caffe/fpga_reader.hpp"
#include <algorithm>

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

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good()) << "File " << source;
  string filename;
  int label;
  while (infile >> filename >> label)
  {
    FPGAReader::train_manifest.emplace_back(std::make_pair(filename, label));
  }
  /*LOG(INFO) << this->print_current_device() << " A total of " << FPGAReader::train_manifest.size() << " images.";*/
  LOG(INFO) <<" A total of " << FPGAReader::train_manifest.size() << " images.";

  size_t tmp_solver_count = 0U;
  auto& pixel_buffer = FPGAReader::pixel_queue[tmp_solver_count];
  auto& recycle_buffer = FPGAReader::recycle_queue;
  {
    for (auto index = 0 ; index < 1000; index++)
    {
      PackedData* tmp_buf = new PackedData;
      tmp_buf->label_ = new int[batch_size_];
      tmp_buf->data_ = new char[batch_size_ * height_ * width_ * channel_];

      tmp_buf->channel = channel_;
      tmp_buf->height = height_;
      tmp_buf->width = width_;
      tmp_buf->batch_size = batch_size_;

      sprintf(tmp_buf->data_, "producer id : %u, index = %d", lwp_id(), index);
      sprintf((char*)(tmp_buf->label_), "producer id : %u, index = %d", lwp_id(), index);
      while (!pixel_buffer.push(tmp_buf))
      {
        LOG(WARNING) << "Something wrong in push queue.";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  }
  LOG(INFO) << "FPGAReader finished construction function....";
  StartInternalThread(true, Caffe::next_seed());
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
  auto& shuffle_array = FPGAReader::val_manifest;
  std::random_shuffle ( shuffle_array.begin(), shuffle_array.end());
  LOG(INFO) << "shuffle FPGAreader....";
}

template<typename DatumType>
void FPGAReader<DatumType>::InternalThreadEntry()
{
  InternalThreadEntryN(0U);
}

template<typename DatumType>
void FPGAReader<DatumType>::InternalThreadEntryN(size_t thread_id)
{
  LOG(INFO)<< "In FPGA Reader.....loops";
  start_reading_flag_.wait(); // waiting for run.
  size_t tmp_solver_count = 0U;
  auto& pixel_buffer = FPGAReader::pixel_queue[tmp_solver_count];
  auto& recycle_buffer = FPGAReader::recycle_queue;

  LOG(INFO)<< "In FPGA Reader.....after wait";

  //shared_ptr<DatumType> datum = make_shared<DatumType>();
  try
  {
    int index = 1000;
    while (!must_stop(thread_id))
    {
      DatumType* tmp_datum = nullptr;

      if (index == 0)images_shuffles(0);

      if (recycle_buffer.pop(tmp_datum))
      {
        int cycles_index = 0;
        string a(tmp_datum->data_);
        
        LOG_EVERY_N(INFO, 10) << "Received from consumer: " << a;

        sprintf(tmp_datum->data_, "producer id : %u, index = %d", lwp_id(), index++);
        index %= 5000000;

        while (!pixel_buffer.push(tmp_datum))
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
  catch (boost::thread_interrupted&) {}
}

template class FPGAReader<PackedData>;

}  // namespace caffe
