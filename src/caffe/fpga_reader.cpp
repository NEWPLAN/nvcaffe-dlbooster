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

template <typename DatumType>
FPGAReader<DatumType>::FPGAReader(const LayerParameter &param,
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
  FPGAReader::train_manifest[0].clear();
  FPGAReader::train_manifest[1].clear();

  FPGAReader::fpga_pixel_queue.resize(solver_count_);
  FPGAReader::fpga_cycle_queue.resize(solver_count_);
  for (size_t index = 0; index < solver_count_; index++)
  {
    FPGAReader::fpga_pixel_queue[index] = std::make_shared<BlockingQueue<DatumType *>>();
    FPGAReader::fpga_cycle_queue[index] = std::make_shared<BlockingQueue<DatumType *>>();
  }

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good()) << "File " << source;
  string filename;
  int label;
  while (infile >> filename >> label)
  {
    FPGAReader::train_manifest[0].emplace_back(std::make_pair(filename, label));
    FPGAReader::train_manifest[1].emplace_back(std::make_pair(filename, label));
    /*LOG_EVERY_N(INFO,1000)<<filename<<", "<<label;*/
    DLOG(INFO) << filename << ", " << label;
  }

  LOG(INFO) << " A total of " << FPGAReader::train_manifest[0].size() << " images.";

  CHECK_G(FPGAReader::train_manifest[0].size(), 0);

  cached_all_ = (FPGAReader::train_manifest[0].size() * height_ * width_ * channel_ < 1024 * 1024 * 1024 * 4) ? true : false;

  for (int s_index = 0; s_index < solver_count_; s_index++)
  {
    auto &pixel_buffer = FPGAReader::fpga_cycle_queue[s_index];
    for (auto index = 0; index < 16; index++)
    {
      PackedData *tmp_buf = new PackedData;
      tmp_buf->label_ = new int[batch_size_];
      //tmp_buf->data_ = new char[batch_size_ * height_ * width_ * channel_];
      CUDA_CHECK(cudaMallocHost((void **)&(tmp_buf->data_), batch_size_ * height_ * width_ * channel_));

      tmp_buf->channel = channel_;
      tmp_buf->height = height_;
      tmp_buf->width = width_;
      tmp_buf->batch_size = batch_size_;

      sprintf(tmp_buf->data_, "producer id : %u, index = %d", lwp_id(), index);
      for (int ii_index = 0; ii_index < batch_size_; ii_index++)
      {
        tmp_buf->label_[ii_index] = 0;
      }
      pixel_buffer->push(tmp_buf);
    }
  }
  LOG(INFO) << "FPGAReader finished construction function, batch size is: "
            << batch_size_ << ", Solver counter is: " << solver_count_;
  StartInternalThread(false, Caffe::next_seed());
}

template <typename DatumType>
FPGAReader<DatumType>::~FPGAReader()
{
  StopInternalThread();
  LOG(INFO) << "FPGAReader goodbye....";
}

template <typename DatumType>
void FPGAReader<DatumType>::images_shuffles(int shuffle_rank)
{
  CPUTimer timer;
  timer.Start();
  auto &shuffle_array = FPGAReader::train_manifest[shuffle_rank];
  std::random_shuffle(shuffle_array.begin(), shuffle_array.end());
  timer.Stop();
  LOG(INFO) << "shuffle " << shuffle_array.size() << " Images...." << timer.MilliSeconds() << " ms";
}

template <typename DatumType>
void FPGAReader<DatumType>::InternalThreadEntry()
{
  InternalThreadEntryN(0U);
}

template <typename DatumType>
void FPGAReader<DatumType>::InternalThreadEntryN(size_t thread_id)
{
  std::srand(unsigned(std::time(0)));
  LOG(INFO) << "In FPGA Reader.....loops";
  start_reading_flag_.wait(); // waiting for running.
  LOG(INFO) << "In FPGA Reader.....after wait";
  CPUTimer ctime_;
  size_t epoch_cou = 0;
  ctime_.Start();

  int current_shuffle = 0;
  std::future<int> f1 = std::async(std::launch::async, [&, current_shuffle]() {
    FPGAReader::images_shuffles((current_shuffle + 1) % 2);
    return 0;
  });

  int item_nums = FPGAReader::train_manifest[0].size() / batch_size_;
  int total_size = FPGAReader::train_manifest[0].size();
  try
  {
    int index = 100;
    string file_root = "/mnt/dc_p3700/imagenet/mnist/";
    while (!must_stop(thread_id))
    {
      if (index == 0)
      {
        f1.get();
        ctime_.Stop();
        LOG(INFO) << "Finished the " << epoch_cou++ << "th Epoch in " << ctime_.Seconds()
                  << " s. This Epoch contains " << item_nums << " itertations.";
        f1 = std::async(std::launch::async, [&, current_shuffle]() {
          FPGAReader::images_shuffles(current_shuffle % 2);
          return 0;
        });
        current_shuffle = (current_shuffle + 1) % 2;
        ctime_.Start();
      }
      auto &current_manfist = FPGAReader::train_manifest[current_shuffle];

      for (int s_index = 0; s_index < solver_count_; s_index++)
      {
        DatumType *tmp_datum = nullptr;

        if (must_stop(thread_id))
          break;
        producer_pop(tmp_datum, s_index);
        string a(tmp_datum->data_);
        DLOG_EVERY_N(INFO, 100) << "Received from consumer: " << a;
        sprintf(tmp_datum->data_, "producer id : %u, index = %d", lwp_id(), index++);
        index %= item_nums;
        if (must_stop(thread_id))
          break;

        size_t each_one_size = channel_ * width_ * height_;

        for (int _inde = 0; _inde < batch_size_; _inde++)
        {
          auto &file_item = current_manfist[(_inde + index * batch_size_) % total_size];
          string file_path = file_root + file_item.first;
          if (_cache_all)
          {
            auto iter = _cache_vect.find(file_path);
            if (iter == _cache_vect.end())
            {
              char *tmpbuf = new char[each_one_size + 1];
              FILE *fp = fopen(file_path.c_str(), "rb");
              CHECK(fp != nullptr);
              CHECK(each_one_size == fread(tmpbuf, sizeof(char), each_one_size, fp));
              fclose(fp);
              _cache_vect[file_path] = tmpbuf;
              iter = _cache_vect.find(file_path);
              CHECK(iter != _cache_vect.end());
            }
            memcpy(tmp_datum->data_ + each_one_size * _inde, iter->second, each_one_size);
            tmp_datum->label_[_inde] = file_item.second;
          }
          else
          {
            FILE *fp = fopen(file_path.c_str(), "rb");
            CHECK(fp != nullptr);
            CHECK(each_one_size == fread(tmp_datum->data_ + each_one_size * _inde, sizeof(char), each_one_size, fp));
            tmp_datum->label_[_inde] = file_item.second;
            fclose(fp);
          }
        }
        producer_push(tmp_datum, s_index);
      }
    }
  }
  catch (boost::thread_interrupted &)
  {
  }
}
template <typename DatumType>
bool FPGAReader<DatumType>::producer_pop(DatumType *&packed_data, int bulket)
{
  packed_data = FPGAReader::fpga_cycle_queue[bulket]->pop("producer pop empty");
  return true;
}
template <typename DatumType>
bool FPGAReader<DatumType>::producer_push(DatumType *packed_data, int bulket)
{
  FPGAReader::fpga_pixel_queue[bulket]->push(packed_data);
  return true;
}
template <typename DatumType>
bool FPGAReader<DatumType>::consumer_pop(DatumType *&packed_data, int bulket)
{
  packed_data = FPGAReader::fpga_pixel_queue[bulket]->pop("consumer pop empty");
  return true;
}
template <typename DatumType>
bool FPGAReader<DatumType>::consumer_push(DatumType *packed_data, int bulket)
{
  FPGAReader::fpga_cycle_queue[bulket]->push(packed_data);
  return true;
}

template class FPGAReader<PackedData>;

} // namespace caffe
