#include <boost/thread.hpp>
#include <sys/sysinfo.h>

#include "caffe/util/rng.hpp"
#include "caffe/parallel.hpp"
#include "caffe/fpga_reader.hpp"

namespace caffe 
{

template<typename DatumType>
FPGAReader<DatumType>::FPGAReader(const LayerParameter& param,
      size_t solver_count,
      size_t solver_rank,
      size_t transf_threads_num,
      bool shuffle,
      bool epoch_count_required)
    : InternalThread(Caffe::current_device(),
          solver_rank, 1U, false),
      parser_threads_num_(1U),
      transf_threads_num_(transf_threads_num),
      queues_num_(parser_threads_num_ * transf_threads_num_),
      queue_depth_(queue_depth),
      solver_count_(solver_count),
      solver_rank_(solver_rank),
      skip_one_batch_(skip_one_batch),
      current_rec_(0),
      current_queue_(0),
      shuffle_(shuffle),
      epoch_count_required_(epoch_count_required) {
  CHECK(queues_num_);
  CHECK(queue_depth_);
  batch_size_ = param.data_param().batch_size();

  free_.resize(queues_num_);
  full_.resize(queues_num_);
  LOG(INFO) << (sample_only ? "Sample " : "") << "Data Reader threads: "
      << this->threads_num() << ", out queues: " << queues_num_ << ", depth: " << queue_depth_;
  for (size_t i = 0; i < queues_num_; ++i) 
  {
    full_[i] = make_shared<BlockingQueue<shared_ptr<DatumType>>>();
    free_[i] = make_shared<BlockingQueue<shared_ptr<DatumType>>>();
    for (size_t j = 0; j < queue_depth_; ++j) 
    {
      free_[i]->push(make_shared<DatumType>());
    }
  }
  db_source_ = param.data_param().source();
  init_ = make_shared<BlockingQueue<shared_ptr<DatumType>>>();
  StartInternalThread(false, Caffe::next_seed());
}

template<typename DatumType>
FPGAReader<DatumType>::~FPGAReader() 
{
  StopInternalThread();
}

template<typename DatumType>
void FPGAReader<DatumType>::InternalThreadEntry() 
{
  InternalThreadEntryN(0U);
}

template<typename DatumType>
void FPGAReader<DatumType>::InternalThreadEntryN(size_t thread_id) 
{

  shared_ptr<DatumType> init_datum = make_shared<DatumType>();
  cm.fetch(init_datum.get());
  init_->push(init_datum);

  if (!sample_only_) {
    start_reading_flag_.wait();
  }
  cm.rewind();
  size_t skip = skip_one_batch_ ? batch_size_ : 0UL;

  size_t queue_id, ranked_rec, batch_on_solver, sample_count = 0UL;
  shared_ptr<DatumType> datum = make_shared<DatumType>();
  try 
  {
    while (!must_stop(thread_id)) 
    {
      cm.next(datum);
      // See comment below
      ranked_rec = (size_t) datum->record_id() / cm.full_cycle();
      batch_on_solver = ranked_rec * parser_threads_num_ + thread_id;
      queue_id = batch_on_solver % queues_num_;

      if (thread_id == 0 && skip > 0U) 
      {
        --skip;
        continue;
      }

      full_push(queue_id, datum);

      if (sample_only_) 
      {
        ++sample_count;
        if (sample_count >= batch_size_) 
        {
          // sample batch complete
          break;
        }
      }
      datum = free_pop(queue_id);
    }
  } catch (boost::thread_interrupted&) {}
}

template class FPGAReader<PackedData>;

}  // namespace caffe
