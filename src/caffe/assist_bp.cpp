#include <boost/thread.hpp>
#include <sys/sysinfo.h>

#include "caffe/util/rng.hpp"
#include "caffe/parallel.hpp"
#include "caffe/assist_bp.hpp"
#include <algorithm>
#include <cstdlib>

namespace caffe
{

AssistBP::AssistBP(size_t solver_rank)
  : InternalThread(Caffe::current_device(), solver_rank, 1U, false),
    solver_rank_(solver_rank)
{
  en_queue=make_shared<BlockingQueue<int>>();
  de_queue=make_shared<BlockingQueue<int>>();
  StartInternalThread(true, Caffe::next_seed());
}

AssistBP::~AssistBP()
{
  StopInternalThread();
  LOG(INFO) << "AssistBP goodbye....";
}

void AssistBP::InternalThreadEntry()
{
  InternalThreadEntryN(0U);
}

void AssistBP::InternalThreadEntryN(size_t thread_id)
{
  try
  {
    while (!must_stop(thread_id))
    {
      int out_ = en_queue.pop();
      LOG(INFO)<<"In device: " << Caffe::current_device() <<", receive: " << out;
      boost::this_thread::sleep(boost::posix_time::seconds(2));
      de_queue.push(out_);
    }
  }
  catch (boost::thread_interrupted&) {}
}
}  // namespace caffe
