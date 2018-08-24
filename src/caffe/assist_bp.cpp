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
  LOG(INFO)<<"In device: " << Caffe::current_device();
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
      LOG(INFO)<<"In device: " << Caffe::current_device();
      boost::this_thread::sleep(boost::posix_time::seconds(2));
    }
  }
  catch (boost::thread_interrupted&) {}
}
}  // namespace caffe
