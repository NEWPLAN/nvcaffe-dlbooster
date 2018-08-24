#include <boost/thread.hpp>
#include <sys/sysinfo.h>

#include "caffe/util/rng.hpp"
#include "caffe/parallel.hpp"
#include "caffe/assist_bp.hpp"
#include <algorithm>
#include <cstdlib>

namespace caffe
{

AssistBP::AssistBP(size_t solver_rank,
                  const vector<shared_ptr<LayerBase>> train_layer,
                  const vector<vector<Blob*> >& top,
                  const vector<vector<bool> >& need,
                  const vector<vector<Blob*> >& bottom)
  : InternalThread(Caffe::current_device(), solver_rank, 1U, false),
    solver_rank_(solver_rank),
    _layer(train_layer),
    _top_vecs(top),
    _bottom_need_backward(need),
    _bottom_vecs(bottom)
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
      int i = en_queue->pop();
      /*DLOG(INFO)<<"In device: " << Caffe::current_device() <<", receive: " << out_;*/
      _layer[i]->Backward(_top_vecs[i], _bottom_need_backward[i], _bottom_vecs[i]);
      //boost::this_thread::sleep(boost::posix_time::seconds(2));
      de_queue->push(i);
    }
  }
  catch (boost::thread_interrupted&) {}
}
}  // namespace caffe
