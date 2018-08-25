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
                  const vector<vector<Blob*> >&top,
                  const vector<vector<bool> >& need,
                  const vector<vector<Blob*> >& bottom,
                  const vector<int>& param_owners,
                  const map<pair<int, int>, int>& layer_index_params,            
                  const vector<int>& learnable_param_ids,
                  const vector<shared_ptr<Blob>>& learnable_params,
                  const vector<Type>& learnable_types,
                  const BlockingQueue<int>*& reduction_queue
                  )
  : InternalThread(Caffe::current_device(), solver_rank, 1U, false),
    solver_rank_(solver_rank),
    _layer(train_layer),
    _top_vecs(top),
    _bottom_need_backward(need),
    _bottom_vecs(bottom),
    _layer_index_params(layer_index_params),
    _param_owners(param_owners),
    _learnable_param_ids(learnable_param_ids),
    _learnable_params(learnable_params),
    _learnable_types(learnable_types),
    _reduction_queue(reduction_queue)
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

      if(i >= 0)
      {
        if(_layer[i]->has_Backward_w())
        {
          _layer[i]->Backward_gpu_weight(_top_vecs[i], _bottom_need_backward[i], _bottom_vecs[i]);
        }
        for (int j = 0; j < _layer[i]->blobs().size(); ++j) 
        {
          if (_layer[i]->skip_apply_update(j)) continue;

          const int param_id = _layer_index_params[make_pair(i, j)];
          if (_param_owners[param_id] < 0) 
          {
            const int lparam_id = _learnable_param_ids[param_id];
            int t = (int)_learnable_params[lparam_id]->diff_type();
            for (int type_id = 0; type_id < _learnable_types.size(); ++type_id) 
            {
              if (t == _learnable_types[type_id]) 
              {
                _reduction_queue[type_id].push(lparam_id);
                break;
              }
            }
          }  // leave it to the owner otherwise
        }
      }
      else if(i == -1)
      {
        for (int type_id = 0; type_id < _learnable_types.size(); ++type_id) 
        {
          _reduction_queue[type_id].push(-1);
        }
      }
    }
  }
  catch (boost::thread_interrupted&) {}
}
}  // namespace caffe
