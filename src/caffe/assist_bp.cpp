#include <boost/thread.hpp>
#include <sys/sysinfo.h>

#include "caffe/util/rng.hpp"
#include "caffe/parallel.hpp"
#include "caffe/assist_bp.hpp"
#include <algorithm>
#include <cstdlib>

namespace caffe
{

AssistBP::AssistBP(size_t solver_rank, shared_ptr<Net> net)
    : InternalThread(Caffe::current_device(), solver_rank, 1U, false),
      solver_rank_(solver_rank),
      _net(net)
{
  {
    _layer = _net->layers_ptr();
    _top_vecs = _net->top_vecs_ptr();
    _bottom_need_backward = _net->bottom_need_backward_ptr();
    _bottom_vecs = _net->bottom_vecs_ptr();
    _param_owners = _net->param_owners_ptr();
    _layer_index_params = _net->layer_index_params_ptr();
    _learnable_param_ids = _net->learnable_param_ids_ptr();
    _learnable_params = _net->learnable_params_ptr();
    _learnable_types = _net->learnable_types_ptr();
    _reduction_queue = _net->reduction_queue_ptr();
  }
#ifdef subthread_assist_bp   
  if(tp==nullptr)
  tp=make_shared<ThreadPool>(1);
#endif
  en_queue = make_shared<BlockingQueue<int>>();
  de_queue = make_shared<BlockingQueue<int>>();
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
#ifdef __DEBUG_BP
  CPUTimer cp;
  cp.Start();
  int num = -1;
#endif
  try
  {
    while (!must_stop(thread_id))
    {
      int i = en_queue->pop();

#ifdef subthread_assist_bp
      int k = -2;
      if(i > 0)
      {
        DLOG(INFO) << "before pop";
        k = en_queue->pop();
        DLOG(INFO)<< "after pop " << k;
        if(k>=0)
        {
          tp->runTask([&,this]()
          {
            DLOG(INFO)<<"run..." << (*_layer)[k]->name();
            if ((*_layer)[k]->has_Backward_w())
            {
              (*_layer)[k]->Backward_gpu_weight((*_top_vecs)[k], (*_bottom_need_backward)[k], (*_bottom_vecs)[k],1);
            }
          });
        }
      }
#endif
          
#ifdef __DEBUG_BP
      if(num<=i)
      {
        LOG(INFO)<<"start, " << num;
        num = i;cp.Start();
      }
#endif
      LOG_EVERY_N(INFO, 1000) << "recv from solver " << rank_ << " ->: " << (*_layer)[i]->name();


      if (i >= 0)
      {
        if ((*_layer)[i]->has_Backward_w())
        {
          (*_layer)[i]->Backward_gpu_weight((*_top_vecs)[i], (*_bottom_need_backward)[i], (*_bottom_vecs)[i]);
        }
        for (int j = 0; j < (*_layer)[i]->blobs().size(); ++j)
        {
          if ((*_layer)[i]->skip_apply_update(j))
            continue;

          const int param_id = (*_layer_index_params)[make_pair(i, j)];
          if ((*_param_owners)[param_id] < 0)
          {
            const int lparam_id = (*_learnable_param_ids)[param_id];
            int t = (int)(*_learnable_params)[lparam_id]->diff_type();
            for (int type_id = 0; type_id < (*_learnable_types).size(); ++type_id)
            {
              if (t == (*_learnable_types)[type_id])
              {
                (*_reduction_queue)[type_id]->push(lparam_id);
                break;
              }
            }
          } // leave it to the owner otherwise
        }
#ifdef subthread_assist_bp      
        if(k>=0)
        {
          DLOG(INFO)<< "before complete : " <<k;
          tp->waitWorkComplete();
          DLOG(INFO)<< "after complete : " <<k;
          for (int j = 0; j < (*_layer)[k]->blobs().size(); ++j)
          {
            if ((*_layer)[k]->skip_apply_update(j))
              continue;

            const int param_id = (*_layer_index_params)[make_pair(k, j)];
            if ((*_param_owners)[param_id] < 0)
            {
              const int lparam_id = (*_learnable_param_ids)[param_id];
              int t = (int)(*_learnable_params)[lparam_id]->diff_type();
              for (int type_id = 0; type_id < (*_learnable_types).size(); ++type_id)
              {
                if (t == (*_learnable_types)[type_id])
                {
                  (*_reduction_queue)[type_id]->push(lparam_id);
                  break;
                }
              }
            } // leave it to the owner otherwise
          }
        }
        else if(k == -1)
        {
          DLOG(INFO)<< "on layer: " <<k;
          for (int type_id = 0; type_id < (*_learnable_types).size(); ++type_id)
          {
            (*_reduction_queue)[type_id]->push(-1);
          }
        }
#endif
      }

      else if (i == -1)
      {
        for (int type_id = 0; type_id < (*_learnable_types).size(); ++type_id)
        {
          (*_reduction_queue)[type_id]->push(-1);
        }
#ifdef __DEBUG_BP
        cp.Stop();
        LOG(INFO) << " back over weight: "<< cp.MilliSeconds()<< " ms";
#endif
      }
    }
  }
  catch (boost::thread_interrupted &)
  {
  }
}
} // namespace caffe
