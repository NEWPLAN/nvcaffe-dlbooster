#ifndef CAFFE_ASSIST_BP_HPP_
#define CAFFE_ASSIST_BP_HPP_

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/thread_pool.hpp"

//#define MAX_GPU_PER_MACHINE_SUPPORT 32

namespace caffe
{


class AssistBP : public InternalThread
{

public:
  AssistBP(size_t solver_rank,
            const vector<shared_ptr<LayerBase>> train_layer,
            const vector<vector<Blob*> >&top,
            const vector<vector<bool> >& need,
            const vector<vector<Blob*> >& bottom,
            const vector<int>& param_owners,
            const map<pair<int, int>, int>& layer_index_params,            
            const vector<int>& learnable_param_ids,
            const vector<shared_ptr<Blob>>& learnable_params,
            const vector<Type>& learnable_types,
            const vector<BlockingQueue<int>>& reduction_queue
            );
  virtual ~AssistBP();
  shared_ptr<BlockingQueue<int>> en_queue;
  shared_ptr<BlockingQueue<int>> de_queue;

protected:
  void InternalThreadEntry() override;
  void InternalThreadEntryN(size_t thread_id) override;

  const size_t  solver_rank_;
  vector<shared_ptr<LayerBase>> _layer;
  vector<vector<Blob*> > _top_vecs;
  vector<vector<bool> > _bottom_need_backward;
  vector<vector<Blob*> > _bottom_vecs;

  map<pair<int, int>, int>& _layer_index_params;
  const vector<int>& _param_owners;
  vector<int>& _learnable_param_ids;
  vector<shared_ptr<Blob>>& _learnable_params;
  vector<Type>& _learnable_types;
  vector<BlockingQueue<int>>& _reduction_queue;

  DISABLE_COPY_MOVE_AND_ASSIGN(AssistBP);
};


}  // namespace caffe

#endif  // CAFFE_ASSIST_BP_HPP_
