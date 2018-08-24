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
            const vector<vector<Blob*> >& bottom);
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

  DISABLE_COPY_MOVE_AND_ASSIGN(AssistBP);
};


}  // namespace caffe

#endif  // CAFFE_ASSIST_BP_HPP_
