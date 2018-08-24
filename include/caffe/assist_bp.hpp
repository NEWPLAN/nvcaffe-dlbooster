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
  AssistBP(size_t solver_rank);
  virtual ~AssistBP();

protected:
  void InternalThreadEntry() override;
  void InternalThreadEntryN(size_t thread_id) override;

  const size_t  solver_rank_;

  shared_ptr<BlockingQueue<int>> en_queue;
  shared_ptr<BlockingQueue<int>> de_queue;


  DISABLE_COPY_MOVE_AND_ASSIGN(AssistBP);
};


}  // namespace caffe

#endif  // CAFFE_ASSIST_BP_HPP_
