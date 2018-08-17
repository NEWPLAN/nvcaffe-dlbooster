#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/fpga_data_layer.hpp"
#include "caffe/parallel.hpp"

namespace caffe
{

template<typename Ftype, typename Btype>
FPGADataLayer<Ftype, Btype>::FPGADataLayer(const LayerParameter& param, size_t solver_rank)
  : BasePrefetchingDataLayer<Ftype, Btype>(param, solver_rank),
    cache_(param.data_param().cache()),
    shuffle_(param.data_param().shuffle())
{
  LOG(INFO) << " IS in auto mode ?: " << this->auto_mode_ ? "Yes" : "No";
  init_offsets();
}

template<typename Ftype, typename Btype>
void FPGADataLayer<Ftype, Btype>::init_offsets()
{
  CHECK_EQ(this->transf_num_, this->threads_num());
  random_vectors_.resize(this->transf_num_);
  for (size_t i = 0; i < this->transf_num_; ++i)
  {
    if (!random_vectors_[i])
    {
      random_vectors_[i] = make_shared<TBlob<unsigned int>>();
    }
  }
}

template<typename Ftype, typename Btype>
FPGADataLayer<Ftype, Btype>::~FPGADataLayer()
{
  this->StopInternalThread();
}

template<typename Ftype, typename Btype>
void FPGADataLayer<Ftype, Btype>::InitializePrefetch()
{
  if (layer_inititialized_flag_.is_set())
  {
    return;
  }
  CHECK_EQ(this->threads_num(), this->transf_num_);
  LOG(INFO) << this->print_current_device() << " Transformer threads: "
            << this->transf_num_ << (auto_mode ? " (auto)" : "(fixed)");
  layer_inititialized_flag_.set();
}

template<typename Ftype, typename Btype>
size_t FPGADataLayer<Ftype, Btype>::queue_id(size_t thread_id) const
{
  return thread_id % this->queues_num_;
}
template<typename Ftype, typename Btype>
void FPGADataLayer<Ftype, Btype>::start_reading()
{
  train_reader->start_reading();
  //reader_->start_reading();
}

//newplan added
template<typename Ftype, typename Btype>
void FPGADataLayer<Ftype, Btype>::DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
  const LayerParameter& param = this->layer_param();
  const int batch_size = param.data_param().batch_size();
  const bool shuffle = shuffle_ && this->phase_ == TRAIN;

  //newplan added
  const size_t new_height = param.data_param().new_height();
  const size_t new_width = param.data_param().new_width();
  const size_t new_channel = param.data_param().new_channel();

  if (this->auto_mode_)
  {
    LOG(INFO) << "This FPGADataLayer is in auto mode.";
  }
  //LOG parameters
  {
    LOG(INFO) << " in FPGADataLayer parameters:" << std::endl
              << "batch size: " << batch_size << std::endl
              << "height: " << new_height <<  std::endl
              << "width: " << new_width <<  std::endl
              << "channel: " << new_channel;
    CHECK_GT(new_height, 0);
    CHECK_GT(new_width, 0);
    CHECK_GT(new_channel, 0);
  }
  if (this->rank_ == 0 && this->phase_ == TRAIN)
  {
    if (!train_reader)
    {
      train_reader = std::make_shared<FPGAReader<PackedData>>(param,
                     Caffe::solver_count(),
                     this->rank_,
                     batch_size,
                     shuffle,
                     this->phase_ == TRAIN);
      train_reader->start_reading();
      LOG(INFO) << "create train reader....";

      FPGADataLayer::train_reader_ = train_reader;
    }
  }
  else if (this->rank_ == 0 && this->phase_ == TEST)
  {
    LOG(INFO) << "IN Root rank and test phase...";
  }
  LOG(INFO) << "out of sides. ";

  train_reader = FPGADataLayer::train_reader_;

  // Read a data point, and use it to initialize the top blob.
  this->ResizeQueues();
  init_offsets();

  // newplan added
  if (this->phase_ == TRAIN)
  {
    LOG(INFO) << "IN TRAIN phase...";
    const int cropped_height = param.transform_param().crop_size();
    const int cropped_width = param.transform_param().crop_size();
    //Packing packing = NHWC;  // OpenCV
    vector<int> top_shape = {(int)batch_size, (int)new_channel, cropped_height, cropped_width};
    top[0]->Reshape(top_shape);

    if (this->is_gpu_transform())
    {
      CHECK(Caffe::mode() == Caffe::GPU);
      LOG(INFO) << this->print_current_device() << " Transform on GPU enabled";
      tmp_gpu_buffer_.resize(this->threads_num());
      for (int i = 0; i < this->tmp_gpu_buffer_.size(); ++i)
      {
        this->tmp_gpu_buffer_[i] = make_shared<GPUMemory::Workspace>();
      }
    }
    // label
    vector<int> label_shape(1, batch_size);
    if (this->output_labels_)
    {
      vector<int> label_shape(1, batch_size);
      top[1]->Reshape(label_shape);
    }
    this->batch_transformer_->reshape(top_shape, label_shape, this->is_gpu_transform());
    LOG(INFO) << this->print_current_device() << " Output data size: "
              << top[0]->num() << ", "
              << top[0]->channels() << ", "
              << top[0]->height() << ", "
              << top[0]->width();
  }
}

template<typename Ftype, typename Btype>
void FPGADataLayer<Ftype, Btype>::load_batch(Batch* batch, int thread_id, size_t queue_id)
{
  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const bool use_gpu_transform = this->is_gpu_transform();
  const int cropped_height = this->layer_param_.transform_param().crop_size();
  const int cropped_width = this->layer_param_.transform_param().crop_size();
  const int new_height = this->layer_param_.data_param().new_height();
  const int new_width = this->layer_param_.data_param().new_width();
  const int new_channel = this->layer_param_.data_param().new_channel();

  Packing packing = NHWC;  // OpenCV

  //infer shape of blobs
  vector<int> top_shape = {batch_size, new_channel, cropped_height, cropped_width};
  if (top_shape != batch->data_->shape())
  {
    batch->data_->Reshape(top_shape);
  }
  size_t datum_sizeof_element = 0UL;
  int datum_len = top_shape[1] * top_shape[2] * top_shape[3];
  size_t datum_size = 0UL;

  if (use_gpu_transform)
  {
    CHECK_GT(datum_len, 0);
    CHECK_LE(sizeof(uint8_t), sizeof(Ftype));
    datum_sizeof_element = sizeof(uint8_t);

    vector<int> random_vec_shape(1, batch_size * 3);
    random_vectors_[thread_id]->Reshape(random_vec_shape);
    datum_size = datum_len * datum_sizeof_element;
  }
  if (this->output_labels_)
  {
    batch->label_->Reshape(vector<int>(1, batch_size));
  }
  Ftype* top_label = this->output_labels_ ?
                     batch->label_->template mutable_cpu_data_c<Ftype>(false) : nullptr;

    void* dst_gptr = nullptr;
    Btype* dst_cptr = nullptr;
    if (use_gpu_transform)
  {
    size_t buffer_size = top_shape[0] * top_shape[1] * new_height * new_width;
    tmp_gpu_buffer_[thread_id]->safe_reserve(buffer_size);
    dst_gptr = tmp_gpu_buffer_[thread_id]->data();
  }
  else
  {
    dst_cptr = batch->data_->template mutable_cpu_data_c<Btype>(false);
  }

  CHECK(train_reader != nullptr);

  PackedData* abc = nullptr;
  int cycles_index = 0;

  while (!train_reader->pop_packed_data(abc))
  {
    LOG_EVERY_N(WARNING, 10) << "Something wrong in pop queue.";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  {
    if (top_label != nullptr)
    {
      for (size_t label_index = 0; label_index < batch_size; label_index++)
        top_label[label_index] = 1;
    }

    if (use_gpu_transform)
    {
      cudaStream_t stream = Caffe::thread_stream(Caffe::GPU_TRANSF_GROUP);
      size_t buffer_size = top_shape[0] * top_shape[1] * new_height * new_width;

      CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst_gptr), abc->data_, buffer_size, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      for (size_t item_id = 0; item_id < batch_size; item_id++)
      {
        this->bdt(thread_id)->Fill3Randoms(&random_vectors_[thread_id]->
                                           mutable_cpu_data()[item_id * 3]);
      }
    }
    else
    {
      LOG(FATAL) << "require enabling transform GPU";
    }
    if (use_gpu_transform)
    {
      this->fdt(thread_id)->TransformGPU(top_shape[0], top_shape[1],
                                         new_height,  // non-crop
                                         new_width,  // non-crop
                                         datum_sizeof_element,
                                         dst_gptr,
                                         batch->data_->template mutable_gpu_data_c<Ftype>(false),
                                         random_vectors_[thread_id]->gpu_data(), true);
      packing = NCHW;
    }
  }
  string a(abc->data_);
  sprintf(abc->data_, "From consumer thread id : %u", lwp_id());
  while (!train_reader->recycle_packed_data(abc))
  {
    LOG_EVERY_N(WARNING, 10) << "Something wrong in push queue.";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  LOG_EVERY_N(INFO, 100) << "loading from pixel queue:" << a;

  batch->set_data_packing(packing);
  batch->set_id(123);
}

INSTANTIATE_CLASS_FB(FPGADataLayer);
REGISTER_LAYER_CLASS_R(FPGAData);

}  // namespace caffe
