#include <torch/torch.h>

template <typename scalar_t>
void nms_cuda_compute(int boxes_num,
                      const scalar_t* dets,
                      const scalar_t* scores,
                      int64_t* keep,
                      int64_t* num_to_keep,
                      float iou_threshold,
                      cudaStream_t stream) {
  // Implementation of nms_cuda_compute function
}

void nms_cuda(int boxes_num,
              const torch::Tensor& dets,
              const torch::Tensor& scores,
              torch::Tensor& keep,
              torch::Tensor& num_to_keep,
              float iou_threshold,
              cudaStream_t stream) {
  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_cuda", [&] {
    nms_cuda_compute<scalar_t>(boxes_num,
                              dets.data_ptr<scalar_t>(),
                              scores.data_ptr<scalar_t>(),
                              keep.data_ptr<int64_t>(),
                              num_to_keep.data_ptr<int64_t>(),
                              iou_threshold,
                              stream);
  });
} 