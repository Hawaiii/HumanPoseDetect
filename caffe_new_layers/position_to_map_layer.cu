#include <vector>

#include "caffe/layers/position_to_map_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PositionToMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      Forward_cpu(bottom, top);
}




INSTANTIATE_LAYER_GPU_FUNCS(PositionToMapLayer);

}  // namespace caffe
~                                                                                     
~  
