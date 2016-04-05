#include <vector>

#include "caffe/layers/position_to_map_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PositionToMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  const int channel = bottom[1]->count() / 3;  //the number of joints
  top_shape.push_back(bottom[0]->num());       //batch size
  top_shape.push_back(channel);
  top_shape.push_back(bottom[0]->height());    //image height
  top_shape.push_back(bottom[0]->width());     //image width
  
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PositionToMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  

}

template <typename Dtype>
void PositionToMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {



}



 
#ifdef CPU_ONLY
STUB_GPU(PositionToMapLayer);
#endif


INSTANTIATE_CLASS(DropOutputLayer);
REGISTER_LAYER_CLASS(DropOutput);

}  // namespace caffe



 
