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
  const float sigma = this->layer_param_.gaussian_param().std();
  const int halfWindowSize = this->layer_param_.gaussian_param().half_window_size();
  
  const Dtype* bottom_joint = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();

  for (int i = 0; i < count; ++i)
  {
    top_data[i]=0;
  }
   
  for (int i=0; i<num; ++i)
  {
   bool visible = (bottom_joint[i*3]==1)
   if (visible)
   {//assign value according to the joint location and gaussian value
    int x=bottom_joint[i*3+1];   // NOTE: need to check x is the row or coloumn ?!
    int y=bottom_joint[i*3+2];
    
    


   } 

  }

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



 
