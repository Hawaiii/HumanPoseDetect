#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

#include "caffe/layers/position_to_map_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PositionToMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  CHECK(bottom[1]->channels()%3 == 0);
  const int channels = bottom[1]->channels() / 3;  //the number of joints
  top_shape.push_back(bottom[0]->num());       //batch size
  top_shape.push_back(channels);
  top_shape.push_back(bottom[0]->height());    //image height
  top_shape.push_back(bottom[0]->width());     //image width
  //for (int i=0; i<top_shape.size();++i)
  //std::cout<<top_shape[i]<<std::endl;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PositionToMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const float sigma = this->layer_param_.gaussian_param().std(); //standard deviation of gaussian
  const int halfWindowSize = this->layer_param_.gaussian_param().half_window_size(); // windowSize=2*halfWindowSIze+1
  
  const Dtype* bottom_joint = bottom[1]->cpu_data();
  const int joint_count = bottom[1]->count();
  const int joint_num = bottom[1]->num();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = top[0]->count();       // count= batchSize*channels*width*height
  const int num = top[0]->num();           // get batch size
  const int channels = top[0]->channels(); // get number of channels = number of joints
  const int height = top[0] -> height(); // get image height
  const int width = top[0] -> width();   // get image width
  
  //std::cout<<1<<std::endl;

  for (int i = 0; i < count; ++i)
  {
    top_data[i]=0;
  } 

  //std::cout<<2<<std::endl;
 
  for (int i=0; i<num; ++i)         // loop for images in a batch
  {
    for (int j=0; j<channels; ++j)   //loop for joints in an image
    {
      bool visible = (bottom_joint[i*(joint_count/joint_num)+j*3+2]!=-1);
      if (visible)
      {
        //assign values according to the joint location and gaussian value
        int x=bottom_joint[i*(joint_count/joint_num)+j*3];   // NOTE: x indicates column and y indicates row 
        int y=bottom_joint[i*(joint_count/joint_num)+j*3+1];   // Here I assume that x indicates column, starts from 1, and is the second value in joint txt, check with Hawaii
        for (int h=std::max(y-halfWindowSize-1,0); h<std::min(y+halfWindowSize,height); ++h)   //loop for each element in gaussian window
        {
          for (int w=std::max(x-halfWindowSize-1,0); w<std::min(x+halfWindowSize,width); ++w)
           {
             double value=100/(pow(sigma,2)*2*M_PI)*exp((-pow((w-x),2)-pow((h-y),2))/(2*pow(sigma,2))); 
              top_data[i*count/num+j*width*height+h*width+w]=value;
           }
        }    
       }
     } 
   }
   //std::cout<<3<<std::endl;
}



 
#ifdef CPU_ONLY
STUB_GPU(PositionToMapLayer);
#endif


INSTANTIATE_CLASS(PositionToMapLayer);
REGISTER_LAYER_CLASS(PositionToMap);

}  // namespace caffe



 
