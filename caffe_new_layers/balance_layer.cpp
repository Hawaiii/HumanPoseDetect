#include <vector>
#include <iostream>

#include "caffe/layers/balance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// int rand_int(int i){
//   return caffe_rng_rand() % i;
// }

template <typename Dtype>
void BalanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  top[0]->ReshapeLike(*bottom[0]);
  mask_.ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void BalanceLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom){
  // const Dtype* bottom_data = bottom[0]->cpu_data(); //bigscore: num x 16 x h x w
  const Dtype* label = bottom[1]->cpu_data(); //jointmap: num x 16 x h x w
  Dtype* mask_data = mask_.mutable_cpu_data();

  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int npixels = width*height;

  // Write all masks to zero
  for (int i = 0; i < count; i++){
    mask_data[i] = 0;
  }
  for (int batch = 0; batch < num; batch++){
    for (int ijoint = 0; ijoint < channel; ijoint++){
      // Read number of positive samples for each joint
      int pos_count = 0;
      for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
          int idx = batch*count/num+ijoint*npixels + i*width + j;
          if (label[idx] > 0){
            mask_data[idx] = 1;
            pos_count++;
          }
        }
      }
  

    // Set mask for negative samples
      int neg_count = 0;
      while (neg_count < pos_count){
        int select_neg = caffe_rng_rand() % npixels;
        if (mask_data[batch*count/num+ijoint*npixels + select_neg] == 0){
          mask_data[batch*count/num+ijoint*npixels + select_neg] = 1;
          neg_count++;
        }
      }

//    std::cout << "joint:" << ijoint << " neg_count:" << neg_count << std::endl;
    }
  }

}


template <typename Dtype>
void BalanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data(); //bigscore: 1 x 16 x h x w
  const Dtype* label = bottom[1]->cpu_data(); //jointMap: 1 x 16 x h x w
  Dtype* top_data = top[0]->mutable_cpu_data();

  set_mask(bottom);
  Dtype* mask_data = mask_.mutable_cpu_data();

  const int count = bottom[0]->count();
  int cnt = 0;
  for (int i = 0; i < count; i++){
    if (mask_data[i] > 0){
      top_data[i] = bottom_data[i];
     // if(bottom_data[i]!= 0) std::cout << "mask:" << bottom_data[i] << std::endl;
      cnt++;
    } else {
      top_data[i] = label[i];
      //if(bottom_data[i]!= 0) std::cout << bottom_data[i] << std::endl;
    }
    //if(bottom_data[i]!=0) std::cout<<bottom_data[i]<<std::endl;
  }
//  std::cout << cnt << std::endl;
}


template <typename Dtype>
void BalanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* mask_data = mask_.mutable_cpu_data();
  
  const int count = bottom[0]->count();

  for (int i = 0; i < count; i ++) 
  {
    if (mask_data[i] > 0){
      bottom_diff[i] = top_diff[i];
    } else {
      bottom_diff[i] = 0;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(BalanceLayer);
#endif

INSTANTIATE_CLASS(BalanceLayer);
REGISTER_LAYER_CLASS(Balance);

}  // namespace caffe
