#ifndef __LENET5_H__
#define __LENET5_H__

#include <cstring>
#include<vector>
#include<math.h>
#include<cmath>
#include <iostream>
#include<string>
#include<fstream>
using namespace std;

class Alexnet
{
    public:
        Alexnet();
        int Net(vector<int> &weight1,vector<int> &data);
        virtual ~ Alexnet() {};
        int Conv_nxn(vector <int> &input, vector <int> &kernel);
        int relu(int x);
        int MaxPool_nxn(vector<int> &input);
        
        void pad(vector <int> &input, vector <int> &input_padding, int length, int width, int channel, int padding, bool flag);
        void ConvLayer(vector<int> &input, vector<int> &C1_value, vector<int> &weights, int conv1_l, int conv1_w, int conv1_c, 
                        int real_width, int real_length, int channel, int conv1_k_s, int stride, int index,int layer_num);
        void MaxpoolLayer(vector<int> &input, vector<int> &M2_value, int in_c, int in_l, int in_w, int stride, int kernel_size);
        void FullyConnLayer(vector<int> &input, vector<int> &F_value, vector<int> &weights,int in_size, int out_size, int index_w,bool isrelu,int layer_num);
        
        //void ConvLayer_3(int input[1176],int *C3_value,vector<int> &weights);
        //void MaxpoolLayer_4(int input[1600],int *M4_value);
        
        //void FullyConnLayer_6(int input[120],int *F6_value,vector<int> &weights);
        //void FullyConnLayer_7(int input[84],int *F7_value,vector<int> &weights);
        float Softmax_1_8(int input[10],float *probability,int *res);

        vector <int> input_padding;
        vector <int> input_maxpool1;

        vector <int> conv2_padding;
        vector <int> input_conv2;
        vector <int> input_maxpool2;

        vector <int> conv3_padding;
        vector <int> input_conv3;

        vector <int> conv4_padding;
        vector <int> input_conv4;

        vector <int> conv5_padding;
        vector <int> input_conv5;
        vector <int> input_maxpool3;
        
        vector <int> input_fc1;
        vector <int> input_fc2;
        vector <int> input_fc3;
        vector <int> input_fc4;
        //int weights[61706];
        float probability[10];
        int result;
        int scale; //每层缩放的大小（右移位数）
        int index;
        int weight_scale;
        float pre_layer_scale; //记录上一层数据的缩放因子
        /*
        int length;
	    int width;
	    int padding;
	    int channel;
        int real_length;
	    int real_width;
        int conv1_l;
	    int conv1_w;
	    int conv1_c;
        int conv1_p; //padding
        int conv1_k_s; //kernel_size
        */



};
#endif