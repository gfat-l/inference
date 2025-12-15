#include "alexnet.h"
using namespace std;

Alexnet::Alexnet(){}

int Alexnet::Net(vector<int> &weights,vector<int> &data){
	//std::ofstream fout("input.txt"); // 1
	//std::ofstream fout1("fc2.txt");
	scale = 0;
	weight_scale = 112;
    
	//下面是padding所需的参数和顺序
	//vector <int> &input, vector <int> &input_padding, int length, int width, int channel,int padding flag代表补1.5圈0
    //下面是conv的所有需要的参数和顺序
	//vector<int> &input, vector<int> &C1_value, vector<int> &weights, 
	//int conv1_l, int conv1_w, int conv1_c, int real_width, int real_length,
	//int channel, int conv1_k_s, int stride, int index,int small)
	//下面是pool所有需要的参数和顺序
	//vector<int> &input, vector<int> &M2_value, 
	//int in_l, int in_w, int in_c, int stride, int kernel_size
	//下面是FC所需要的所有参数和顺序
	//vector<int> &input, vector<int> &F_value, vector<int> &weights, int in_size, int out_size, int index_w
	pad(data,input_padding,224,224,3,2, true);
	/*
	for(int i=0; i<input_padding.size();i++){
		fout << input_padding[i] <<"\n";
	}
	fout.close();*/
	ConvLayer(input_padding, input_maxpool1, weights, 55,55,48, 227,227,3, 11, 4, 0, 1);
    MaxpoolLayer(input_maxpool1,input_conv2, 55,55,48, 2, 3);
	
	/*for(int i=0; i<input_conv2.size();i++){
		fout1 << input_conv2[i] <<"\n";
	}
	fout1.close();*/

	pad(input_conv2,conv2_padding,27,27,48,2,false);
	ConvLayer(conv2_padding, input_maxpool2, weights, 27,27,128, 31,31,48, 5, 1, 17472,2);
    MaxpoolLayer(input_maxpool2,input_conv3, 27,27,128, 2, 3);

	pad(input_conv3,conv3_padding,13,13,128,1,false);
	ConvLayer(conv3_padding, input_conv4, weights, 13,13,192, 15,15,128, 3, 1, 171200,3);

	pad(input_conv4,conv4_padding,13,13,192,1,false);
	ConvLayer(conv4_padding, input_conv5, weights, 13,13,192, 15,15,192, 3, 1, 392576,4);

	pad(input_conv5,conv5_padding,13,13,192,1,false);
	ConvLayer(conv5_padding, input_maxpool3, weights, 13,13,128, 15,15,192, 3, 1, 724544,5);
	MaxpoolLayer(input_maxpool3,input_fc1, 13,13,128, 2, 3);
	
	FullyConnLayer(input_fc1, input_fc2, weights, 6*6*128, 2048, 945856, true,6);
	/*for(int i=0; i<input_fc2.size();i++){
		fout1 << input_fc2[i] <<"\n";
	}
	fout1.close();*/
	FullyConnLayer(input_fc2, input_fc3, weights, 2048, 2048, 10385088,true,7);
	FullyConnLayer(input_fc3, input_fc4, weights, 2048, 10, 14581440,false,8);

	
    int max = input_fc4[0];
	int index = 0;
	for(int i=0;i<10;i++)
	{
		if(input_fc4[i]>max){
			max = input_fc4[i];
			index = i;
		}

		cout<<i<<": "<<input_fc4[i]<<endl;
		
	}
	cout<<"index: "<<index<<endl;
	return index;
}


void Alexnet::pad(vector <int> &input, vector <int> &input_padding, int length, int width, int channel, int padding, bool flag){
	int count = 0;
	int real_length;
	int real_width;
	if(flag)
	{
		real_length = length + 3;
	    real_width = width + 3;
	}
	else{
		real_length = length + 2 * padding;
		real_width = width + 2 * padding;
	}
	
	for(int i=0; i<channel;i++)
		for(int j=0; j<real_length;j++)
        	for(int k=0; k<real_width; k++)
        	{
				input_padding.push_back(0);
			}
	//补零
	if(flag){
		for(int i=0; i<channel; i++)
			for(int j=0; j < real_length;j++)
				for(int k=0; k < real_width; k++)
				{
				if(j>=2 && j<length+2 && k>=2 && k<width+2){
					input_padding[i*real_length*real_width + j*real_width +k] = input[count];
					count++; 
				}
				}
	}
	else{
		for(int i=0; i<channel; i++)
			for(int j=0; j < real_length;j++)
				for(int k=0; k < real_width; k++)
				{
				if(j>=padding && j<length+padding && k>=padding && k<width+padding){
					input_padding[i*real_length*real_width + j*real_width +k] = input[count];
					count++; 
				}
				}
	}
}

int Alexnet::Conv_nxn(vector <int> &input, vector <int> &kernel){
	int y;
	int result = 0;
	//cout<<"input_size: "<<input.size()<<endl;
	//cout<<"kernel_size: "<<kernel.size()<<endl;
	for(y = 0; y < input.size(); y++){
		result += input[y] * kernel[y];  
	}
	return result;
}


int Alexnet::relu(int x)
{
    if(x>0)
        return x;
    else
        return 0;
}

//nxn最大池化
int Alexnet::MaxPool_nxn(vector <int> &input){
    int res = 0;
    int i;
    for(i=0;i<input.size();i++){
        if(input[i]>res)
            res = input[i];
    }
    return res;
}

//输入大小为6*28*28
void Alexnet::MaxpoolLayer(vector<int> &input, vector<int> &M2_value, int in_l, int in_w, int in_c, int stride, int kernel_size){
	int k_num,i_y,i_x,matrix_x,matrix_y;
	int count = 0;

	for(k_num = 0; k_num < in_c; k_num++){
		for(i_y = 0; i_y <= in_l-kernel_size; i_y+=stride){
			for(i_x = 0;  i_x <= in_w-kernel_size; i_x+=stride){ //stride=2
				//matrix里面存放data
                vector<int> matrix;
				int index_now = i_x + i_y * in_w + k_num * in_l * in_w;
				for(matrix_y = 0; matrix_y < kernel_size; matrix_y++){
					for(matrix_x = 0; matrix_x < kernel_size; matrix_x++){
						int input_index = index_now + matrix_x + matrix_y * in_w ;
                        matrix.push_back(input[input_index]);
					}
				}
				M2_value.push_back(MaxPool_nxn(matrix));
				matrix.clear();
				count++;
			}
		}
	}
}

//input为6*14*14
void Alexnet::ConvLayer(vector<int> &input, vector<int> &C1_value, vector<int> &weights, 
					int conv1_l, int conv1_w, int conv1_c, int real_width, int real_length,
					int channel, int conv1_k_s, int stride, int index, int layer_num){

	int k_num,nk_num,i_y,i_x,matrix_x,matrix_y;
	int mat_i;
    for(nk_num = 0; nk_num < conv1_c; nk_num++){ //conv1_c个kernel 48
		int bias;
		if(layer_num==1){
			bias = weights[nk_num+index+conv1_k_s*conv1_k_s*channel*conv1_c]*255;
			pre_layer_scale = 255;
		}
		else{
			if(scale>0){
			bias = (weights[nk_num+index+conv1_k_s*conv1_k_s*channel*conv1_c]*pre_layer_scale*weight_scale) /pow(2,scale);
			//bias = weights[nk_num+index+conv1_k_s*conv1_k_s*channel*conv1_c];
			}
			else{
				bias = weights[nk_num+index+conv1_k_s*conv1_k_s*channel*conv1_c]*pre_layer_scale*weight_scale;
			}
		}
		
		for(i_y = 0; i_y < conv1_l; i_y++){ //output l 55
			for(i_x = 0; i_x < conv1_w; i_x++){ //output w 55
				int res = 0;
				int res_total = 0;
				//int index_now1 = i_x + i_y * conv1_w + nk_num * conv1_l * conv1_w;
				int index_now = stride * (i_x + i_y * real_width);
				//输入矩阵的channel
                for(k_num = 0; k_num < channel; k_num++){
					//获得weight的矩阵
                    vector <int> matrix_2;
					vector <int> matrix;
					for(mat_i = 0;mat_i<conv1_k_s*conv1_k_s;mat_i++){
						int weights_index = mat_i + k_num*conv1_k_s*conv1_k_s + nk_num*conv1_k_s*conv1_k_s*channel + index; //index为上面所有层weight
						matrix_2.push_back(weights[weights_index]);  //需要根据weight怎么存的确定
					}
                    //获得data的矩阵
					for(matrix_y = 0; matrix_y <conv1_k_s; matrix_y++){
						for(matrix_x = 0; matrix_x <conv1_k_s; matrix_x++){
							//int matrix_index = matrix_x + matrix_y * conv1_w;
							int input_value_index = index_now + matrix_x + matrix_y * real_width + real_width*real_length*k_num;
							
							matrix.push_back(input[input_value_index]);
						}
					}
                    //求和
					res_total += Conv_nxn(matrix,matrix_2);
					matrix.clear();
					matrix_2.clear();
				}
				C1_value.push_back(res_total + bias) ; //加入bias
			}
		}
	}
	//下一层缩放因子，每一层改变一次
	if(layer_num!=1){
		if(scale>0){
			pre_layer_scale = (pre_layer_scale*weight_scale) /(float)pow(2,scale);
		}
		else{
			pre_layer_scale = (pre_layer_scale*weight_scale);
		}
	}
	int max = C1_value[0];
	for(int in = 0; in < C1_value.size(); in++)
	{
		C1_value[in] = relu(C1_value[in]);
		if(max<C1_value[in])
			max = C1_value[in];
	}
	//cout<<layer_num<<": "<<max<<endl;
	int position = log(max)/log(2);
	scale = position-7;
	if(scale>0){
		for(int in = 0; in < C1_value.size(); in++)
		{
			C1_value[in] = C1_value[in]>>scale;
		}
	}
	
	//cout<<scale<<endl;
	//cout<<layer_num<<": "<<pre_layer_scale<<endl;
}

//in_size=400,out_size=120,index_w是该层第一个权重的为初始位置
void Alexnet::FullyConnLayer(vector<int> &input, vector<int> &F_value, vector<int> &weights, int in_size, int out_size, int index_w,bool isrelu,int layer_num){
	int i_y,i_x;
	for(i_y = 0; i_y < out_size; i_y++){
		int res = 0;
		int bias;
		if(layer_num==1){
			bias = weights[i_y + index_w + out_size*in_size]*255;
			pre_layer_scale = 255;
		}
		else{
			if(scale>0){
				bias = (weights[i_y + index_w + out_size*in_size]*pre_layer_scale*weight_scale)/pow(2,scale);
				//bias = weights[i_y + index_w + out_size*in_size];
			}
			else{
				bias = weights[i_y + index_w + out_size*in_size]*pre_layer_scale*weight_scale;
			}
		}
		//
		for(i_x = 0;  i_x < in_size; i_x++){
			int index = i_x + i_y * in_size + index_w;
			res += input[i_x] * weights[index];
		}
		if(isrelu)
			F_value.push_back(relu(res + bias));
		else
			F_value.push_back(res + bias);
	}

	if(layer_num!=1){
		if(scale>0){
			pre_layer_scale = (pre_layer_scale*weight_scale) /(float)pow(2,scale);
		}
		else{
			pre_layer_scale = pre_layer_scale*weight_scale;
		}
	}
	//量化
	if(isrelu){
		int max = F_value[0];
		for(int in = 0; in < out_size; in++)
		{
			if(max<F_value[in])
				max = F_value[in];
		}
		//cout<<layer_num<<": "<<max<<endl;
		int position = log(max)/log(2);
		scale = position-7;
		if(scale>0){
			for(int in = 0; in < out_size; in++)
			{
				F_value[in] = F_value[in]>>scale;
			}
		}
	}
	
	//cout<<scale<<endl;
	//cout<<layer_num<<": "<<pre_layer_scale<<endl;
	
}

float Alexnet::Softmax_1_8(int input[10],float *probability,int *res){
	int index;
	float sum = 0;
	for(index = 0; index < 10; index++ ){
		probability[index] = expf(input[index]/1000);
		sum += probability[index];
	}
	int max_index = 0;
	for(index = 0; index < 10; index++ ){
			res[index] = probability[index]/sum;
			float res1 = res[index];
			float res2 = res[max_index];
			if(res1 > res2){
				max_index = index;
			}
	}
	return max_index;
}