#include<vector>
#include "alexnet.h"
#include<iostream>
#include<fstream>
#include "main.h"
#include <sstream>
using namespace std;

int main()
{
    std::ifstream fin_weight,fin_label;
    fin_weight.open("weight_quaz_alexnet.txt",std::ios::in);
    fin_label.open("D:/deep-learning-for-image-processing-master/pytorch_classification/Test2_alexnet/data/label.txt",std::ios::in);
    if(!fin_weight.is_open() | !fin_label.is_open())
    {
        std::cout<<"cannot open the file"; 
        return 1;
    }
    char line[1024] = {0};
    vector<int> weight;
    vector<int> label;
    //从文件中提取“行”
    while(fin_weight.getline(line,sizeof(line)))
    {
        std::stringstream word(line);
        int num;
        while(word>>num)
            weight.push_back(num);
    }
    while(fin_label.getline(line,sizeof(line)))
    {
        std::stringstream word(line);
        int num;
        while(word>>num)
            label.push_back(num);
    }

    char filename_body[10];
    vector <int> pred;
    int num=0;
    int count = 0;
    for(int i=2000; i<3270; i++){
        cout<<"num:"<<num<<endl;
        cout<<"count:"<<count<<endl;
        Alexnet net;
        vector<float> data;
        vector<int> data_quaz;
        std::ifstream fin_data;
        char filename[100] = "D:\\deep-learning-for-image-processing-master\\pytorch_classification\\Test2_alexnet\\data\\data_";
        memset(filename_body, 0, sizeof(filename_body));
        itoa(i, filename_body, 10);
        strcat(filename,filename_body);
        char filename_tail[5] = ".txt";
        strcat(filename,filename_tail);
        //cout<<filename<<endl;

        fin_data.open(filename,std::ios::in);
        if(!fin_data.is_open())
        {
            std::cout<<"cannot open the file"; 
            return 1;
        }

        while(fin_data.getline(line,sizeof(line)))
        {
            std::stringstream word(line);
            float num;
            while(word>>num)
                data.push_back(num);
        }
        for(int j= 0; j<data.size();j++)
        {
            int d = data[j]*255;
            data_quaz.push_back(d);
            //cout<<j<<" : "<<data_quaz[j]<<endl;
        }
        pred.push_back(net.Net(weight,data_quaz));
        if(pred[i-2000] == label[i])
        {
            count++;
        }
        num++;
    }
    /*for(int i=0; i<2000; i++){
        if(label[i] == pred[i]){
            count++;
        }
    }*/
    float accuracy = count/1270.0;
    cout<<"accuracy: "<<accuracy<<endl;
    
    
    return 0;
}
