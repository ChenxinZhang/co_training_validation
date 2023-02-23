import os
# import sys
# envpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0,envpath)
from run import train_model
import random
random.seed(5)
from importlib import import_module
import torch
import utils 
import utils_fasttext

# 数据准备
def init_dataset(datapath,rate:int):
    save_dir = os.path.dirname(datapath)
    filename = os.path.basename(datapath).split(".")[0]
    with open(datapath,"r",encoding = "utf-8") as f:
        data = f.readlines()
        random.shuffle(data)
        new_data = data[:int(len(data)*rate)]
        to_be_predicted_data = data[int(len(data)*rate):]
    new_data_0 = new_data[:len(new_data)//3]
    new_data_1 = new_data[len(new_data)//3:2*len(new_data)//3]
    new_data_2 = new_data[2*len(new_data)//3:]
    dataset_list = [new_data_0,new_data_1,new_data_2]
    for index in range(3):        
        newdata_path = os.path.join(save_dir, filename + str(index) + "_val.txt")
        with open(newdata_path,"w",encoding = "utf-8") as f:
            print("newdata_path:")
            f.writelines(dataset_list[index])
    to_be_predicted_filepath = os.path.join(save_dir,"to_be_predicted.txt")
    to_be_predicted_data_withoutlabel = [line.strip().split('\t')[0] for line in to_be_predicted_data]
    with open(to_be_predicted_filepath,"w",encoding = "utf-8") as f:
            print("newdata_path:")
            f.writelines(to_be_predicted_data_withoutlabel)



def label(model,data_iter,threshold):
    model.eval()
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
    
    
        
# 协同训练
 ## 训练模型
 ### 模型不需要保存
 ## 打伪标签
 ## 标签修正
 ## 数据再准备
 
# 生成最终模型
if __name__ == '__main__':
    # 准备一份有标签数据集和无标签数据集
    # 将有标签数据集分成三份
    model_name = ["TextCNN"]# ,"TextRNN","FastText"]
    data_filepath_list = ["D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/train.txt","D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/test.txt","D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/dev.txt"]
    init_dataset(data_filepath_list[0],0.3)
    data_path = {"train_0":"D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/train0_val.txt",
                 "train_1":"D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/train1_val.txt",
                 "train_2":"D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/train2_val.txt",
                 "test":"D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/test.txt",
                 "dev":"D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/dev.txt"                
                 }    
    data_path0 = {"test":data_path["test"],"dev":data_path["dev"],"train":data_path["train_0"]}
    data_path1 = {"test":data_path["test"],"dev":data_path["dev"],"train":data_path["train_1"]}
    data_path2 = {"test":data_path["test"],"dev":data_path["dev"],"train":data_path["train_2"]}
    data_path_list = [data_path0,data_path1,data_path2]
    train_parameters = {}
    # # 分别采用fast、cnn、rnn进行编码形成三种表征
    # # 三份数据分别进行三种表征训练出九种分类模型
    
    for model in model_name:
        for dataset in data_path_list:
            config = train_model(model,dataset)
            train_parameters[model+dataset["train"].split("/")[-1].split(".")[-2]] = config
       
    print("train_parameters:",train_parameters)   
    # # 读取模型
    # TextCNN = import_module('models.' + 'TextCNN')
    # # TextRNN = import_module('models.' + 'TextRNN')
    # # FastText = import_module('models.' + 'FastText')
    # # print("TextCNN",TextCNN.Model.__dict__)
    # # TextCNN_state_dict=torch.load("D:\\picc\\WorkDirectory-master\\NPS\\Chinese-Text-Classification-Pytorch-master\\THUCNews\\saved_dict\\TextCNN.ckpt") #加载模型，如果只有数值就只会加载模型数据，如果有字典，则会加载模型数据和字典数据
    # # print("TextCNN_state_dict",TextCNN_state_dict)

    TextCNN = TextCNN.Model(train_parameters["TextCNN"]).to(train_parameters["TextCNN"].device)
    TextCNN.load_state_dict(torch.load("D:\\picc\\WorkDirectory-master\\NPS\\Chinese-Text-Classification-Pytorch-master\\THUCNews\\saved_dict\\TextCNN.ckpt"))  #返回是否成功
    TextCNN.eval()
    
    
    # 读取需要预测的数据
    
    # 进行预测
    
    # 将预测数据打上标签后写入文本
    
    # 进行投票
    
    # 将数据加入到有标签数据集后进行训练，用一开始有标签的数据进行训练再次对数据集预测剔除可信性度不高的伪标签
    
    # 迭代直到无标签数据集不在减少，或是迭代到一定轮次之后停止训练
    
         
    