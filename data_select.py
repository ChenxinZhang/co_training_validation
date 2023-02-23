# coding: UTF-8
import os
with open('D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/dev.txt', "r",encoding="utf-8") as f:
    train_data = f.readlines()
    # selected_data = train_data[:1000]
    # f.writelines(selected_data)
selected_data = train_data[:1000]
print(len(selected_data))

def dataselect(filepath,rate):
    with open(filepath, "r",encoding="utf-8") as f:
        dataset = f.readlines()
        length = int(len(dataset) * rate)
        selected_data = dataset[:length]
    filename = os.path.basename(filepath).split(".")[0] 
    filename = filename +"_"+ str(length)+".txt"
    dir = os.path.dirname(filepath)
    new_filepath = os.path.join(dir,filename)
    with open(new_filepath, "w",encoding="utf-8") as f:
        f.writelines(selected_data)
dev_path = 'D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/dev.txt'
train_path = 'D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/train.txt'
test_path = 'D:/picc/WorkDirectory-master/NPS/Chinese-Text-Classification-Pytorch-master/THUCNews/data/test.txt'
rate = 0.1
dataselect(dev_path,rate)   
dataselect(train_path,rate) 
dataselect(test_path,rate) 