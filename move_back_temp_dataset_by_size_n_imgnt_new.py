import pickle
import os, sys
import re
from shutil import copyfile
from shutil import move
from shutil import copytree
from operator import itemgetter
import random, datetime
source_dir=sys.argv[1] #Has to be parent dir 
n=int(sys.argv[2])
print("Dataset size: ",n)
counter=1
# try:
    # with open("class_images_dict_imgnt_top"+str(n)+'.pkl','rb') as f: #class_images_dict_imgnt_top100.pkl
        # class_images_dict=pickle.load(f)
# except:
    # print("class_images_dict_imgnt_top"+str(n)+'.pkl'+" not found")
    # exit()
with open('class_images_dict_imgnt.pkl','rb') as f:
    imgnt=pickle.load(f)
res=(dict(sorted(imgnt.items(), key = itemgetter(1), reverse = True)[:n]))
# answer=all(item in imgnt_200.items() for item in class_images_dict.items())
# print("Dataset correct =",answer)
# exit()
# print(sorted( ((v,k) for k,v in res.items()), reverse=True))
classes=list(res.keys())
print(len(classes))

# split_source_dir=(source_dir.split("/")[:len(source_dir.split("/"))-1])
# parent_dir="/".join(split_source_dir)
destination_dir="/ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_"+str(n)
# destination_dir="C:/tmp/imgnet/imgnet_top_"+str(n)
print(destination_dir)
start_time=datetime.datetime.now()
print(start_time)

# exit()
#First create train set
done_count=1
not_there=[x for x in classes if not x in os.listdir(destination_dir+"/train")]

alldirs=["/train/", "/val/","/test/"]
if len(not_there)==0:
    for split in alldirs:
        for folder in classes:
            # full_sub_dir=os.path.join(source_dir,folder)
            full_sub_dir=source_dir+split+folder
            full_dest_dir=destination_dir+split+folder
            # print("copying from: "+full_sub_dir+" copying to: "+full_dest_dir)
            # exit()
            move(full_dest_dir, full_sub_dir)
            if done_count%100==0:
                print(done_count)
            done_count+=1
else:
    print("These folders are not present in source_dir: ",not_there)
    exit()

# split_source_dir=(source_dir.split("/")[:len(source_dir.split("/"))-1])
# source_dir="/".join(split_source_dir)

# a=os.scandir(destination_dir+"/train")
# base_dir=(destination_dir+"/val")
# val_images_counter=1
# print("doing val splits")
# for folder in a:
    # # print(folder.name)
    # images=[x for x in os.scandir(folder.path)]
    # number_of_val_images=int(len(images)*0.15)
    # val_images_counter+=number_of_val_images
    # random.seed(30)
    # random_image_set=[x for x in random.sample(images,number_of_val_images)]
    # # print(len(images))
    # for file in random_image_set:
        # full_dir=os.path.join(base_dir,folder.name)
        # if not os.path.exists(full_dir):
            # os.makedirs(full_dir)
        # move(file.path, os.path.join(full_dir,file.name))

# a=os.scandir(destination_dir+"/train")
# base_dir=(destination_dir+"/test")
# print("doing test splits")
# for folder in a:
    # # print(folder.name)
    # images=[x for x in os.scandir(folder.path)]
    # number_of_val_images=int(len(images)*0.176) #17.6% to get a 15% split from the remaining 85% of the train set
    # val_images_counter+=number_of_val_images
    # random.seed(30)
    # random_image_set=[x for x in random.sample(images,number_of_val_images)]
    # # print(len(images))
    # for file in random_image_set:
        # full_dir=os.path.join(base_dir,folder.name)
        # if not os.path.exists(full_dir):
            # os.makedirs(full_dir)
        # move(file.path, os.path.join(full_dir,file.name))

end_time=datetime.datetime.now()
print(end_time)
print("TIme taken to move = ",(end_time-start_time))
