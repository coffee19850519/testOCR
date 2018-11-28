import cv2,os
#import matplotlib.pyplot as plt
#import numpy as np

def image_label_generator(file_path):
    for file in os.listdir(file_path):
      _,file_name = os.path.split(file)
      file_name,file_ext = os.path.splitext(file_name)
      if file_ext in ['.jpg','.jepg','.tiff','.png']:
        #img_fp = open(file_fold+file_name+file_ext,'r')
        label_fp = open(os.path.join(file_path,'object_'+file_name),'r')
        label_fp.seek(0)
        yield cv2.imread(os.path.join(file_path, file_name+file_ext)),label_fp.readlines(), \
              os.path.join(file_path,'marked_'+file_name+file_ext)
        label_fp.close()

def uniform_labels(file_path):
  for file in os.listdir(file_path):
    _, file_name = os.path.split(file)
    file_name, file_ext = os.path.splitext(file_name)
    if file_ext is  '.txt':
      new_labels = []
      # img_fp = open(file_fold+file_name+file_ext,'r')
      label_fp = open(os.path.join(file_path, 'object_' + file_name), 'rw')
      label_fp.seek(0)
      for line in label_fp.readlines():
        shape, coord = line.split('\t')
        if shape != 'rect':
          # capture its coordinate points
          coords = list(map(int, coord.split(',')))
          new_labels.append('circle '+str(coords[0]-coords[2])+','+str(coords[
                                                               1]-coords[
            2])+','+str(coords[0]+coords[2])+','+str(coords[1]+coords[2]))
      label_fp.seek(0)
      label_fp.writelines(new_labels)
      label_fp.close()
      del new_labels


if __name__== "__main__":
    for img,labels,result_file_name in image_label_generator(
        r'C:\Users\coffe\Desktop\training_data'):
      for label_txt in labels:
        shape, coord = label_txt.split('\t')
        # capture its coordinate points
        coords = list(map(int, coord.split(',')))
        if shape != 'circle':
          #draw rectangle
          cv2.rectangle(img,(coords[0],coords[1]),(coords[2],coords[3]),(0,
                                                                         255,
                                                                         0),2)
        else:
          #draw circle
          cv2.circle(img,(coords[0],coords[1]),coords[2],(0,
                                                                         255,
                                                                         0),2)

        del shape,coord
      #save marked result
      cv2.imwrite(result_file_name,img)
      #release memory
      del img,label_txt,result_file_name