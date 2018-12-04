import cv2,os
#import matplotlib.pyplot as plt
#import numpy as np

def image_label_generator(file_path):
    for file in os.listdir(file_path+"/images/"):
      _,file_name = os.path.split(file)
      #if "marked" not in file_name:
      file_name,file_ext = os.path.splitext(file_name)
      if file_ext in ['.jpg','.jepg','.tiff','.png']:
        #img_fp = open(file_fold+file_name+file_ext,'r')
        label_fp = open(os.path.join(file_path+"/labels/",file_name),'r')
        label_fp.seek(0)
        yield cv2.imread(os.path.join(file_path+"/images/", file_name+file_ext)),label_fp.readlines(), \
              os.path.join(file_path+'/masked_by_label/',file_name+file_ext)
        label_fp.close()


if __name__== "__main__":
    for img,labels,result_file_name in image_label_generator(
        r'/home/labadmin/duolin/pathway annotation/pathway figures/training_data'):
      for label_txt in labels:
        shape, coord = label_txt.split('\t')
        # capture its coordinate points
        coords = list(map(int, coord.split(',')))
        if shape != 'circle':
          #draw rectangle
          cv2.rectangle(img,(coords[0]-1,coords[1]-1),(coords[2]+1,coords[3]+1),(255,
                                                                         255,
                                                                         255),thickness=-1)
        else:
          #draw circle
          cv2.circle(img,(coords[0],coords[1]),coords[2]+1,(255,255,
                                                                         255),thickness=-1)
        
        del shape,coord
      #save marked result
      cv2.imwrite(result_file_name,img)
      #release memory
      del img,label_txt,result_file_name