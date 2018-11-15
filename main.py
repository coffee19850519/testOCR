from extract_text_from_image import Extract_text_from_image
import cv2,os

def find_all_images_in_fold(input_path):
    all_file_names = []
    # if input_path is a file
    if os.path.isfile(input_path):
        # obtain its fold
        input_path, _ = os.path.split(input_path)
    #return all files in this fold
    for file in os.listdir(input_path):
        all_file_names.append(os.path.join(input_path,file))
    return all_file_names


if __name__== "__main__":
        input_path = r'C:\Users\coffe\Desktop\pathway figures\cin_00006.png'
    #for file in find_all_images_in_fold(input_path):
        try:
            img = cv2.imread(input_path)
            test = Extract_text_from_image(img,input_path)
            test.pipeline()
            del test, img
        except:
            #if a certain file is not image file, then
            print('current path '+ input_path + ' is invalid')
            #break


