import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import os,shutil
from configuration import Configer

class Extract_text_from_image:
    
    def __init__(self,input_image,input_path):
        self.srcImageDict, file_name = os.path.split(input_path)
        self.srcImageName, self.srcImageExt = os.path.splitext(file_name)
        self.srcImage = input_image
        
        self.binaryImage = None
        self.binaryImageINV = None
        self.dilateImage = None
        #self.markedImage = None
        self.resImage = None
        
        self.resRegions = []
        
        #generate a fold for results
        self.srcImageDict = os.path.join(self.srcImageDict, self.srcImageName)
        #if it existed, remove the fold first
        #if os.path.isdir(self.srcImageDict):
        shutil.rmtree(self.srcImageDict,True)
        #then create it
        os.mkdir(self.srcImageDict)
        
        #generate a fold for contour ROI
        self.contourFold = os.path.join(self.srcImageDict, 'contours')
        #if it existed, remove the fold first
        shutil.rmtree(self.contourFold, True)
        #then create it
        os.mkdir(self.contourFold)
        
        #srcImage
        #if not os.path.exists():
        
        #initialize saving paths
        self.resTextPath = os.path.join(self.srcImageDict,'textResult.txt')
        #self.denoisingImagePath = os.path.join(self.srcImageDict, 'denoising' + self.srcImageExt)
        self.binaryImagePath = os.path.join(self.srcImageDict, 'binary' + self.srcImageExt)
        self.dilateImagePath = os.path.join(self.srcImageDict, 'dilate' + self.srcImageExt)
        self.binaryImageINVPath = os.path.join(self.srcImageDict, 'binaryINV' + self.srcImageExt)
        #self.markedImagePath = os.path.join(self.srcImageDict, 'markedImage' + self.srcImageExt)
        self.resImagePath = os.path.join(self.srcImageDict, 'resImage' + self.srcImageExt)
    
    def preprocess(self):
        # fastNlMeansDenoisingColored(InputArray src, OutputArray dst, float h=3, float hColor=3, int templateWindowSize=7, int searchWindowSize=21 )
        
        gray = cv2.fastNlMeansDenoisingColored(self.srcImage,
                                               Configer.DENOISE_STRENGTH,
                                               Configer.DENOISE_COLOR_STRENGTH,
                                               Configer.DENOISE_TEMPLATE_SIZE,
                                               Configer.SEARCH_SIZE)
        
        
        self.binaryImage = cv2.adaptiveThreshold(cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, Configer.BLOCK_SIZE, Configer.C)
        
        self.binaryImageINV = cv2.bitwise_not(self.binaryImage)
        
        if Configer.DEBUG:
            #cv2.imwrite(self.denoisingImagePath, gray)
            cv2.imwrite(self.binaryImagePath, self.binaryImage)
            cv2.imwrite(self.binaryImageINVPath,self.binaryImageINV)
    
    
        del gray
    
    def dilate_image(self):
        # dilate operation
        ele = cv2.getStructuringElement(cv2.MORPH_RECT, Configer.DILATE_KERNEL)
        self.dilateImage = cv2.dilate(self.binaryImageINV, ele, iterations=2)
        # erode operation
        # gray = cv2.erode(gray, Configer.DILATE_KERNEL)
        # dilate again
        # self.dilateImage = cv2.dilate(gray, ele, iterations=3)
        if Configer.DEBUG:
            cv2.imwrite(self.dilateImagePath, self.dilateImage)
    
    # potential rule
    def getDarkColorPercent(self, img, img_height, img_width):
        
        imgSize = img_width * img_height
        result = cv2.threshold(img, 100, -1, cv2.THRESH_TOZERO)[1]
        nonzero = cv2.countNonZero(result)
        del  result
        if nonzero > 0:
            return (imgSize - nonzero) / float(imgSize)
        else:
            return 0
    
    
    def findTextRegion(self):
        regions = []
        image, contours, hierarchy = cv2.findContours(self.dilateImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #image, contours, hierarchy = cv2.findContours(self.binaryImageINV, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 2. remove meaningless candidates
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            # remove candidates with too small & large area
            if (area > Configer.MAX_AREA) or (area < Configer.MIN_AREA):
                continue
            
            
            # fit bounding rectangle
            rect = cv2.minAreaRect(cnt)
            #gain its vertex points
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # remove candidates with volumn shape
            wh_ratio = abs(box[0][0] - box[2][0]) / abs(box[0][1] - box[2][1])
            if (wh_ratio >  Configer.MAX_WH_RATIO) or \
                    (wh_ratio <  Configer.MIN_WH_RATIO ) or \
                    abs(box[0][0] - box[2][0]) < 10 or abs(box[0][1] - box[2][1]) < 5:
                continue
            else:
                regions.append(box)
            
            #if (width > self.srcImage.shape[1] / 2 and rect[1][1] > self.srcImage.shape[0] / 20):
        if Configer.DEBUG:
            markedImage = self.srcImage.copy()
            cv2.drawContours(markedImage, regions,-1,(0,255,0),2)
            markedImagePath = os.path.join(self.srcImageDict, 'markedImage' + self.srcImageExt)
            cv2.imwrite(markedImagePath,markedImage)
            del  markedImage
        
        return regions
    
    def findarrowsRegion(self):
        regions = []
        image, contours, hierarchy = cv2.findContours(self.dilateImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #image, contours, hierarchy = cv2.findContours(self.binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 2. remove meaningless candidates
        markedImage = self.srcImage.copy()
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            # remove candidates with too small & large area
            if (area > Configer.MAX_AREA) or (area < Configer.MIN_AREA):
                continue
            
            
            # fit bounding rectangle
            rect = cv2.minAreaRect(cnt)
            
            #gain its vertex points
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # get color ratio
            h = abs(box[0][1] - box[2][1])
            w = abs(box[0][0] - box[2][0])
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            y1 = min(Ys)
            
            regionImage = image[y1:y1 + h, x1:x1 + w]
            colar_ratio=self.getDarkColorPercent(regionImage,h,w)
            if colar_ratio < 0.6:
                continue
            cv2.putText(markedImage, "%.2f" % colar_ratio, (box[2][0], box[2][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 0))
            #print(str(colar_ratio)+"\n")
            # remove candidates with volumn shape
            wh_ratio = abs(box[0][0] - box[2][0]) / abs(box[0][1] - box[2][1])
            if (wh_ratio >  Configer.MAX_WH_RATIO) or \
                    (wh_ratio <  Configer.MIN_WH_RATIO ) or \
                    abs(box[0][0] - box[2][0]) < 10 or abs(box[0][1] - box[2][1]) < 5:
                continue
            else:
                regions.append(box)
            
            #if (width > self.srcImage.shape[1] / 2 and rect[1][1] > self.srcImage.shape[0] / 20):
        if Configer.DEBUG:
            
            cv2.drawContours(markedImage, regions,-1,(255,0,0),2)
            markedImagePath = os.path.join(self.srcImageDict, 'markedImage_arrows' + self.srcImageExt)
            cv2.imwrite(markedImagePath,markedImage)
            #cv2.putText(markedImage, result, (region[2][0], region[2][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 0))
            del  markedImage
        
        return regions
    
    
    def ocr_text_from_image(self, regions):
        text = []
        #regist pytesseract path
        pytesseract.pytesseract.tesseract_cmd = Configer.TERSSERACT_PATH
        #generate a copy to binaryImage for saving results
        self.resImage = self.srcImage.copy()
        
        idx = 0
        
        for region in regions:
            # crop regions from raw image considering offset
            h = abs(region[0][1] - region[2][1])
            w = abs(region[0][0] - region[2][0])
            Xs = [i[0] for i in region]
            Ys = [i[1] for i in region]
            x1 = min(Xs)
            y1 = min(Ys)
            
            regionImage = self.binaryImage[y1:y1 + h+Configer.OFFSET,
                          x1:x1 + w+Configer.OFFSET]
            if Configer.DEBUG and regionImage.size > 0:
                cv2.imwrite(os.path.join(self.contourFold, str(idx)+self.srcImageExt), regionImage)
            
            idx = idx + 1
            
            # scaling ROI for better recognition
            textImage = cv2.resize(regionImage, (int(Configer.SCALE * w), int(Configer.SCALE * h)),
                                                 interpolation=cv2.INTER_CUBIC)
            # call tersseract to reconize text
            result = pytesseract.image_to_string(textImage, lang='eng', config=Configer.TERSSERACT_CONFIG)
            if result is not None and result != '':
                # draw rectangle on binary image
                cv2.drawContours(self.resImage, [region], -1, (0, 255, 0), 2)
                # put the detected text to binary image
                cv2.putText(self.resImage, result, (region[2][0], region[2][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 0))
                text.append(result + '\n')
                self.resRegions.append(region)
            del regionImage,textImage
        
        # save all result to file
        result_fp = open(self.resTextPath, 'w+', encoding='utf-8')
        result_fp.writelines(text)
        result_fp.close()
        
        if Configer.DEBUG:
            cv2.imwrite(self.resImagePath,  self.resImage)
    
    
    
    def pipeline(self):
        #step 1 preprocess to source image and generate binary image
            self.preprocess()
        #step 2 dilate the binary image for contour detection
            self.dilate_image()
            self.findarrowsRegion()
        #step 3 find contour candidates from dilate image
            regions = self.findTextRegion()
        #step 4 ocr
            self.ocr_text_from_image(regions)
    
    
    def show_result(self):
        if Configer.DEBUG:
            f, axarr = plt.subplots(2, 3)
            axarr[0, 0].imshow('source image',self.srcImage)
            axarr[0, 1].imshow('binary image', self.binaryImage)
            axarr[0, 2].imshow('binary image INV', self.binaryImageINV)
            axarr[1, 0].imshow('dilation',self.dilateImage)
            axarr[1, 1].imshow('result',self.markedImage)
            plt.show()
        else:
            print('Results only can be showed on DEBUG mode')
