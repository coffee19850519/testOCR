import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import os,shutil
from configuration import Configer

class Extract_text_from_image:

    def __init__(self,input_image,input_path):
        self.__srcImageDict, file_name = os.path.split(input_path)
        self.__srcImageName, self.__srcImageExt = os.path.splitext(file_name)
        self.__srcImage = input_image

        self.__binaryImage = None
        self.__binaryImageINV = None
        self.__dilateImage = None
        #self.__markedImage = None
        self.__resImage = None

        self.__resRegions = []

        #generate a fold for results
        self.__srcImageDict = os.path.join(self.__srcImageDict, self.__srcImageName)
        #if it existed, remove the fold first
        #if os.path.isdir(self.__srcImageDict):
        shutil.rmtree(self.__srcImageDict,True)
        #then create it
        os.mkdir(self.__srcImageDict)

        #generate a fold for contour ROI
        self.__contourFold = os.path.join(self.__srcImageDict, 'contours')
        #if it existed, remove the fold first
        shutil.rmtree(self.__contourFold, True)
        #then create it
        os.mkdir(self.__contourFold)

        #srcImage
        #if not os.path.exists():

        #initialize saving paths
        self.__resTextPath = os.path.join(self.__srcImageDict,'textResult.txt')
        #self.__denoisingImagePath = os.path.join(self.__srcImageDict, 'denoising' + self.__srcImageExt)
        self.__binaryImagePath = os.path.join(self.__srcImageDict, 'binary' + self.__srcImageExt)
        self.__dilateImagePath = os.path.join(self.__srcImageDict, 'dilate' + self.__srcImageExt)
        self.__binaryImageINVPath = os.path.join(self.__srcImageDict, 'binaryINV' + self.__srcImageExt)
        #self.__markedImagePath = os.path.join(self.__srcImageDict, 'markedImage' + self.__srcImageExt)
        self.__resImagePath = os.path.join(self.__srcImageDict, 'resImage' + self.__srcImageExt)

    def preprocess(self):
        # fastNlMeansDenoisingColored(InputArray src, OutputArray dst, float h=3, float hColor=3, int templateWindowSize=7, int searchWindowSize=21 )

        gray = cv2.fastNlMeansDenoisingColored(self.__srcImage,
                                               Configer.DENOISE_STRENGTH,
                                               Configer.DENOISE_COLOR_STRENGTH,
                                               Configer.DENOISE_TEMPLATE_SIZE,
                                               Configer.SEARCH_SIZE)


        self.__binaryImage = cv2.adaptiveThreshold(cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, Configer.BLOCK_SIZE, Configer.C)

        self.__binaryImageINV = cv2.bitwise_not(self.__binaryImage)

        if Configer.DEBUG:
            #cv2.imwrite(self.__denoisingImagePath, gray)
            cv2.imwrite(self.__binaryImagePath, self.__binaryImage)
            cv2.imwrite(self.__binaryImageINVPath,self.__binaryImageINV)


        del gray

    def dilate_image(self):
        # dilate operation
        ele = cv2.getStructuringElement(cv2.MORPH_RECT, Configer.DILATE_KERNEL)
        self.__dilateImage = cv2.dilate(self.__binaryImageINV, ele, iterations=2)
        # erode operation
        # gray = cv2.erode(gray, Configer.DILATE_KERNEL)
        # dilate again
        # self.__dilateImage = cv2.dilate(gray, ele, iterations=3)
        if Configer.DEBUG:
            cv2.imwrite(self.__dilateImagePath, self.__dilateImage)

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
        image, contours, hierarchy = cv2.findContours(self.__dilateImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

            #if (width > self.__srcImage.shape[1] / 2 and rect[1][1] > self.__srcImage.shape[0] / 20):
        if Configer.DEBUG:
            markedImage = self.__srcImage.copy()
            cv2.drawContours(markedImage, regions,-1,(0,255,0),2)
            markedImagePath = os.path.join(self.__srcImageDict, 'markedImage' + self.__srcImageExt)
            cv2.imwrite(markedImagePath,markedImage)
            del  markedImage

        return regions



    def ocr_text_from_image(self, regions):
        text = []
        #regist pytesseract path
        pytesseract.pytesseract.tesseract_cmd = Configer.TERSSERACT_PATH
        #generate a copy to binaryImage for saving results
        self.__resImage = self.__srcImage.copy()

        idx = 0

        for region in regions:
            # crop regions from raw image considering offset
            h = abs(region[0][1] - region[2][1])
            w = abs(region[0][0] - region[2][0])
            Xs = [i[0] for i in region]
            Ys = [i[1] for i in region]
            x1 = min(Xs)
            y1 = min(Ys)

            regionImage = self.__binaryImage[y1:y1 + h+Configer.OFFSET,
                          x1:x1 + w+Configer.OFFSET]
            if Configer.DEBUG and regionImage.size > 0:
                cv2.imwrite(os.path.join(self.__contourFold, str(idx)+self.__srcImageExt), regionImage)

            idx = idx + 1

            # scaling ROI for better recognition
            textImage = cv2.resize(regionImage, (int(Configer.SCALE * w), int(Configer.SCALE * h)),
                                                 interpolation=cv2.INTER_CUBIC)
            # call tersseract to reconize text
            result = pytesseract.image_to_string(textImage, lang='eng', config=Configer.TERSSERACT_CONFIG)
            if result is not None and result != '':
                # draw rectangle on binary image
                cv2.drawContours(self.__resImage, [region], -1, (0, 255, 0), 2)
                # put the detected text to binary image
                cv2.putText(self.__resImage, result, (region[2][0], region[2][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 0))
                text.append(result + '\n')
                self.__resRegions.append(region)
            del regionImage,textImage

        # save all result to file
        result_fp = open(self.__resTextPath, 'w+', encoding='utf-8')
        result_fp.writelines(text)
        result_fp.close()

        if Configer.DEBUG:
            cv2.imwrite(self.__resImagePath,  self.__resImage)



    def pipeline(self):
        #step 1 preprocess to source image and generate binary image
            self.preprocess()
        #step 2 dilate the binary image for contour detection
            self.dilate_image()
        #step 3 find contour candidates from dilate image
            regions = self.findTextRegion()
        #step 4 ocr
            self.ocr_text_from_image(regions)


    def show_result(self):
        if Configer.DEBUG:
            f, axarr = plt.subplots(2, 3)
            axarr[0, 0].imshow('source image',self.__srcImage)
            axarr[0, 1].imshow('binary image', self.__binaryImage)
            axarr[0, 2].imshow('binary image INV', self.__binaryImageINV)
            axarr[1, 0].imshow('dilation',self.__dilateImage)
            axarr[1, 1].imshow('result',self.__markedImage)
            plt.show()
        else:
            print('Results only can be showed on DEBUG mode')
