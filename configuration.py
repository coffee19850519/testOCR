class Configer:

    #debug marker
    DEBUG = True

    #denoise parameters
    DENOISE_STRENGTH = 3
    DENOISE_COLOR_STRENGTH = 3
    DENOISE_TEMPLATE_SIZE = 9
    SEARCH_SIZE = 21

    #binary parameters
    BLOCK_SIZE = 5
    C = 15

    #dilate parameters
    DILATE_KERNEL = (2, 2)

    #contour parameters
    MAX_AREA = 1000
    MIN_AREA = 10
    MAX_WH_RATIO = 15
    MIN_WH_RATIO = 0.5
    OFFSET = 5
    SCALE = 2

    #tesseract paramters
    TERSSERACT_PATH = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    TERSSERACT_CONFIG = '-l eng --oem 1 --psm 3'
