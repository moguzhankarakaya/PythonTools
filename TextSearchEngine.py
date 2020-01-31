import pytesseract, cv2, os

import numpy                        as np
from   imutils.object_detection import non_max_suppression
from   PIL                      import ImageOps, ImageEnhance

import argparse
parser = argparse.ArgumentParser(prog='Text on Image Search Engine', 
                                  description=''' This script is implemented to query an image against given string or list of strings
                                                  Usage: 
                                                  -m/--model PATH_TO_DNN_MODEL_FILE
                                                  -t/--tesseract PATH_TO_TESSERACT_ENGINE
                                                  -i/--image PATH_TO_IMAGE  
                                                  Optional Parameters:
                                                  -q/--queryString String or list of string that has to be searched inside the image. 
                                                  If non given entire search result will be returned.''')
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-m', '--model',       help='Specify path to model file (.pb  file) for DNN.',                 required=True,  dest='MODEL_PATH',   nargs=1  )
required.add_argument('-m', '--model',       help='Specify path to tesseract engine (tesseract.exe  file).',         srequired=True, dest='TESSR_PATH',   nargs=1  )
required.add_argument('-i', '--image',       help='Specify path to image file (.png file) for Search Engine class.', required=True,  dest='IMAGE',        nargs=1  )
optional.add_argument('-q', '--queryString', help='String or list of strings to query against.',                     default=None,   dest='QUERY_ITEMS',  nargs='*')
args = parser.parse_args()
MODEL_PATH = args.MODEL_PATH[0]
IMAGE      = args.IMAGE[0]
pytesseract.pytesseract.tesseract_cmd = args.TESSR_PATH[0]

##
# \class TextSearchEngine
# TextSearchEngine is a class object that is used to query given 
# string or list of strings on an image.
# It utilizes both DNN functionalities of OpenCV library and 
# OCR engine to detect text location and extract the string out of it.
# DNN utilities work with a given pre-trained neural network and on 
# production mode pre-processed input image is fed into it in order to 
# get NN outputs. In the case of this implementation East Text Detection Network
# which is commenly used text detection network that is trained with image under wide range of 
# lighteining conditions and different languages, is used as a pre-trained NN.
# \param pathToImage A PNG image to query text on it.
# \param enhanceImage (optional) A boolean param to pre-process image to invert its colors and sharpen
# text before parsing text ROIs and apply OCR on those ROIs
#
class TextSearchEngine:
  MODEL_PATH = MODEL_PATH
  def __init__(self, pathToImage, enhanceImage = True):
    self.imagePath = pathToImage
    self.enhance   = enhanceImage
  
  ##
  # \brief Class member function that is used to query image against a given string.
  # \param searchString String that will be searched on the image.
  # \return Tuple in the form of (rectangle bounding box, found text inside that rectangle)
  #
  def queryStringOnImage(self, searchString):
    results = self.getSequenceOfTextFromImage()
    for (rect, text) in results:
      if searchString in text:
        return (rect, text)
    return (None, None)

  ##
  # \brief Class member function that is used to query image against a given list of strings.
  # \param searchStrings List of strings that will be searched on the image.
  # \return List of tuples in the form of (rectangle bounding box, found text inside that rectangle)
  #
  def queryStringsOnImage(self, searchStrings):
    scanResults = self.getSequenceOfTextFromImage()
    searchResults = []
    for (rect, text) in scanResults:
      if any([searchString in text for searchString in searchStrings]):
        searchResults.append((rect, text))
    return searchResults

  ##
  # \brief Class member function that is used to scan the entire image.
  # \return List of tuples in the form of (rectangle bounding box, found text inside that rectangle)
  #
  def getSequenceOfTextFromImage(self):
    orogonalImage, resizedImage, widthScale, heightScale = self._getOriginalAndResizedImage()
    ROICoords = self._getEstimatedTextROICoords(resizedImage)
    results = self._extractTextFromROIs(orogonalImage, ROICoords, widthScale, heightScale, paddingScale=0.20)
    return results

  def _getImage(self):
    image = cv2.imread(self.imagePath)
    if self.enhance:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image = (255 - image)
      image = cv2.equalizeHist(image)
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

  def _getOriginalAndResizedImage(self):
    image = self._getImage()
    original = image.copy()
    (initH, initW) = image.shape[:2]
    # Calculate closest dimensions which is multiple of 32
    # Use these dimensions to scale image
    # Text detection network accepts images with dimensions that are multiple of 32
    width  = initW // 32 * 32
    height = initH // 32 * 32
    widthScale  = initW / float(width)
    heightScale = initH / float(height)
    resized = cv2.resize(image, (width, height))
    return original, resized, widthScale, heightScale
  
  def _getEstimatedTextROICoords(self, image):
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    network    = cv2.dnn.readNet(TextSearchEngine.MODEL_PATH)
    # if tuple(np.mean(image, axis=(0,1))) routine to calculate image mean by channel
    # doens't give good results while text position detection
    # use these magic number as a tuple -> (123.68, 116.78, 103.94) as image mean
    blob       = cv2.dnn.blobFromImage(image, 1.0, (image.shape[1], image.shape[0]), tuple(np.mean(image, axis=(0,1))), swapRB=True, crop=False)
    
    network.setInput(blob)

    (scores, geometry)   = network.forward(layerNames)
    (rects, confidences) = self._parseNetworkOutput(scores, geometry, 0.5)
    ROICoords = non_max_suppression(np.array(rects), probs=confidences)

    return ROICoords

  def _parseNetworkOutput(self, scores, geometry, minConfidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
      scoresData = scores[0, 0, y]
      xData0 = geometry[0, 0, y]
      xData1 = geometry[0, 1, y]
      xData2 = geometry[0, 2, y]
      xData3 = geometry[0, 3, y]
      anglesData = geometry[0, 4, y]

      for x in range(0, numCols):
        if scoresData[x] < minConfidence:
          continue

        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

    return (rects, confidences)

  def _extractTextFromROIs(self, image, ROICoords, widthScale, heightScale, paddingScale=0.50):
    results = []

    BBoxText = []
    for (startX, startY, endX, endY) in ROICoords:
      startX = int(startX * widthScale)
      startY = int(startY * heightScale)
      endX   = int(endX * widthScale)
      endY   = int(endY * heightScale)

      dX = int((endX - startX) * paddingScale)
      dY = int((endY - startY) * paddingScale)

      startX = max(0, startX - 2*dX)
      startY = max(0, startY - 0*dY)
      endX = min(image.shape[1], endX + 2*dX)
      endY = min(image.shape[0], endY + 2*dY)

      BBoxText.append((startX, startY, endX, endY))  
    BBoxText = sorted(BBoxText, key=lambda bb:bb[1])

    BBoxLSorted = []
    BBoxLTemp   = [BBoxText[0]]
    cursorY     = BBoxText[0][1]
    for bbox in BBoxText[1:]:
      (_, startY, _, _) = bbox
      if cursorY + 10 < startY: 
        #line finalize
        sorted(BBoxLTemp, key=lambda bb:bb[0])
        BBoxLSorted += sorted(BBoxLTemp, key=lambda bb:bb[0])
        BBoxLTemp.clear()
        BBoxLTemp.append(bbox)
        cursorY = startY
        continue
      BBoxLTemp.append(bbox)
      cursorY = startY
    BBoxLSorted += sorted(BBoxLTemp, key=lambda bb:bb[0])
    BBoxLTemp.clear()

    BBoxLine = []
    lineStartX = BBoxLSorted[0][0]; lineStartY = BBoxLSorted[0][1]
    lineEndX   = BBoxLSorted[0][2]; lineEndY   = BBoxLSorted[0][3]
    cursorY    = lineEndY;          cursorX    = lineEndX
    for bbox in BBoxLSorted[1:]:
      (startX, startY, endX, endY) = bbox
      if cursorY - int((lineEndY-lineStartY)*0.25) < startY or cursorX + 10 < startX  or startX < cursorX - int((lineEndX - lineStartX)*1.25):
        #line finalize
        BBoxLine.append((lineStartX, lineStartY, lineEndX, lineEndY))
        #continue new line
        lineStartX = startX;   lineStartY = startY
        lineEndX   = endX;     lineEndY   = endY
        cursorY    = lineEndY; cursorX    = lineEndX
        continue
      #expend the line box if necessary
      lineStartX = min(lineStartX, startX)
      lineStartY = min(lineStartY, startY)
      lineEndX   = max(lineEndX, endX)
      lineEndY   = max(lineEndY, endY)
      cursorY    = lineEndY
      cursorX    = lineEndX
    BBoxLine.append((lineStartX, lineStartY, lineEndX, lineEndY))

    for (startX, startY, endX, endY) in BBoxLine:
      roi = image[startY:endY, startX:endX]

      config = ("-l eng --oem 3 --psm 10")
      text = pytesseract.image_to_string(roi, config=config)

      results.append(((startX, startY, endX, endY), text))

    results = sorted(results, key=lambda r:r[0][1])

    return results

def visualizeResult(results, pathToImage):
  image = cv2.imread(pathToImage)

  for ((startX, startY, endX, endY), text) in results:
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))

    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = image.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # show the output image
    cv2.imshow("Text Detection", output)
    cv2.waitKey(0)

  
if __name__=='__main__':
  searchStrings = args.QUERY_ITEMS

  searchEngine = TextSearchEngine(IMAGE)

  if searchStrings:
    BBoxOfTextLocations = searchEngine.queryStringsOnImage(searchStrings)
  else:
    BBoxOfTextLocations = searchEngine.getSequenceOfTextFromImage()

  visualizeResult(BBoxOfTextLocations, IMAGE)