import cv2, os, time
from PIL import Image, ImageOps, ImageEnhance

class TemplateMatcher(object):
  class ImageNotFound(Exception): 
    pass

  def __init__(self, imagePath, templatePath, threshold=None):
    self.imagePath = imagePath
    if not os.path.exists(self.imagePath):    raise TemplateMatcher.ImageNotFound("Image at location '{}' does not exist.".format(self.imagePath))
    self.templatePath = templatePath
    if not os.path.exists(self.templatePath): raise TemplateMatcher.ImageNotFound("Image at location '{}' does not exist.".format(self.templatePath))
    self.threshold = threshold

  def getMatchCoordinate(self):
    if not hasattr(self, 'matchCoord'):
      # Get the sliced viewer window and small reference image according to laterality.
      image    = cv2.imread(self.imagePath)
      template = cv2.imread(self.templatePath)
      
      if not any([img is None for img in [image, template]]):
        h, w = template.shape[0:-1]
        method = cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(template, image, method)
        # We want the minimum squared difference
        mn,_,mnLoc,_ = cv2.minMaxLoc(result)
        # Extract the coordinates of the best match w or w/o given threshold
        # If no threshold is provided the best possible mathc will be returned
        if self.threshold:
          if mn < self.threshold:
            self.matchCoord = (mnLoc[0], mnLoc[1], mnLoc[0]+w, mnLoc[1]+h)
          else:
            self.matchCoord = None
        else:
          self.matchCoord = (mnLoc[0], mnLoc[1], mnLoc[0]+w, mnLoc[1]+h)
      else:
        self.matchCoord = None
    return self.matchCoord

def VisualizeResult(imagePath, rectagle):
  image = cv2.imread(imagePath)
  cv2.rectangle(image, (rectagle[0], rectagle[1]), (rectagle[2], rectagle[3]), (0, 0, 255), thickness=2)
  cv2.imshow("Template Match Position", image)
  cv2.waitKey(0)
  

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(prog='Template Matcher', 
                                   description=''' This script is implemented to match a template image on a given target image.
                                                   Usage: 
                                                   -i/--image PATH_TO_IMAGE 
                                                   -t/--template PATH_TO_TEMPLATE
                                               ''')
  required = parser.add_argument_group('required arguments')
  required.add_argument('-i', '--image',    help='Specify path to model file (.pb  file) for DNN.',                 required=True, dest='IMAGE',    nargs=1  )
  required.add_argument('-t', '--template', help='Specify path to image file (.png file) for Search Engine class.', required=True, dest='TEMPLATE', nargs=1  )
  args = parser.parse_args()
  IMAGE      = args.IMAGE[0]
  TEMPLATE   = args.TEMPLATE[0]

  tm = TemplateMatcher(IMAGE, TEMPLATE)
  rect = tm.getMatchCoordinate()
  VisualizeResult(IMAGE, rect)