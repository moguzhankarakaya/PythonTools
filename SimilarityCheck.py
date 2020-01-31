import os
import sys
import cv2
import copy
import json
import argparse
import numpy as np

from skimage.metrics import structural_similarity as similarityIndex

class log:
  @staticmethod
  def info(msg):
    print(msg)

parser = argparse.ArgumentParser(prog='Similarity Check', 
                                  description=''' This script is implemented to calculate structural similarity 
                                                  index between two images (expected/observed). 
                                                  It can be used with either
                                                  [-d/--directory PATH] or
                                                  [-e/--expected PATH_TO_EXPECTED_IMAGE -o/--observed PATH_TO_OBSERVED_IMAGE 
                                                  -O/--outdir PATH_TO_SAVE_RESULTS]. Optionally mask params in json format 
                                                  can be parsed if path to json file is specified.''')
scenario_1 = parser.add_argument_group('Required Arguments For Directory Run Mode')
scenario_1.add_argument('-d', '--directory', help='Specify the folder with "*_exp.png" and "*_obs.png" images', default=None, dest='folderPath', nargs=1)
scenario_2 = parser.add_argument_group('Required Arguments For Single Exp/Obs Image Run Mode')
scenario_2.add_argument('-e', '--expected',  help='Specify the expected image path. File name should contain "*_epx.png" suffix', default=None, dest='expectedImage', nargs=1)
scenario_2.add_argument('-o', '--observed',  help='Specify the observed image path. File name should contain "*_obs.png" suffix', default=None, dest='observedImage', nargs=1)
scenario_2.add_argument('-O', '--outdir',    help='Specify the folder for output images to be saved',  default=None, dest='outFolderPath', nargs=1)
parser.add_argument('--masks', help='Specify path to JSON file for masks to be used in FThreshold check.', default=None, dest='MASKS',   nargs=1)
args = parser.parse_args()

class SimilarityCheckOptions:
  def __init__(self, kernelSize, globalFailRatio, localFailureDensity, similarityIndexetric):
    self.ks   = kernelSize
    self.failureRatio  = globalFailRatio
    self.localFailureRatio  = localFailureDensity
    self.similarityIndex = similarityIndexetric

  def returnToFactorySettings(self):
    defaultOptions = self.GetDefaultOptions()
    self.ks   = defaultOptions['Kernel Size']
    self.failureRatio  = defaultOptions['Global Failure Ratio']
    self.localFailureRatio  = defaultOptions['Local Failure Density']
    self.similarityIndex = defaultOptions['similarityIndex']

  @classmethod
  def GetDefaultOptions(cls):
    return {'Kernel Size' : (10, 10), 'Global Failure Ratio' : 0.001, 'Local Failure Density' : 0.01, 'similarityIndex' : 0.95}

  def IsItDefault(self):
    defaultOptions = self.GetDefaultOptions()
    if defaultOptions['Kernel Size'] != self.ks:
      return False
    if defaultOptions['Global Failure Ratio'] != self.failureRatio:
      return False
    if defaultOptions['Local Failure Density'] != self.localFailureRatio:
      return False
    if defaultOptions['similarityIndex'] != self.similarityIndex:
      return False
    return True

##
# \class SimilarityCheck
# Class making screenshot verification using Structural Similarity Index Metric (similarityIndex)
# Implementation contains self debuging methods used in standalone script run.
# In case of detailed analysis required, please provide expected and observed images.
#
class SimilarityCheck:
  ##
  # Class variable to look-up debug image output names
  #
  ImageListDict = [{"Expected Image"   : 'expectedImage' },
                   {"Observed Image"   : 'observedImage' },
                   {"Differance Image" : 'diffImage'     },]

  ##
  # \class SimilarityCheck.ImageNotExists
  # Exception for wrong image path
  #
  class ImageNotExists(Exception): pass
  
  ##
  # \class SimilarityCheck.ImageList
  # Look-up enumaration for debug images
  #
  class ImageList:
    ExpectedImage   = 0
    ObservedImage   = 1
    DifferanceImage = 2
  
  ##
  # \class SimilarityCheck.VerdictVerboseList
  # Look-up enumaration for final verdict log
  #
  class VerdictVerboseList:
    GlobalPixelThresholdExceeded  = 0
    LocalPixelThresholdSatisfied  = 1
    similarityIndexScoreThresholdNotAchieved = 2
  
  ##
  # \class SimilarityCheck.VerdictVerbose
  # Container for final verification summary 
  # used string representation of the class
  #
  class VerdictVerbose:
    def __init__(self, verbose, options):
      self.verbose = verbose
      self.options = options
      self.summary = {}
    def attachSummary(self, summary):
      self.summary = copy.deepcopy(summary)

  ##
  # \brief Constructor of SimilarityCheck class.
  # \param expectedImagePath path to the expected image
  # \param observedImagePath path to the observed image
  # \param vpParams (optional) VPParameters class instance to include mask 
  #                            information for image comperison. Default 
  #                            no mask will be applied for comperison.
  #
  def __init__(self, expectedImagePath, observedImagePath, **kwargs):
    if not all(os.path.exists(path) for path in [expectedImagePath, observedImagePath]):
      raise SimilarityCheck.ImageNotExists("Epected and observed images should be available. Check paths: {}".format(", ".join([expectedImagePath, observedImagePath])))
    
    self.vpParams                    = kwargs['vpParams']      if 'vpParams'   in kwargs                          else None
    SimilarityCheck.KERNEL_SIZE = kwargs['FThresholdOptions'].ks if 'FThresholdOptions' in kwargs and kwargs['FThresholdOptions'] else SimilarityCheckOptions.GetDefaultOptions()['Kernel Size']
    
    self.expectedImage = self._clearMaskedArea(cv2.imread(expectedImagePath))
    self.observedImage = self._clearMaskedArea(cv2.imread(observedImagePath))
    
    assert self.expectedImage.shape == self.observedImage.shape, "Expected and observed images are different size. Exp: {}, Obs: {}".format(self.expectedImage.shape, self.observedImage.shape)
    
    self.height, self.width = self.expectedImage.shape[0:2]
    self.channel = 1
    if len(self.expectedImage.shape) > 2:
      self.channel = self.expectedImage.shape[-1]

    self.diffImage         = np.repeat(np.max(np.where(self.expectedImage - self.observedImage == 0, 0, 255), axis=2).astype(dtype=np.uint8)[:,:,np.newaxis], 3, axis=2)
    self.diffMatrix        = np.max(np.where(self.expectedImage - self.observedImage == 0, 0, 1), axis=2).astype(dtype=np.uint8)
    self.pxDensityDictList = self._calculatePixelDensities()

  ##
  # \brief Function to calculate total number of failed pixels.
  # \return Total number of failed pixels in the entire image (Not in a cropped kernel)
  #
  def getNumberOfDiffPixels(self):
    return np.sum(self.diffMatrix)
  
  ##
  # \brief Function to calculate ratio of total number of failed 
  # pixels to total number of pixels
  # \return Global Fail Ratio 
  # (failureRatio -> Ratio of total number of failed pixels to total number of pixels.)
  #
  def getRatioOfDiffPixels(self):
    return self.getNumberOfDiffPixels()/(self.width * self.height)

  ##
  # \brief Function to return failure summary for 
  # pixel with max local fail density (localFailureRatio)
  # \return Summary dictionary for failed pixels with max localFailureRatio
  #
  def getMaxLocalDensityLocationSummary(self):
    maxLocalDensity    = 0
    returnPxDensityDict = {}
    for pxDensityDict in self.pxDensityDictList:
      if pxDensityDict['FailureDensity'] > maxLocalDensity:
        maxLocalDensity    = pxDensityDict['FailureDensity']
        returnPxDensityDict = pxDensityDict
    return returnPxDensityDict

  ##
  # \brief Function giving final verdict for Discepancy Failure Tolerance
  # \return boolean variable for final verdict. 'True' : PASS, 'False' : FAILURE
  #
  def getFinalVerdict(self, FThresholdOptions=None):
    if hasattr(self, 'finalVerdict'):
      return self.__dict__['finalVerdict']
    if not FThresholdOptions:
      FThresholdOptions = SimilarityCheckOptions(None, None, None, None)
      FThresholdOptions.returnToFactorySettings()
    if self.getRatioOfDiffPixels() < FThresholdOptions.failureRatio:
      issimilarityIndexCalcIsDone = False
      for pxDensityDict in self.pxDensityDictList:
        localDensity, densityKernelWindow = pxDensityDict['FailureDensity'], pxDensityDict['KernelWindow']
        if localDensity > FThresholdOptions.localFailureRatio:
          issimilarityIndexCalcIsDone = True
          if self._calculatesimilarityIndexIndex(densityKernelWindow) < FThresholdOptions.similarityIndex:
            logMsg = '''\n
                Total number of failed pixel is '{}'. Density of failing pixels over the all region is '{}'.
                Similarity check has failed on pixel '({},{})'. Kernel window is from '({},{})' to '({},{})'.
                Calculated similarityIndex score on that region is '{}' Score threshold is {}. 
                Allowed kernel fail density is {}.\n
            '''.format(self.getNumberOfDiffPixels(), self.getRatioOfDiffPixels(), 
                      pxDensityDict['AnchorPixel'][0], pxDensityDict['AnchorPixel'][1],
                      densityKernelWindow[1],densityKernelWindow[0],densityKernelWindow[3],densityKernelWindow[2],
                      self._calculatesimilarityIndexIndex(densityKernelWindow), FThresholdOptions.similarityIndex, FThresholdOptions.localFailureRatio )
            log.info(logMsg)
            self.verdictVerbose = SimilarityCheck.VerdictVerbose(SimilarityCheck.VerdictVerboseList.similarityIndexScoreThresholdNotAchieved, FThresholdOptions)
            self.verdictVerbose.attachSummary(pxDensityDict)
            self.finalVerdict   = False
            return False
      logMsg = '''\n
          Total number of failed pixel is '{}'. Density of failing pixels over the all region is '{}'.
          Validation is done using '{}x{}' kernel and '{}' as the similarityIndex Score threshold.\n
      '''.format(self.getNumberOfDiffPixels(), self.getRatioOfDiffPixels(), 
                 SimilarityCheck.KERNEL_SIZE[0], SimilarityCheck.KERNEL_SIZE[1], FThresholdOptions.similarityIndex )
      log.info(logMsg)
      self.verdictVerbose = SimilarityCheck.VerdictVerbose(SimilarityCheck.VerdictVerboseList.LocalPixelThresholdSatisfied, FThresholdOptions)
      summary = self.getMaxLocalDensityLocationSummary(); summary.update({'similarityIndexRequired':issimilarityIndexCalcIsDone})
      self.verdictVerbose.attachSummary(summary)
      self.finalVerdict   = True
      return True
    logMsg = '''\n
        Total number of failed pixel is '{}'. Density of failing pixels over the all region is '{}'.
        That is above threshold defined for total number of failing pixel density '{}'.\n
    '''.format(self.getNumberOfDiffPixels(), self.getRatioOfDiffPixels(), FThresholdOptions.failureRatio )
    log.info(logMsg)
    self.verdictVerbose = SimilarityCheck.VerdictVerbose(SimilarityCheck.VerdictVerboseList.GlobalPixelThresholdExceeded, FThresholdOptions)
    self.finalVerdict = False
    return False

  ##
  # \brief Function to display a debugging image with OpenCV window
  # \param ImageListEnum enumaration from ImageList class to choose which image to display
  #
  def showImage(self, ImageListEnum):
    imageToShow = SimilarityCheck.ImageListDict[ImageListEnum]
    imageWindowName    = list(imageToShow.keys())[0]
    imageAttributeName = list(imageToShow.values())[0]
    cv2.imshow(imageWindowName, self.__dict__[imageAttributeName])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  ##
  # \brief Fnction to save a debugging image to a given path
  # \param ImageListEnum enumaration from ImageList class to choose which image to display
  # \param pathToSave path to save debugging image of choice
  #
  def saveImage(self, ImageListEnum, pathToSave):
    imageToSave        = SimilarityCheck.ImageListDict[ImageListEnum]
    imageAttributeName = list(imageToSave.values())[0]
    cv2.imwrite(pathToSave, self.__dict__[imageAttributeName])

  ##
  # \brief Function to save differance (between expected and observed image) matrix as image file
  # \param pathToSave path to save debugging image of choice
  #
  def saveDiffMatrix(self, pathToSave):
    from matplotlib import pyplot as plt
    plt.imsave(pathToSave, self.diffMatrix)

  ##
  # \brief Function to calculate similarityIndex index on a given kernel window coordinates
  # \param kernelWindow  required parameter to crop the expected and observed images.
  #                      It is four integer to specify coordinates from UP - LEFT to DOWN - RIGHT
  # \param useWangParams (optional) Optinal boolean param to use original implementation form Wang's paper.
  #                      By default this parameter is true. This implementation would be used only if 
  #                      the kernel size is grater than 10x10
  # \param debug         (optional) When it is set to True, it returns also cropped images in the debugging mode.
  #                      By default this parameter is false.
  #
  def _calculatesimilarityIndexIndex(self, kernelWindow, useWangParams=True, debug=False):
    if kernelWindow:
      WangImplementation = {}
      if useWangParams and SimilarityCheck.KERNEL_SIZE[0] > 10 and SimilarityCheck.KERNEL_SIZE[1] > 10:
          WangImplementation = {'gaussian_weights' : True, 'sigma' : 1.5, 'use_sample_covariance' : False}
      expKernel    = self.expectedImage[kernelWindow[0]:kernelWindow[2], kernelWindow[1]:kernelWindow[3], ]
      obsKernel    = self.observedImage[kernelWindow[0]:kernelWindow[2], kernelWindow[1]:kernelWindow[3], ]
      similarityIndexResult   = similarityIndex(expKernel, obsKernel, multichannel=True, **WangImplementation)
      if debug:
        return (similarityIndexResult, expKernel, obsKernel)
      return similarityIndexResult
    return 1.0

  ##
  # \brief Function to return list of summary dictionaries of failed pixels.
  # It calculates kernel Window aroung the failure pixel, the amount of failed pixels  
  # inside the kernel window and localFailureRatio (Local Failure Density) of that particular failed
  # pixel.  
  #
  def _calculatePixelDensities(self):
    diffMatrixSummaryList = []
    diffPixels = np.argwhere(self.diffMatrix == 1) 
    for diff in diffPixels:
      summary = {}
      kernelWindow    = self._getKernelAroundAnchor(diff, SimilarityCheck.KERNEL_SIZE)
      diffMtxWmdwed   = self.diffMatrix[kernelWindow[0]:kernelWindow[2], kernelWindow[1]:kernelWindow[3]]
      numOfFailedPixs = np.sum(diffMtxWmdwed)
      failureDensity  = numOfFailedPixs / (diffMtxWmdwed.shape[0] * diffMtxWmdwed.shape[1])
      summary['AnchorPixel']    = (diff[0], diff[1])
      summary['KernelWindow']   = kernelWindow
      summary['FailurePixel']   = numOfFailedPixs
      summary['FailureDensity'] = failureDensity
      diffMatrixSummaryList.append(summary)
    return sorted(diffMatrixSummaryList, key = lambda i: i['FailureDensity'])
  
  ##
  # \brief Function to calculate kernel window around the anchor pixel
  # \param anchor the middle pixel to draw the kernel around it.
  # \param kernelSize the size of the kernel window as a tupple
  #
  def _getKernelAroundAnchor(self, anchor, kernelSize):
    topOffset = kernelSize[0] // 2; btmOffset = kernelSize[0] - topOffset  
    lftOffset = kernelSize[1] // 2; rhtOffset = kernelSize[1] - lftOffset
    topIdx = anchor[0] - topOffset; btmIdx = anchor[0] + btmOffset
    lftIdx = anchor[1] - lftOffset; rhtIdx = anchor[1] + rhtOffset
    topIdx, btmIdx = SimilarityCheck._shiftBoundries(topIdx, btmIdx, 0, self.height)
    lftIdx, rhtIdx = SimilarityCheck._shiftBoundries(lftIdx, rhtIdx, 0, self.width ) 
    return (topIdx, lftIdx, btmIdx, rhtIdx)

  ##
  # \brief Function to implement masks specified with VPParams on a image like array.
  # \param imageLike is 2D array with either 1 channel (B/W) or 3 channels (RGB)
  #
  def _clearMaskedArea(self, imageLike):
    if self.vpParams:
      for mask in self.vpParams:
        if mask['type'] == 'negative':
          topIdx = int(mask['y']); btmIdx = int(mask['y']) + int(mask['height'])
          lftIdx = int(mask['x']); rhtIdx = int(mask['x']) + int(mask['width'])
          imageLike[topIdx:btmIdx, lftIdx:rhtIdx,] = 0
      positiveMaskExists = False
      for mask in self.vpParams:
        if mask['type'] == 'positive':
          positiveMaskExists = True
          break
      if positiveMaskExists:
        tempImage = np.zeros(imageLike.shape)
        for mask in self.vpParams:
          if mask['type'] == 'positive':
            topIdx = int(mask['y']); btmIdx = int(mask['y']) + int(mask['height'])
            lftIdx = int(mask['x']); rhtIdx = int(mask['x']) + int(mask['width'])
            tempImage[topIdx:btmIdx, lftIdx:rhtIdx,] = imageLike[topIdx:btmIdx, lftIdx:rhtIdx,]
        imageLike = tempImage
    return imageLike

  ##
  # \brief Function to create full debug images.
  # This function is only used for stand-alone mode. 
  # \param saveDbgFile path to save .dbg images
  # \param saveKrnFile path to save .krn images
  #
  def _debug(self, saveDbgFile, saveKrnFile):
    maxWidth       = 800
    distanceBtwImg = 20

    resizedWidth   = self.width
    resizedHeight  = self.height
    if resizedWidth > maxWidth:
      scaleRatio = maxWidth / resizedWidth
      resizedWidth  = int(resizedWidth * scaleRatio)
      resizedHeight = int(resizedHeight* scaleRatio)
    emptyCanvas  = np.ones((2*resizedHeight + 3*distanceBtwImg, 2*resizedWidth + 3*distanceBtwImg, self.channel))
    emptyCanvas *= 35
    resizedExpected = cv2.resize(self.expectedImage, (resizedWidth, resizedHeight), interpolation=cv2.INTER_AREA)
    resizedObserved = cv2.resize(self.observedImage, (resizedWidth, resizedHeight), interpolation=cv2.INTER_AREA)
    resizedDiff     = cv2.resize(self.diffImage,     (resizedWidth, resizedHeight), interpolation=cv2.INTER_AREA)

    emptyCanvas[distanceBtwImg:resizedHeight+distanceBtwImg,distanceBtwImg:resizedWidth+distanceBtwImg,] = resizedExpected
    emptyCanvas[distanceBtwImg:resizedHeight+distanceBtwImg,resizedWidth+2*distanceBtwImg:2*resizedWidth+2*distanceBtwImg,] = resizedObserved
    emptyCanvas[resizedHeight+2*distanceBtwImg:2*resizedHeight+2*distanceBtwImg,resizedWidth+(-resizedWidth//2+distanceBtwImg):2*resizedWidth+(-resizedWidth//2+distanceBtwImg),] = resizedDiff
    
    kernel       = (SimilarityCheck.KERNEL_SIZE[0] * resizedWidth  / self.width, SimilarityCheck.KERNEL_SIZE[1] * resizedHeight / self.height)
    failurePoint = (-kernel[0], -kernel[1])
    for pxDensityDict in self.pxDensityDictList:
      nextFailurePoint     = (pxDensityDict['AnchorPixel'][0] * resizedWidth  / self.width, pxDensityDict['AnchorPixel'][1] * resizedHeight / self.height)
      if ((failurePoint[0] - nextFailurePoint[0])**2 + (failurePoint[1] - nextFailurePoint[1])**2) > (kernel[0]**2 + kernel[1**2]):
        failurePoint = nextFailurePoint
        rctExp = self._getKernelAroundAnchor(failurePoint, kernel)
        rctExp = (rctExp[0]+distanceBtwImg, rctExp[1]+distanceBtwImg, rctExp[2]+distanceBtwImg, rctExp[3]+distanceBtwImg)
        rctObs = (rctExp[0], rctExp[1]+resizedWidth+distanceBtwImg, rctExp[2], rctExp[3]+resizedWidth+distanceBtwImg)
        rctDff = (rctExp[0]+resizedHeight+distanceBtwImg, rctExp[1]+resizedWidth-resizedWidth//2, rctExp[2]+resizedHeight+distanceBtwImg, rctExp[3]+resizedWidth-resizedWidth//2)
        cv2.rectangle(emptyCanvas, (int(rctExp[1]),int(rctExp[0])), (int(rctExp[3]),int(rctExp[2])), (0,0,255), 1)
        cv2.rectangle(emptyCanvas, (int(rctObs[1]),int(rctObs[0])), (int(rctObs[3]),int(rctObs[2])), (0,0,255), 1)
        cv2.rectangle(emptyCanvas, (int(rctDff[1]),int(rctDff[0])), (int(rctDff[3]),int(rctDff[2])), (0,0,255), 1)

    cv2.imwrite(saveDbgFile, emptyCanvas)
    
    from matplotlib import pyplot as plt
    totOutput = 3
    pltPerOut = 2
    if len(self.pxDensityDictList) < totOutput*pltPerOut:
      totOutput = max(1, len(self.pxDensityDictList) // pltPerOut)
      if len(self.pxDensityDictList) < totOutput*pltPerOut:
        totOutput = 1
        pltPerOut = 1
    for out in range(totOutput):
      fig, axes = plt.subplots(nrows=pltPerOut,ncols=1, figsize=(15.0, 15.0), dpi=250)
      for idx in range(pltPerOut):
        pxDensityDict = self.pxDensityDictList[len(self.pxDensityDictList) - (idx+1) - (pltPerOut*out)]
        rct = self._getKernelAroundAnchor(pxDensityDict['AnchorPixel'], SimilarityCheck.KERNEL_SIZE)
        score, expKernel, obsKernel = self._calculatesimilarityIndexIndex(rct, debug=True)
        try:
          axes[idx].set_title("similarityIndex Score : {:5f}".format(score), pad=50)
          axes[idx].tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
          axes[idx]._frameon = False
        except TypeError:
          axes.set_title("similarityIndex Score : {:5f}".format(score), pad=50)
          axes.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
          axes._frameon = False
        axExp = fig.add_subplot(pltPerOut,2,2*idx+1)
        axExp.imshow(expKernel)
        axExp.set_title('Kernel From Exp. Image')
        axObs = fig.add_subplot(pltPerOut,2,2*idx+2)
        axObs.imshow(obsKernel)
        axObs.set_title('Kernel From Obs. Image')
        for x in range(expKernel.shape[0]):
          for y in range(expKernel.shape[1]):
            if np.sum(np.abs(expKernel[x,y,] - obsKernel[x,y,]))/3 > 0:
              axExp.text(y-0.3, x, f"{np.sum(expKernel[x,y,])/3:.1f}", fontdict=dict(size=10, color='red'), bbox=dict(fill=False, edgecolor='white', linewidth=1))
              axObs.text(y-0.3, x, f"{np.sum(obsKernel[x,y,])/3:.1f}", fontdict=dict(size=10, color='red'), bbox=dict(fill=False, edgecolor='white', linewidth=1))
      plt.subplots_adjust(hspace=0.3)
      plt.savefig(saveKrnFile.split('.png')[0] + '_' + str(out+1) + '.png')
      plt.clf(); plt.close()

  ##
  # \brief Representation of SimilarityCheck object for logging purposes.
  #
  def __repr__(self):
    result  = 'PASSED' if self.getFinalVerdict() else 'FAILED'
    if self.verdictVerbose.options.IsItDefault():
      returnString  = 'Default options are used to give verdict'
    else:
      returnString  = 'Options that are used to give verdict is as follows'
      returnString += '\n\t\t\tKernel Size         : {}'.format(self.verdictVerbose.options.ks)
      returnString += '\n\t\t\tFailure Ratio       : {}'.format(self.verdictVerbose.options.failureRatio)
      returnString += '\n\t\t\tLocal Failure Ratio : {}'.format(self.verdictVerbose.options.localFailureRatio)
      returnString += '\n\t\t\tSimilarity Index    : {}'.format(self.verdictVerbose.options.similarityIndex)
    returnString += '\nSimilarity final verdit: {}'.format(result)
    if self.verdictVerbose.verbose == SimilarityCheck.VerdictVerboseList.GlobalPixelThresholdExceeded:
      returnString += '\nFailure Ratio has been exceeded.'
    elif self.verdictVerbose.verbose == SimilarityCheck.VerdictVerboseList.LocalPixelThresholdSatisfied:
      if self.verdictVerbose.summary['similarityIndexRequired']:
        returnString += '\nSome calculated Local Failure Ratios are above given threshold.'
        returnString += '\nFinal verdict is PASS as similarityIndex Score is above threshold for all of them.'
        returnString += '\nMaximum Local Failure Ratio is                 : {} '.format(self.verdictVerbose.summary['FailureDensity'])
        returnString += '\nMaximum Local Failure Ratio location           : {} '.format(self.verdictVerbose.summary['AnchorPixel'])
        returnString += '\nsimilarityIndex Score at Max Local Failure Ratio location : {} '.format(self._calculatesimilarityIndexIndex(self.verdictVerbose.summary['KernelWindow']))
      else:
        returnString += '\nAll Local Failure Ratios are below given threshold. No Similarity Index Score is calculated.'
    elif self.verdictVerbose.verbose == SimilarityCheck.VerdictVerboseList.similarityIndexScoreThresholdNotAchieved:
        returnString += '\nLocal Failure Ratio at {} above threshold'.format(self.verdictVerbose.summary['AnchorPixel'])
        returnString += 'and failed to satisfy required similarityIndex score'
        returnString += '\nLocal Failure Ratio is                            : {} '.format(self.verdictVerbose.summary['FailureDensity'])
        returnString += '\nLocal Failure Ratio location                      : {} '.format(self.verdictVerbose.summary['AnchorPixel'])
        returnString += '\nSimilarity Index Score at above Local Failure Ratio location  : {} '.format(self._calculatesimilarityIndexIndex(self.verdictVerbose.summary['KernelWindow']))
    
    return returnString + '\n'

  ##
  # \brief Helper function to correct limits of two integer variables w.r.t. given limits
  # \param x1 LHS value
  # \param x2 RHS value
  # \param x1Limit LHS limit for x1
  # \param x2Limit RHS limit for x2
  #
  @classmethod
  def _shiftBoundries(cls, x1, x2, x1Limit, x2Limit):
    if x1 < x1Limit and x2 > x2Limit: 
      x1 = 0
      x2 = x2Limit
    elif x1 < x1Limit:
      x2  = x2 + (x1Limit - x1)
      x1  = x1Limit
    elif x2 > x2Limit:
      x1  = x1 - (x2 - x2Limit)
      x2  = x2Limit
    return max(x1, x1Limit), min(x2, x2Limit)

##
# \brief Heler function for stand-alone debugging useage to walk thorugh given directiory
# to find pairs of expected and observed images, and prepare final tupple for 
# expected, observed, differance, matrix, debug and kernel image paths.
#
def walkFolderForTestImages(folderPath):
  expImages = []
  obsImages = []
  dffImages = []
  mtxImages = []
  dbgImages = []
  krnImages = []

  coupledTestImages = []
  testImages        = [x for x in sorted(os.listdir(folderPath)) if any(key in x for key in ['exp', 'obs'])]
  for testImage in testImages:
    if 'exp' in testImage:
      if testImage.replace('exp', 'obs') in testImages:
        if testImage not in coupledTestImages:
          coupledTestImages.append(testImage)
        if testImage.replace('exp', 'obs') not in coupledTestImages:
          coupledTestImages.append(testImage.replace('exp', 'obs'))
    if 'obs' in testImage:
      if testImage.replace('obs', 'exp') in testImages:
        if testImage not in coupledTestImages:
          coupledTestImages.append(testImage)
        if testImage.replace('obs', 'exp') not in coupledTestImages:
          coupledTestImages.append(testImage.replace('obs', 'exp'))
  
  coupledTestImages = sorted(coupledTestImages)
  for idx in range(0, len(coupledTestImages), 2):
    expImages.append(os.path.join(folderPath, coupledTestImages[idx]))
    obsImages.append(os.path.join(folderPath, coupledTestImages[idx+1]))
    dffImages.append(os.path.join(folderPath, coupledTestImages[idx].replace('exp', 'dff')))
    mtxImages.append(os.path.join(folderPath, coupledTestImages[idx].replace('exp', 'mtx')))
    dbgImages.append(os.path.join(folderPath, coupledTestImages[idx].replace('exp', 'dbg')))
    krnImages.append(os.path.join(folderPath, coupledTestImages[idx].replace('exp', 'krn')))

  testImagesTuple = zip(expImages, obsImages, dffImages, mtxImages, dbgImages, krnImages)
  return testImagesTuple

##
# \brief Heler function for stand-alone debugging useage to prepare final tupple for 
# expected, observed, differance, matrix, debug and kernel image paths.
#
def getTestImageTupple(folderPath, expectedImage, observedImage):
  expImages = []; obsImages = []; dffImages = []
  mtxImages = []; dbgImages = []; krnImages = []

  expImages.append(os.path.join(expectedImage))
  obsImages.append(os.path.join(observedImage))
  dffImages.append(os.path.join(folderPath, os.path.basename(expectedImage).replace('exp', 'dff')))
  mtxImages.append(os.path.join(folderPath, os.path.basename(expectedImage).replace('exp', 'mtx')))
  dbgImages.append(os.path.join(folderPath, os.path.basename(expectedImage).replace('exp', 'dbg')))
  krnImages.append(os.path.join(folderPath, os.path.basename(expectedImage).replace('exp', 'krn')))

  testImagesTuple = zip(expImages, obsImages, dffImages, mtxImages, dbgImages, krnImages)
  return testImagesTuple

##
# \brief Helper function to run diagnostic for SimilarityCheck class
#
def runDiagnostic(testImagesTuple, vpParams=None):
  for exp, obs, dff, mtx, dbg, krn in testImagesTuple:
    print('\n*********** {} ***********'.format(exp.split('_exp')[0]))

    discTol = SimilarityCheck(exp, obs, vpParams=vpParams)
    discTol.saveImage(SimilarityCheck.ImageList.DifferanceImage, dff)
    discTol.saveDiffMatrix(mtx)
    
    discTol._debug(dbg, krn)

    print(discTol)

##
# \brief Main method to call for stand-alone test
#    
def main():
  vpParams = None
  if args.MASKS:
    vpParams = JsonToVPParameters(args.MASKS[0]).getInstance()
  if args.folderPath and any(a for a in [args.expectedImage, args.observedImage, args.outFolderPath]):
    parser.error("Please provide either single image to directory path to work on or a directory of images. For usage please use -h/--help.")
  elif args.folderPath:
    runDiagnostic(walkFolderForTestImages(args.folderPath[0]), vpParams=vpParams)
  elif args.expectedImage and args.observedImage and args.outFolderPath:
    runDiagnostic(getTestImageTupple(args.outFolderPath[0], args.expectedImage[0], args.observedImage[0]), vpParams)
  else:
    parser.error("Single image run mode requires '*_exp.png', '*_obs.png' images and output directory. For usage please use -h/--help.")

if __name__ == "__main__":
  main()