import math, sys, time, glob, os
import geojson
from openslide import open_slide, deepzoom
import matplotlib.pyplot as plt
from PIL import ImageStat, Image
from skimage.measure import approximate_polygon
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from skimage.draw import polygon

from collections import OrderedDict

Image.MAX_IMAGE_PIXELS = None #Dangerous if you don't trust your images

####
from scipy.ndimage import measurements, binary_fill_holes
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, dilation, square
from skimage.segmentation import watershed
from skimage.segmentation import boundaries


def segmentation(wsiPath,  maskPath=None):

    #val=96
    #pos=int(val/2)

    #maskLevel = 3
    minCellularRegionDensityPerTile = 0.2
    slide = open_slide(wsiPath)
    dz = deepzoom.DeepZoomGenerator(slide, tile_size=tileSize, overlap=0, limit_bounds=True)
    
    maskLevel = len(slide.level_downsamples)-1
    
    maskTileSize = tileSize*slide.level_downsamples[tileLevel]//slide.level_downsamples[maskLevel]
    dzLevel = dz.level_count-tileLevel-1

    if not maskPath is None:
        mask = cv2.imread(maskPath)
        mask = Image.fromarray(mask).convert("L")
        mask = mask.resize(slide.level_dimensions[maskLevel])
        fn = lambda x : 1 if x > 0 else 0
        mask = mask.point(fn, mode='1')
    else:
        mask = slide.read_region((0,0), maskLevel, slide.level_dimensions[maskLevel]).convert("L")
        fn = lambda x : 0 if x > 200 or x < 50 else 1
        mask = mask.point(fn, mode='1')

    for i in range(dz.level_tiles[dzLevel][0]):
        for j in range(dz.level_tiles[dzLevel][1]):
        
            sampleOutputPath = os.path.join(outputFolder, sampleName+'_'+str(i)+'_'+str(j)+'.png')
        
            if not os.path.exists(sampleOutputPath):        
                coord = dz.get_tile_coordinates(dzLevel, (i, j))
                if coord[2] != (tileSize, tileSize):
                    continue
                else:
                    coord = coord[0]
                cenX = (coord[0]+tileSize*slide.level_downsamples[tileLevel]//2)//slide.level_downsamples[maskLevel]
                cenY = (coord[1]+tileSize*slide.level_downsamples[tileLevel]//2)//slide.level_downsamples[maskLevel]
                maskRegion = mask.crop((cenX-(maskTileSize//2), cenY-(maskTileSize//2), cenX+(maskTileSize//2), cenY+(maskTileSize//2)))
                if ImageStat.Stat(maskRegion).mean[0] > minCellularRegionDensityPerTile:
                    tile = dz.get_tile(dzLevel, (i, j)).convert("RGB")
                    
                    ntile=Image.new('RGB', (tileSize+shift, tileSize+shift))
                    ntile.paste(tile,(half_shift,half_shift))
                    #plt.imshow(image)
                    
                    dataCounter = processOneTile(ntile, coord, np.int(slide.level_downsamples[tileLevel]))
                    #print(f"{i}/{dz.level_tiles[dzLevel][0]}, {j}/{dz.level_tiles[dzLevel][1]}, {len(GEOData)}, {dataCounter}")

                    if dataCounter > 0:
                        ##sampleOutputPath = os.path.join(outPath, os.path.splitext(sampleName)[0]+'_'+str(i)+'_'+str(j)+'.json')
                        ##with open(sampleOutputPath, 'w') as outfile:
                        ##    geojson.dump(GEOData[:dataCounter], outfile)
                        
                        value=json.dumps(GEOData[:dataCounter])
                        im=get_img_from_json_coords(tileSize+shift,value)
                        nim = im[half_shift:tileSize+half_shift,half_shift:tileSize+half_shift]
                        plt.imsave(sampleOutputPath, nim, cmap='gray')


if __name__ == '__main__':
    tileSize = int(sys.argv[1])
    dataPath = sys.argv[2]
    ext = sys.argv[3]
    modelPath = sys.argv[4]
    modelMode = sys.argv[5]
    nr_types = int(sys.argv[6])
    typeInfoPath = sys.argv[7]
    outPath = sys.argv[8]
    startInd = int(sys.argv[9])
    endInd = int(sys.argv[10])
    if len(sys.argv) == 13:
        maskPath = sys.argv[11]
        maskFileExt = sys.argv[12]
    else:
        maskPath = ''
        maskFileExt = ''

    tileLevel = 0 #0 means segmentation at 0.25mpp. Tested at 0.5mpp and working well but missing half of the nuclei (750k nuclei at 0.25mpp vs. 450k at  0.5mpp)
    batchSize = 1
    patchSize = tileSize
    
    shift=96
    half_shift=int(shift/2)
        
    batch = np.zeros((batchSize, patchSize+shift, patchSize+shift, 3))
    batchInfo = [None]*batchSize

    start = time.time()
    files = sorted(glob.glob(os.path.join(dataPath, '*'+ext)))
    
    with open(typeInfoPath) as f:
        type_info = geojson.load(f)
    print('Type info has been loaded.')

    model = load_model(modelPath, modelMode, nr_types=nr_types)
    print('Model has been loaded.')
    
    
    nfiles=len(files)
    print(f'{nfiles} folders have been read.', files)

    if endInd == -1:
        endInd = nfiles-1
    
    if startInd == -1:
        startInd = nfiles-1
    
    incr=1
    if startInd > endInd:
        incr=-1
        endInd-=2
    
    for i in range(startInd, endInd+1,incr):
        #sampleName = os.path.basename(files[i])
        #sampleMaskPath = os.path.join(maskPath, os.path.splitext(sampleName)[0]+maskFileExt)
        
        sampleName = os.path.basename(files[i]).replace(ext,"")
        sampleMaskPath = os.path.join(maskPath, sampleName+maskFileExt)
                
        outputFolder=os.path.join(outPath,sampleName)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        
        print(files[i], sampleMaskPath, sampleName)
        #print(sampleName)
        if not os.path.exists(sampleMaskPath):
            #print("No mask file... Generating the mask from WSI...")
            segmentation(files[i])
        else:
            segmentation(files[i], sampleMaskPath)

    finish = time.time()
    print("Nucleus segmentation done!")
    print(f'Elapsed time: {finish-start}')

