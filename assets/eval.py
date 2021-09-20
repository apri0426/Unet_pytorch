import math
import sys
import os
import time
import numpy as np
import scipy
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_nrmse as nrmse
import cv2
import warnings
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance
#from scipy.misc import imsave
#from scipy.ndimage import imread
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps


font = { 'size'   : 25, 'family':'serif'}
matplotlib.rc('font', **font)
##
# Globals
##

warnings.filterwarnings('ignore')

# specify resized image sizes
height = 168.0
width = 480

##
# Functions
##



"""
consider all pixels in the golden image 
return the normalized RMSE for these pixels 
"""

def evaluate(golden, predict):
    delta = predict-golden
    # compute normalized RMSE for nonzero entries 
    #return np.linalg.norm(delta, ord=2) / np.linalg.norm(golden, ord=2)
    return np.linalg.norm(delta, ord=2) / (golden.max()-golden.min()) / np.sqrt(golden.size)


"""
consider the nonzero pixels in the golden image 
return the normalized RMSE for these pixels 
"""
def evaluate_nonzero(golden, predict):
    nonzero_idx = np.where(golden > 0)
    nonzero_golden = golden[nonzero_idx]
    nonzero_predict = predict[nonzero_idx]
    nonzero_delta = nonzero_predict-nonzero_golden
    # compute normalized RMSE for nonzero entries 
    #return np.linalg.norm(nonzero_delta, ord=2) / np.linalg.norm(golden, ord=2)
    return np.linalg.norm(nonzero_delta, ord=2) / (golden.max()-golden.min()) / np.sqrt(golden.size)




def get_img(img, norm_size=False, norm_exposure=False):
  '''
  Prepare an image for image processing tasks
  '''
  # flatten returns a 2d grayscale array
  #img = imread(path, flatten=True).astype(int)
  # resizing returns float vals 0:255; convert to ints for downstream tasks
  if norm_size:
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
  if norm_exposure:
    img = normalize_exposure(img)
  return img


def get_histogram(img):
  '''
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w)


def normalize_exposure(img):
  '''
  Normalize the exposure of an image.
  '''
  img = img.astype(int)
  hist = get_histogram(img)
  # get the sum of vals accumulated by each position in hist
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize each position in the output image
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)


def earth_movers_distance(path_a, path_b):
  '''
  Measure the Earth Mover's distance between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  hist_a = get_histogram(img_a)
  hist_b = get_histogram(img_b)
  return wasserstein_distance(hist_a, hist_b)


def structural_sim(path_a, path_b):
  '''
  Measure the structural similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  img_a = get_img(path_a)
  img_b = get_img(path_b)
  sim, diff = compare_ssim(img_a, img_b, full=True)
  return sim


def pixel_sim(path_a, path_b):
  '''
  Measure the pixel-level similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  return np.sum(np.absolute(img_a - img_b)) / (height*width) / 255


def pixel_sim_nonzero(path_a, path_b):
  '''
  Measure the pixel-level similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  ind=np.where(img_b>np.min(img_b))
  return np.sum(np.absolute(img_a[ind] - img_b[ind])) / ind[0].shape[0] / 255.0








def sift_sim(path_a, path_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  # initialize the sift feature detector
  orb = cv2.ORB()

  # get the images
  img_a = path_a
  img_b = path_b

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)
  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  #imgg=drawMatches(img_a, kp_a, img_b, kp_b, matches)
  #cv2.imwrite('temp.png',imgg)
  #pdb.set_trace()
  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / float(len(matches))






def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1])
    out[:rows2,cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)

    return out
placer = ['pass']
benchs = ['1']
for bench_n in benchs:
    ssim_eval_h = []
    ssim_eval_v = []
    nrms_v = []
    lstructural_sim_v = []
    lpixel_sim_v = []
    lsift_sim_v = []
    error_all_h = []
    error_nz_h = []

    lemd_v = []
    nrms_h = []
    lstructural_sim_h = []
    lpixel_sim_h = []
    lsift_sim_h = []
    lemd_h = []
    error_all_v = []
    error_nz_v = []
    for k in range(len(placer)):
        ############################
        #bench_n=7
        base=False
        save_sep_images=False
        ############################

        # bench=('Congestion_%d_self/' %(bench_n))
        # res=('cong_%d_self/' %(bench_n))
        # base_bench=('test_cong_%d/' %(bench_n))

        # dir_real='/home/zhuominchai/deliv/model/datasets/Congestion_01/test_B/'
        dir_real=r'C:/Users/41917/desktop/assets'

        # dir_rudy='/home/zhuominchai/deliv/model/datasets/Congestion_01/test_A/'
        dir_rudy='C://Users//41917//Desktop//assets'

        if base:
            dir_pred='/home/local/eda07/malawieh/FPGA/FPGACong/src/pix2pix-tensorflow/'+base_bench
            save_sep_images=False
        else:
            # dir_pred='/home/zhuominchai/deliv/model/results/cong_1_11/test_latest/'
            dir_pred='C://Users//41917//Desktop//assets'
            #最后需要//

        # image_count = len(os.listdir(dir_real))
        #
        # ssim_eval_h=np.zeros((image_count,))
        # ssim_eval_v=np.zeros((image_count,))
        # nrms_v=np.zeros((image_count,))
        # lstructural_sim_v=np.zeros((image_count,))
        # lpixel_sim_v=np.zeros((image_count,))
        # lsift_sim_v=np.zeros((image_count,))
        # error_all_h=np.zeros((image_count,))
        # error_nz_h=np.zeros((image_count,))
        #
        # lemd_v=np.zeros((image_count,))
        # nrms_h=np.zeros((image_count,))
        # lstructural_sim_h=np.zeros((image_count,))
        # lpixel_sim_h=np.zeros((image_count,))
        # lsift_sim_h=np.zeros((image_count,))
        # lemd_h=np.zeros((image_count,))
        # error_all_v=np.zeros((image_count,))
        # error_nz_v=np.zeros((image_count,))



        # i = 0
        for n in range(1,11):
                #image = Image.open(dirr+fil)
                #inverted_image = PIL.ImageOps.invert(image)
                #inverted_image.save(dir2+fil)
                fil = '_' + str(n) + '.png'

                pred_name= 'pred' + fil
                # pred_name=dir_pred+'images/'+fil[:-4]+'_label_synthesized_image.png'
                if not os.path.exists(pred_name):
                    continue
                pred = cv2.imread(pred_name)
                real = cv2.imread('gt' + fil)
                #pred=img[:,0:256,2]
                #real=img[:,256:2*256,2]

                rudy= cv2.imread('ori' + fil)
                #rudy=feat[:,:,1]+feat[:,:,2]
                #rudy=(0.5*(feat[:,:,1]+feat[:,:,2])).astype('uint8')
                # print(fil)
                # print('pass')
                pred = cv2.resize(pred, (480, 168))
                real = cv2.resize(real, (480, 168), fx=1, fy=1)
                rudy = cv2.resize(rudy, (480, 168), fx=1, fy=1)
                pred_v=pred[:,:,1]
                # print(pred_v.shape)
                # print(pred_v.mean())
                #pred_v=rudy[:,:,2]
                pred_h=pred[:,:,2]
                # print(real.size)
                real_v=real[:,:,1]
                real_h=real[:,:,2]
                # print(fil)
                # print(pred_v.shape)
                # print(pred_v.mean())
                #pred_h=rudy[:,:,1]

                '''
                pred_v=np.zeros(pred.shape)
                pred_v[:,:,2]=pred[:,:,2]
                pred_h=np.zeros(pred.shape)
                pred_h[:,:,1]=pred[:,:,1]
                real_v=np.zeros(real.shape)
                real_v[:,:,2]=real[:,:,2]
                real_h=np.zeros(real.shape)
                real_h[:,:,1]=real[:,:,1]
                '''
                indv=np.where(pred_v>0)
                pred_v_nz=pred_v[indv]
                real_v_nz=real_v[indv]
                indh=np.where(pred_h>0)
                pred_h_nz=pred_h[indh]
                real_h_nz=real_h[indh]




                # #nrms_v[i]=nrmse(real_v_nz,pred_v_nz,'min-max')
                # nrms_v[i]=nrmse(real_v,pred_v,'min-max')
                # ssim_eval_v[i]=ssim(pred_v, real_v)
                # lstructural_sim_v[i]= structural_sim(real_v, pred_v)
                # lpixel_sim_v[i]= pixel_sim(real_v, pred_v)
                # #lpixel_sim_v[i]= pixel_sim_nonzero(real_v, pred_v)
                # #lsift_sim_v[i]= sift_sim(real_v, pred_v)
                # lemd_v[i]= earth_movers_distance(real_v, pred_v)
                # error_all_v[i]=evaluate(real_v,pred_v)
                # error_nz_v[i]=evaluate_nonzero(real_v,pred_v)
                #
                # #nrms_h[i]=nrmse(real_h_nz,pred_h_nz,'min-max')
                # nrms_h[i]=nrmse(real_h,pred_h,'min-max')
                # ssim_eval_h[i]=ssim(pred_h, real_h)
                # lstructural_sim_h[i]= structural_sim(real_h, pred_h)
                # lpixel_sim_h[i]= pixel_sim(real_h, pred_h)
                # #lpixel_sim_h[i]= pixel_sim_nonzero(real_h, pred_h)
                # #lsift_sim_h[i]= sift_sim(real_h, pred_h)
                # lemd_h[i]= earth_movers_distance(real_h, pred_h)
                # error_all_h[i]=evaluate(real_h,pred_h)
                # error_nz_h[i]=evaluate_nonzero(real_h,pred_h)

                nrms_v.append(nrmse(real_v,pred_v,'min-max'))
                ssim_eval_v.append(ssim(pred_v, real_v))
                lstructural_sim_v.append(structural_sim(real_v, pred_v))
                lpixel_sim_v.append(pixel_sim(real_v, pred_v))
                lemd_v.append(earth_movers_distance(real_v, pred_v))
                error_all_v.append(evaluate(real_v,pred_v))
                error_nz_v.append(evaluate_nonzero(real_v,pred_v))

                nrms_h.append(nrmse(real_h,pred_h,'min-max'))
                ssim_eval_h.append(ssim(pred_h, real_h))
                lstructural_sim_h.append(structural_sim(real_h, pred_h))
                lpixel_sim_h.append(pixel_sim(real_h, pred_h))
                lemd_h.append(earth_movers_distance(real_h, pred_h))
                error_all_h.append(evaluate(real_h,pred_h))
                error_nz_h.append(evaluate_nonzero(real_h,pred_h))

                if save_sep_images:
                    if not os.path.isdir(dir_pred+'H/'):
                        os.mkdir(dir_pred+'H/')
                        os.mkdir(dir_pred+'V/')
                    pred_v=np.zeros(pred.shape)
                    pred_v[:,:,2]=pred[:,:,2]
                    pred_h=np.zeros(pred.shape)
                    pred_h[:,:,1]=pred[:,:,1]
                    real_v=np.zeros(real.shape)
                    real_v[:,:,2]=real[:,:,2]
                    real_h=np.zeros(real.shape)
                    real_h[:,:,1]=real[:,:,1]

                    cv2.imwrite(dir_pred+'V/P_'+fil,pred_v)
                    cv2.imwrite(dir_pred+'H/P_'+fil,pred_h)
                    cv2.imwrite(dir_pred+'V/R_'+fil,real_v)
                    cv2.imwrite(dir_pred+'H/R_'+fil,real_h)
                # i += 1



        if save_sep_images:
            if not os.path.isdir(dir_pred + 'H/'):
                os.mkdir(dir_pred + 'H/')
                os.mkdir(dir_pred + 'V/')
            pred_v = np.zeros(pred.shape)
            pred_v[:, :, 2] = pred[:, :, 2]
            pred_h = np.zeros(pred.shape)
            pred_h[:, :, 1] = pred[:, :, 1]
            real_v = np.zeros(real.shape)
            real_v[:, :, 2] = real[:, :, 2]
            real_h = np.zeros(real.shape)
            real_h[:, :, 1] = real[:, :, 1]

            cv2.imwrite(dir_pred + 'V/P_' + fil, pred_v)
            cv2.imwrite(dir_pred + 'H/P_' + fil, pred_h)
            cv2.imwrite(dir_pred + 'V/R_' + fil, real_v)
            cv2.imwrite(dir_pred + 'H/R_' + fil, real_h)






    print("####################Bench %g" % (int(bench_n)))
    # print(" %g %g %g %g %g %g %g %g " % (np.mean(nrms_v),np.mean(ssim_eval_v),np.mean(lpixel_sim_v  ), np.mean(lemd_v),np.mean(nrms_h),np.mean(ssim_eval_h),np.mean(lpixel_sim_h  ), np.mean(lemd_h) ))
    print("Horizontal nrmse %g" % (np.mean(np.array(nrms_h, dtype = float))))
    print("Vertical nrmse %g" % (np.mean(np.array(nrms_v, dtype = float))))
    print("Horizontal ssim %g" % (np.mean(np.array(ssim_eval_h, dtype = float))))
    print("Vertical ssim %g" % (np.mean(np.array(ssim_eval_v, dtype = float))))
    print("Horizontal pixel error %g" % (np.mean(np.array(lpixel_sim_h, dtype = float))))
    print("Vertical pixel error %g" % (np.mean(np.array(lpixel_sim_v, dtype = float))))
    print("Horizontal earth mover distance %g" % (np.mean(np.array(lemd_h, dtype = float))))
    print("Vertical earth mover distance %g" % (np.mean(np.array(lemd_v, dtype = float))))





'''
bins = np.linspace(0.0, 1, 50)

plt.hist(ssim_timing_sha, bins, alpha=0.5, label='S')
plt.hist(ssim_timing_rayg, bins, alpha=0.5, label='R')
plt.legend(loc='upper right')
plt.show(block=False)

plt.figure()

plt.hist(nrms_sha, bins, alpha=0.5, label='S')
plt.hist(nrms_rayg, bins, alpha=0.5, label='R')
plt.legend(loc='upper right')
plt.show(block=False)



plt.figure()

bins = np.linspace(0.8, 1, 25)
plt.hist(lstructural_sim_h, bins , label='CGAN')
lgd=plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True)
#plt.show(block=False)
plt.savefig('ssim_hist_h.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure()
bins = np.linspace(0, 0.3, 20)
plt.hist(nrms_h, bins , label='CGAN')
lgd=plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True)
plt.savefig('nrms_hist_h.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')


plt.figure()

bins = np.linspace(0.8, 1, 25)
plt.hist(lstructural_sim_v, bins , label='CGAN')
lgd=plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True)
#plt.show(block=False)
plt.savefig('ssim_hist_v.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure()
bins = np.linspace(0, 0.3, 20)
plt.hist(nrms_v, bins , label='CGAN')
lgd=plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True)
plt.savefig('nrms_hist_v.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')


plt.figure()

bins = np.linspace(0, 0.3, 25)
plt.hist(error_all_h, bins , label='CGAN')
lgd=plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True)
#plt.show(block=False)
plt.savefig('error_all_hist_h.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure()
bins = np.linspace(0, 0.3, 25)
plt.hist(error_nz_h, bins , label='CGAN')
lgd=plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True)
plt.savefig('error_nz_h.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')


plt.figure()

bins = np.linspace(0, 0.3, 25)
plt.hist(error_all_v, bins , label='CGAN')
lgd=plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True)
#plt.show(block=False)
plt.savefig('error_all_v.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure()
bins = np.linspace(0, 0.3, 25)
plt.hist(error_nz_v, bins , label='CGAN')
lgd=plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True)
plt.savefig('error_nz_v.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')


'''















