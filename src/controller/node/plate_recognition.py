#!/usr/bin/python3
import sys
import cv2
import numpy as np
import os
from itertools import groupby


if __name__ == "__main__":

    def all_equal(iterable):
        #https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
        g = groupby(iterable)
        return next(g, True) and not next(g, False)


    # image path
    path = os.path.dirname(os.path.realpath(__file__)) + "/"
    img_folder = "pictures"
    img_name = "img_1.png"
    full_path = os.path.join(path, img_folder, img_name)

    frame = cv2.imread(full_path)
    height,width,channels = frame.shape

    cv2.imshow("image", frame)
    cv2.waitKey(0)




    # lower_blue = np.array([70, 50, 70])
    # upper_blue = np.array([128, 255, 255])
    # mask_blue = cv2.inRange(frame, lower_blue, upper_blue)


    # frame_isolated = frame

    # for i in range(height):
    #     for j in range(width):
    #         rgb = np.asarray(frame[i][j])
    #         frame_isolated[i][j] = [255 if (all_equal(rgb) and i > 90 and i < 110) else 0 for i in rgb]

    # cv2.imshow("image", mask_blue)
    # cv2.waitKey(0)

    #frame_parking_isolated = [[[255 for j in i] if all_equal(i) else [0 for j in i] for i in row] for row in frame]
    #cv2.imshow("frame", np.asarray(frame_parking_isolated))


    #frame_parking_isolated = [frame[i]]
    
    # parking thing where R,G,B all same value
    #frame = 

    # binary, want thing we're looking for to be 1, everything else 0
    # to get centroid (x_bar, y_bar), do x_bar=M10/M00, y_bar=M01/M00
    # M00 is 0th order moment in x and y, area of non-zero pixels (road)
    # moment = cv2.moments(frame_binary_thresh_inv)
    # x_centroid = moment["m10"] / (moment["m00"])
    

    # #SIFT and FLANN matching
# #===================================
#     sift = cv2.SIFT_create()
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     MIN_MATCH_COUNT = 10

#     # image path
#     path = os.path.dirname(os.path.realpath(__file__)) + "/"
#     img_folder = "pictures"
#     img_name = "img_1.png"
#     full_path = os.path.join(path, img_folder, img_name)

#     # reference parking spot
#     ref_img_name = "p_reference_course.png"
#     ref_path_full = os.path.join(path, img_folder, ref_img_name)

#     # read in image, grayscale, get keypoints
#     ref_frame = cv2.imread(ref_path_full)
#     ref_frame_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
#     kp1, des1 = sift.detectAndCompute(ref_frame_gray, None) # no mask
#     output_frame = ref_frame
#     output_frame = cv2.drawKeypoints(ref_frame_gray, kp1, output_frame)

#     # FLANN: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#     # FLANN and homography: https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

#     # get second image keypoints
#     check_frame = cv2.imread(full_path)
#     check_frame_gray = cv2.cvtColor(check_frame, cv2.COLOR_BGR2GRAY)
#     kp2, des2 = sift.detectAndCompute(check_frame_gray,None)
#     matches = flann.knnMatch(des1, des2, k=2)

#     good = []
#     for m,n in matches: #two closest matches
#         if m.distance < 0.7*n.distance: #if matches are far enough apart
#             good.append(m)

#     if len(good)>MIN_MATCH_COUNT:
#         src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#         dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#         matchesMask = mask.ravel().tolist()
#         h,w,c = check_frame.shape # height, width, channels
#         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#         dst = cv2.perspectiveTransform(pts,M)
#         img2 = cv2.polylines(check_frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#     else:
#         print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#         matchesMask = None


#     draw_params = dict(matchColor = (0,255,0),
#                 singlePointColor = (255,0,0),
#                 flags = 2)

#     matched_image = cv2.drawMatchesKnn(ref_frame,kp1,check_frame,kp2,matches,None,**draw_params)
#     cv2.imshow("matched image", matched_image)
#     cv2.waitKey(0)
# #===================================