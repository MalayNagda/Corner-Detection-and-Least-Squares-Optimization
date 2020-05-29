import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import ndimage


#Black blank image
blank_image = np.zeros(shape=[300, 300,3], dtype=np.uint8)
num_rows, num_cols = blank_image.shape[:2]

#pts=np.array([[0,0],[300,0],[300,300],[0,300]],np.int32)
#cv2.fillPoly(blank_image,[pts],(0,255,255))

pts = np.array([[185,125],[135,115],[125,180],[170,165]], np.int32)
#White irregular quadrilateral of 50x50 pixels approx in the center
cv2.fillPoly(blank_image,[pts],(255,255,255))
cv2.imshow("Black img and white Quadri", blank_image)

#Translation of the image 
translation_matrix = np.float32([ [1,0,30], [0,1,100] ])
img_translation = cv2.warpAffine(blank_image, translation_matrix, (num_cols+30, num_rows+100))
cv2.imshow("Translated Image", img_translation)
rows,cols = blank_image.shape[:2]
num_rows, num_cols = blank_image.shape[:2]

#Rotation of the translated image
M = cv2.getRotationMatrix2D((rows/2,cols/2),45,1)
num_rows, num_cols = img_translation.shape[:2]

abs_cos = abs(M[0,0]) 
abs_sin = abs(M[0,1])
# find the new width and height bounds
bound_w = int(num_rows * abs_sin + num_cols * abs_cos)
bound_h = int(num_rows * abs_cos + num_cols * abs_sin)
# subtract old image center (bringing image back to original) and adding the new image center coordinates
M[0, 2] =M[0, 2] + bound_w/2 - num_cols/2
M[1, 2] =M[1, 2] +bound_h/2 - num_rows/2

rotated_mat = cv2.warpAffine(img_translation, M, (bound_w+30, bound_h))

cv2.imshow("Rotated_and_translated", rotated_mat)
#cv2.imshow("Black and w", rotated_mat1)
cv2.imwrite('Black and quadri.png',blank_image)
cv2.imwrite('Translated.png',img_translation)
cv2.imwrite('Rotated_and_translated.png',rotated_mat)

cv2.waitKey(0)
cv2.destroyAllWindows()

######## Corner detection ######################
def corner_detect(image):
    
    img=image
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    imarr = np.asarray(img, dtype=np.float64)
    ix = ndimage.sobel(imarr, 0)
    iy = ndimage.sobel(imarr, 1)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy
    ix2 = ndimage.gaussian_filter(ix2, sigma=2)
    iy2 = ndimage.gaussian_filter(iy2, sigma=2)
    ixy = ndimage.gaussian_filter(ixy, sigma=2)
    c, l = imarr.shape[:2]
    result = np.zeros((c, l))
    r = np.zeros((c, l))
    img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rmax = 0
    for i in range(c):
        for j in range(l):
            #print('test ', j)
            m = np.array([[ix2[i, j], ixy[i, j]], [ixy[i, j], iy2[i, j]]], dtype=np.float64)
            #np.reshape(m,(2,2))
            
            r[i, j] = np.linalg.det(m) - (0.04 * (np.power(np.trace(m), 2)))
            if r[i, j] > rmax:
                rmax = r[i, j]
    for i in range(c - 1):
        #print(". .")
        for j in range(l - 1):
            if r[i, j] > 0.075 * rmax and r[i, j] > r[i - 1, j - 1] and r[i, j] > r[i - 1, j + 1] \
                    and r[i, j] > r[i + 1, j - 1] and r[i, j] > r[i + 1, j + 1]:
                result = np.transpose(result)
                result[i, j] = 1
                result = np.transpose(result)
                pc, pr = np.where(result == 1)
                img.itemset((i,j,0),0)
                img.itemset((i,j,1),0)
                img.itemset((i,j,2),255)
    
    for corner in range(0,len(pc)):
        #pc,pr= corner.ravel()
        pc1=pc[corner]
        pr1=pr[corner]
        cv2.circle(img,(pc1,pr1),3,255,-1)
    
    cv2.imshow('cors',img)
    #cv2.imwrite('Detected_corners.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pc,pr,img

pc1,pr1,image=corner_detect(blank_image)
cv2.imwrite('Detected_corners_1.png',image)
pc1=np.delete(pc1,[1,3,5,7],0)
pr1=np.delete(pr1,[1,3,5,7],0)
c_x,c_y=sum(pc1)/len(pc1),sum(pr1)/len(pr1) #Centroid of the quadrilateral
p1=[pc1,pr1] #The four detected corners of quadrilateral
c1=[[c_x,c_x,c_x,c_x],[c_y,c_y,c_y,c_y]]    
h1=np.subtract(p1,c1)
#h1=np.transpose(h1)

pc2,pr2,image=corner_detect(rotated_mat)
cv2.imwrite('Detected_corners_2.png',image)
pc2=np.delete(pc2,[0,2,4,6,8],0)
pr2=np.delete(pr2,[0,2,4,6,8],0)
c_xr,c_yr=sum(pc2)/len(pc2),sum(pr2)/len(pr2) #Centroid of the transformed quadrilateral
p2=[pc2,pr2] #The four detected corners of the transformed quadrilateral
c2=[[c_xr,c_xr,c_xr,c_xr],[c_yr,c_yr,c_yr,c_yr]]
h2=np.subtract(p2,c2)
h2=np.transpose(h2)

############### Recover rotation and translation #######################
H=np.dot(h2,h1)
u,s,v = np.linalg.svd(H)
u=np.transpose(u)
v=np.transpose(v)
R=np.dot(v,u)
t=np.add(np.dot(-R,np.transpose(c2)),np.transpose(c1))
c=np.add(np.dot(R,np.transpose(p2)),t) #Recovered translation and rotation
print('\n',c)
print('\n',np.transpose(p1))
#Mean square error between the corners detected in the untransformed and transformed quadrilateral
rms = sqrt(mean_squared_error(np.transpose(p1), c)) 
print('\n',rms)