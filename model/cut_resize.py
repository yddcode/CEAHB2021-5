import numpy as np
import cv2, os
import matplotlib.image as pm
import matplotlib.pyplot as plt
# C:/Users/guohai/Desktop/dongba/zong/1   E:/guji_resizedata1/
rootDir = 'C:/Users/guohai/Desktop/dongba/zong/1/'
# rootDir = 'F:/data/76-4/'
listdir = os.listdir(rootDir)
a = 950
for file in listdir:
    
    if file.endswith('.jpg'):
        img1 = cv2.imread(rootDir + file, 0)
        # img1 = img1 - int(img1.mean())
        # img1[img1 < 0] = 0
        # # 数据类型转换
        # img1 = np.around(img1)
        # img1 = img1.astype(np.uint8)
        h, w = img1.shape
        
        hh = int(h/1300)
        ww = int(w/1300)
        img1 = cv2.resize(img1, (510*ww, hh*510))
        # cv2.imshow('img', img1)
        for i in range(hh-1):
            for j in range(ww):
                img = img1[j*510+200:(j+1)*510+200, i*510:(i+1)*510]
                # cv2.imshow('img1', img)
                # print(img.shape)
                cv2.waitKey(0)
                pm.imsave('E:/guji_resizedata510/dai_00'+str(a)+str(i)+str(j) + '.jpg', img, cmap='gray')
    a = a + 1
    if a == 976:
        break
        # print(img1.shape)
        # if h / w < 1.2 and h < 2000:
        #     img = cv2.resize(img1, (512, 512))
        #     cv2.imwrite('E:/guji_resizedata/shui80_00' + str(a)+'.jpg', img1)
        #     a = a + 1
        # else:
        #     img = img1[110:h-105, 170:w-160]
        # print(img.shape)
        # img = img1[30:h-10, 30:w-300] # naxi
        # rw = int(w / 1300 +0.5)
        # rh = int(h / 1300 +0.5)
        # print(rw, rh)
        # img = cv2.resize(img1[20:h-20, 20:w-20], (512*rw, 512*rh))
        # for i in range(rw):
        #     for j in range(rh):
        #         aimg = img[j*512:(j+1)*512, i*512:(i+1)*512]
        #         # cv2.imshow('img', aimg)
        #         cv2.imwrite('E:/guji_resizedata/dai_00' + str(a)+'_'+str(i)+str(j)+ '.jpg', aimg)
        #         cv2.waitKey(0)
        # a = a + 1

            # if w < 2000:
            #     print(file)
            #     ratio = h / w
            #     print(ratio)
            #     r = int(ratio+0.5)
            #     print(r)
            #     img = cv2.resize(img1, (512, int(r*512)))
            #     # wr = int(w / (h / 1000.0))
            #     for i in range(r):
            #         print(i)
            #         img1 = img[i*512:(i+1)*512, 0:512]

            #         print(img1.shape)
            #         # cv2.imshow('img'+str(i), img1)
            #         cv2.imwrite('E:/guji_resizedata/shui80_00' + str(a)+'_'+str(i) + '.jpg', img1)
                    
            #         cv2.waitKey(0)
            #     a = a + 1

            # if w > 1999 and w < 3000:
            #     print('> 2000:', file)
            #     r = int(h / w + 0.5)
            #     print(h / w)
            #     print(r)
            #     img1 = cv2.resize(img1, (1024, r*1024))
            #     for i in range(2):
            #         for j in range(2*r):
            #             print(img1.shape)
            #             img = img1[j*512:(j+1)*512, i*512:(i+1)*512]
            #             print(img.shape)
            #             # cv2.imshow('img'+str(i)+str(j), img)
            #             cv2.imwrite('E:/guji_resizedata/shui80_00' + str(a)+'_'+str(i)+str(j) + '.jpg', img)
            #             # a = a + 1
            #             cv2.waitKey(0)
            #     a = a + 1
            # if w > 2999:
            #     print('> 3000:', file)
            #     r = int(h / w + 0.5)
            #     print(h / w)
            #     print(r)
            #     img1 = cv2.resize(img1, (1536, r*1536))
            #     for i in range(3):
            #         for j in range(3*r):
            #             print(img1.shape)
            #             img = img1[j*512:(j+1)*512, i*512:(i+1)*512]
            #             print(img.shape)
            #             # cv2.imshow('img'+str(i)+str(j), img)
            #             cv2.imwrite('E:/guji_resizedata/shui80_00' + str(a)+'_'+str(i)+str(j) + '.jpg', img)
            #             # a = a + 1
            #             cv2.waitKey(0)
            #     a = a + 1

            # img1 = img[r*512:int(ratio*512), 0:512]
            # cv2.imshow('img1', img1)
            # cv2.waitKey(0)
            # print(img.shape)
        # if h == w:
        #     img = cv2.resize(img1, (512, 512))
        # if h < w:
        #     hr = int(h / (w / 1000.0))
        #     img = cv2.resize(img1, (512, 512))
        # cv2.imwrite('E:/guji_resizedata/dai_00' + str(i) + '.jpg', img)
        # i = i + 1