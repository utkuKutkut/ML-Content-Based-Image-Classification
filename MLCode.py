from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2



#This function creates an LEVEL base histograms
def extract_level_color_histogram(image,bins=(2,2,2)):

	img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	#I take the size of images. Accordingly, I mask the certain parts and calculate histogram values only for that parts.
	mask = np.zeros(img.shape[:2], np.uint8)
	mask[0:img.shape[0]/4, 0:img.shape[1]/4] = 255
	masked_img = cv2.bitwise_and(img,img,mask = mask)
	hist_mask = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])
	mask = np.zeros(img.shape[:2], np.uint8)
	mask[0:img.shape[0]/4, img.shape[1]/4:img.shape[1]/2] = 255
	masked_img2 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask2 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[0:img.shape[0]/4, img.shape[1]/2:(img.shape[1])*3/4] = 255
	masked_img3 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask3 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[0:img.shape[0]/4, (img.shape[1])*3/4:img.shape[1]] = 255
	masked_img4 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask4 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[img.shape[0]/4:img.shape[0]/2, 0:img.shape[1]/4] = 255
	masked_img5 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask5 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[img.shape[0]/4:img.shape[0]/2, img.shape[1]/4:img.shape[1]/2] = 255
	masked_img6 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask6 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[img.shape[0]/4:img.shape[0]/2, img.shape[1]/2:(img.shape[1])*3/4] = 255
	masked_img7 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask7 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[img.shape[0]/4:img.shape[0]/2, (img.shape[1])*3/4:img.shape[1]] = 255
	masked_img8 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask8 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[img.shape[0]/2:(img.shape[0])*3/4, 0:img.shape[1]/4] = 255
	masked_img9 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask9 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])


	mask = np.zeros(img.shape[:2], np.uint8)
	mask[img.shape[0]/2:(img.shape[0])*3/4, img.shape[1]/4:img.shape[1]/2] = 255
	masked_img10 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask10 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[img.shape[0]/2:(img.shape[0])*3/4, img.shape[1]/2:(img.shape[1])*3/4] = 255
	masked_img11 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask11 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[img.shape[0]/2:(img.shape[0])*3/4, (img.shape[1])*3/4:img.shape[1]] = 255
	masked_img12 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask12 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[(img.shape[0])*3/4:img.shape[0], 0:img.shape[1]/4] = 255
	masked_img13 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask13 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[(img.shape[0])*3/4:img.shape[0], img.shape[1]/4:img.shape[1]/2] = 255
	masked_img14 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask14 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])


	mask = np.zeros(img.shape[:2], np.uint8)
	mask[(img.shape[0])*3/4:img.shape[0], img.shape[1]/2:(img.shape[1])*3/4] = 255
	masked_img15 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask15 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	mask = np.zeros(img.shape[:2], np.uint8)
	mask[(img.shape[0])*3/4:img.shape[0], (img.shape[1])*3/4:img.shape[1]] = 255
	masked_img16 = cv2.bitwise_and(img,img,mask = mask)
	hist_mask16 = cv2.calcHist([img], [0, 1, 2], mask, bins,
		[0, 256, 0, 256, 0, 256])

	
	
	combineHist =hist_mask+hist_mask2+hist_mask3+hist_mask4+hist_mask5+hist_mask6+hist_mask7+hist_mask8+hist_mask9+hist_mask10+hist_mask11+hist_mask12+hist_mask13+hist_mask14+hist_mask15+hist_mask16

	
	# feature vector
	return combineHist.flatten()




#first, read the training set.
print("[INFO] describing images...")
imagePaths = list(paths.list_images("DataSet"))

#They will hold my features vectors and class label names.
features = []
labels = []



#read all training set
for (i, imagePath) in enumerate(imagePaths):

	#this reads all image in data set one by one
	image = cv2.imread(imagePath)
	
	#I labeled each image path, according to file they belong to. (airplane, leopar ...)
	imageName=imagePath.split('/')
	label = imageName[-2]

	
	#Lets call our function to feature extraction
	#YOU CAN CHANGE FUNCTION NAME HERE. I TESTED EACH FUNCTION BY CHANGING HERE
	hist = extract_level_color_histogram(image)

	
	#and then, simply put these vectors and labels to our array.
	features.append(hist)
	labels.append(label)
	
	# show notification every 50 images

	if i > 0 and i % 10 == 0:
		print("Training data processed {}/{}".format(i, len(imagePaths)))





#Notice that test_size means, the ratio of test data to the given data (we assigned it to trainLabels)
#Program will still select 20 images as training RANDOMLY.


(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=20, shuffle=True)


#just little check up
nTrainLabels=len(trainLabels)
nTestLabels=len(testLabels)
print("\n")
print ("Total number of training data:%d"%(nTrainLabels))
print ("Total number of testign data:%d"%(nTestLabels))
print("\n")



# train and evaluate a k-NN classifer on the histogram as a final step.
print("Evaluating histogram accuracy...")

#This is where you can change our nearest neighbor value (K)
knn = KNeighborsClassifier(n_neighbors=2)


#train our program with train dataset
knn.fit(trainFeat, trainLabels)

#Test our program with test set.
acc = knn.score(testFeat, testLabels)




print("Accuracy calculated: {:.2f}%".format(acc * 100))















