import cv2
cap = cv2.VideoCapture(2)

while True:
	ret, img = cap.read()
	
	# print(ret, img)=
	# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print(img.shape)
	height, width, _ = img.shape 
	'''for y in range(height):
		for x in range(width):
			img[y][x] = [100, 100, img[y][x][2]] 
	'''
	img[:, :,[1]] = 100
	# print(srez)
	cv2.imshow("blue", img)
	# cv2.imshow("srez", srez)
	# cv2.imshow("gray", img_gray)
	key = cv2.waitKey(1)
	# print(key)
	if key == 32:
		print("okay")
		break
cap.release()
cv2.destroyAllWindows()
 
	

	
	



