import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Find the digits from a video file.')
parser.add_argument('videoFile', help='video file for processing. Must be 720x480')
parser.add_argument('-verbose', help='print debug text', action="store_true")
parser.add_argument('-images', help='show debug images', action="store_true")
args = parser.parse_args()
if args.verbose:
    print(args)

### generate template numbers

templates = []
for i in range(10):
    temp = cv2.imread(str(i) + "_32.bti.png",cv2.IMREAD_UNCHANGED)
    
    height,width = temp.shape[:2]
    temp = cv2.resize(temp, (int(width*38*1.0/height),38))
    alpha = temp[:,:,[3,3,3]].astype(float)/255
    temp = (temp[:,:,:3] * alpha + 128 * (1 - alpha)).astype('uint8')
    temp = temp[:,:,0]
    
    templates.append(temp)
    
    if args.images:
        cv2.imshow("Digit",temp)
        cv2.waitKey(20)
    if args.verbose:
        print(temp.shape)
        
blank_template = templates[0].copy()
blank_template[:,:] = 128
templates.append(blank_template)
        
       
### find and read the digits of a single frame

most_recent_digits = [-1,-1,-1,-1,-1]
consecutive_digits = [0,0,0,0,0]
all_digits = []

def read_digits_on_frame(image):
    
    for i in range(5):
        img = image[162:200, 267+40*i+1:267+40*(i+1)-1]
        img = img.copy()
        blur = cv2.GaussianBlur(img,(139,139),0)
        diff = cv2.subtract(128 + cv2.subtract(img,blur),cv2.subtract(blur,img))
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        score = []
        for j in range(11):
            template = templates[j]
            res = cv2.matchTemplate(diff,template,cv2.TM_SQDIFF_NORMED)
            score.append(( j,res.min(axis=0).min(axis=0) ))

        score = sorted(score, key=lambda x: x[1])
        if args.verbose:
            print(",".join(["%d %.3f" % x for x in score]))
        if args.images:
            cv2.imshow('Digit', diff)
            cv2.waitKey(35)
        pred = score[0][0]
            
        if most_recent_digits[i] == pred:
            consecutive_digits[i] += 1
        else:
            most_recent_digits[i] = pred
            consecutive_digits[i] = 1
        all_digits.append(pred)


### check for frames of the challenge mode result screen

# https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python
def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    a2D = np.floor_divide(a2D,64)
    col_range = (4, 4, 4) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)
    
def is_chresult_screen(frame):
    b,g,r = bincount_app(frame)
    if args.verbose:
        print("frame: %d %d %d" % (b,g,r))
    return b==3 and g==2 and r==2

       
### watch the video, find frames of the challenge mode result screen, and read the digits  

cap = cv2.VideoCapture(args.videoFile)

if (cap.isOpened() == False): 
    print("Failure - error opening video stream or file")
    sys.exit(0)
    
skip = 6000
count = 0
while(cap.isOpened()):
    count += 1
    ret, frame = cap.read()
    if not ret:
        print("Failure - no end found")
        break
    if skip > 0:
        skip -= 1
        continue
    if not is_chresult_screen(frame):
        skip = 100
        continue
    
    if args.images:
        cv2.imshow('Frame',frame)
        cv2.waitKey(50)
    read_digits_on_frame(frame)

    if consecutive_digits[4] >= 6 and consecutive_digits[3] >= 4 and consecutive_digits[2] >= 2 and most_recent_digits[0] == 10:
        all_digits = all_digits[-100:]
        print("Success (4)!")
        for i in range(len(all_digits)):
            if i % 5 != 0:
                print(all_digits[i],end="\n" if i % 5 == 4 else "")
        break

    if consecutive_digits[4] >= 8 and consecutive_digits[3] >= 6 and consecutive_digits[2] >= 4 and consecutive_digits[1] >= 2:
        all_digits = all_digits[-100:]
        print("Success!")
        for i in range(len(all_digits)):
            print(all_digits[i],end="\n" if i % 5 == 4 else "")
        break

cap.release()
cv2.destroyAllWindows()
