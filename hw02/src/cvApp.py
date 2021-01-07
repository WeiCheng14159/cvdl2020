import cv2
import numpy as np


class cvApp():
    def __init__(self):

        self.q1_video = "./Q1_Image/bgSub.mp4"
        self.q2_video = "./Q2_Image/opticalFlow.mp4"
        self.q3_video = "./Q3_Image/test4perspective.mp4"
        self.q3_img = "./Q3_Image/rl.jpg"

        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.BLUE = (255, 0, 0)

        '''
        OpenCV HSV color space
        H: 0-179
        S: 0-255
        V: 0-255
        '''
        self.LIGHT_BLUE = (105, 128, 64)
        self.DARK_BLUE = (170, 255, 255)

    def bg_subtraction(self):

        # Image counter
        imgCount = 0

        # Create video object
        cap = cv2.VideoCapture(self.q1_video)

        if not cap.isOpened():
            print("Fail to open video ", self.q1_video)
            return

        # Obtain video frame shape
        ret, frame = cap.read()
        (img_h, img_w, _) = frame.shape

        # Create background subtractor (mean, std)
        bg_mean = None
        bg_std = None

        # First 50 images
        bg_sum = []

        while(cap.isOpened()):

            # Read an image from video
            ret, frame = cap.read()
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (not ret):
                break

            # Convert image to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # OpenCV assume float type image is of range 0.0 ~ 1.0
            gray /= 255.0

            # Increment
            imgCount = imgCount + 1

            if imgCount <= 50:  # Update the background model
                bg_sum.append(gray)
            elif imgCount == 51:  # Calculate mean and std

                # Convert sum of 50 images to numpy ndarray of size (50, 176, 320)
                bg_sum = np.array(bg_sum)

                # Compute mean
                bg_mean = np.mean(bg_sum, axis=0)

                # Compute std
                bg_std = np.std(bg_sum, axis=0)

                # Filter std
                bg_std[bg_std < 1e-2] = 1e-2
            else:

                # Compute diff between this frame and mean frame
                diff = gray - bg_mean

                # Foreground / background threshold
                thres = 10 * bg_std

                for i in range(img_h):
                    for j in range(img_w):
                        gray[i, j] = 1.0 if (diff[i, j] > thres[i, j]) else 0.0

                # cv2.imshow("diff", diff)
                cv2.imshow('gray', gray)
                # cv2.imshow('frame', frame)

        cap.release()
        cv2.destroyAllWindows()

    def __create_blob(self):

        # Blob parameter
        params = cv2.SimpleBlobDetector_Params()

        params.minDistBetweenBlobs = 15

        params.maxThreshold = 30
        params.minThreshold = 5

        params.filterByArea = True
        params.minArea = 1
        params.maxArea = 25

        params.filterByCircularity = True
        params.minCircularity = 0.3
        params.maxCircularity = 1

        params.filterByColor = False

        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.maxConvexity = 1.0

        params.filterByInertia = True
        params.minInertiaRatio = 0.25
        params.maxInertiaRatio = 1.0  # Circle

        return cv2.SimpleBlobDetector_create(params)

    def preprocessing(self):

        # Create a blob detector
        detector = self.__create_blob()

        # Create video object
        cap = cv2.VideoCapture(self.q2_video)

        while(cap.isOpened()):

            # Read an image from video
            ret, frame = cap.read()
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (not ret):
                break

            # Color space conversion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Color masking
            mask = cv2.inRange(hsvImg, self.LIGHT_BLUE, self.DARK_BLUE)
            blueMask = cv2.bitwise_and(gray, gray, mask=mask)

            # Find keypoints using masked image
            keypts = detector.detect(blueMask)
            imgWithKeyPts = frame.copy()

            for k in keypts:

                # Convert keypoints indices to int
                posInt = (int(k.pt[0]), int(k.pt[1]))

                # Draw cross marker
                cv2.drawMarker(imgWithKeyPts, posInt, self.RED, markerType=cv2.MARKER_CROSS,
                               markerSize=11, thickness=1)

                # Draw square marker
                cv2.drawMarker(imgWithKeyPts, posInt, self.RED, markerType=cv2.MARKER_SQUARE,
                               markerSize=11, thickness=1)

            cv2.imshow('kypts', imgWithKeyPts)
            # cv2.imshow('blueMask', blueMask)

        cap.release()
        cv2.destroyAllWindows()

    def video_tracking(self):

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create video object
        cap = cv2.VideoCapture(self.q2_video)

        # Take first frame and find corners in it
        ret, old_frame = cap.read()

        # Color space conversion
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        hsvImg = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

        # Color masking
        hsvMask = cv2.inRange(hsvImg, self.LIGHT_BLUE, self.DARK_BLUE)
        blueMaskOld = cv2.bitwise_and(old_gray, old_gray, mask=hsvMask)

        # Create a blob detector
        detector = self.__create_blob()

        # Find keypoints using masked image
        keyPtsOld = detector.detect(blueMaskOld)

        # Convert OpenCV keypoints to a list of 2D points in float
        p0 = cv2.KeyPoint_convert(keyPtsOld)

        # Create a empty canvas for line drawing
        lineCanvas = np.zeros_like(old_frame)

        while(cap.isOpened()):

            # Read an image from video
            ret, frame = cap.read()
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (not ret):
                break

            # Color space conversion
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Color masking
            hsvMask = cv2.inRange(hsvImg, self.LIGHT_BLUE, self.DARK_BLUE)
            blueMask = cv2.bitwise_and(frame_gray, frame_gray, mask=hsvMask)

            # Calculate optical flow based on gray images
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params)

            # Given a successful optical flow tracking
            if len(status) != 0:

                # Select indices where a flow is found (status == 1)
                flowFoundIndices = np.where(status == 1)[0]

                # Select indices where the error is small
                goodFlowIndices = np.where(err[flowFoundIndices] < 5.0)[0]

                if len(goodFlowIndices) == 0:  # No good flow found
                    continue

                # Good flows
                good_new = p1[goodFlowIndices]
                good_old = p0[goodFlowIndices]

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.astype(int).ravel()
                    c, d = old.astype(int).ravel()
                    lineCanvas = cv2.line(
                        lineCanvas, (a, b), (c, d), self.GREEN, 2)
                    frame = cv2.circle(frame, (a, b), 3, self.RED, -1)

            # Merge drawing and background images
            tracking = cv2.add(frame, lineCanvas)

            cv2.imshow('tracking', tracking)
            # cv2.imshow('blueMask', blueMask)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cap.release()
        cv2.destroyAllWindows()

    def prespective_trans(self):

        # Read input image
        srcImg = cv2.imread(self.q3_img)

        # Create video object
        cap = cv2.VideoCapture(self.q3_video)

        if not cap.isOpened():
            print("Fail to open video ", self.q3_video)
            return

        # Load the dictionary that was used to generate the markers
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

        # Initialize the detector parameters using default values
        parameters = cv2.aruco.DetectorParameters_create()

        while cap.isOpened():

            # Read an image from video
            ret, frame = cap.read()
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (not ret):
                break

            # Detect markers in the image
            markerCorners, markerIds, rejectCandicates = cv2.aruco.detectMarkers(
                frame, aruco_dict, parameters=parameters)

            # If fewer than 4 markers are found then skip to next frame
            if len(markerIds) != 4:
                continue

            # marker #25 is on the upper left corner
            idx = np.squeeze(np.where(markerIds == 25))
            refPnt1 = np.squeeze(markerCorners[idx[0]])[1]

            # marker #33 is on the upper right corner
            idx = np.squeeze(np.where(markerIds == 33))
            refPnt2 = np.squeeze(markerCorners[idx[0]])[2]

            # marker #30 is on the bottom right corner
            idx = np.squeeze(np.where(markerIds == 30))
            refPnt3 = np.squeeze(markerCorners[idx[0]])[0]

            # marker #23 is on the bottom left corner
            idx = np.squeeze(np.where(markerIds == 23))
            refPnt4 = np.squeeze(markerCorners[idx[0]])[0]

            # Draw markers on four corner of the target
            frame = cv2.circle(
                frame, (refPnt1[0], refPnt1[1]), 3, self.RED, -1)
            frame = cv2.circle(
                frame, (refPnt2[0], refPnt2[1]), 3, self.RED, -1)
            frame = cv2.circle(
                frame, (refPnt3[0], refPnt3[1]), 3, self.RED, -1)
            frame = cv2.circle(
                frame, (refPnt4[0], refPnt4[1]), 3, self.RED, -1)

            # pts_dst is the four corners of dest image
            pts_dst = np.array([
                [refPnt1[0], refPnt1[1]],
                [refPnt2[0], refPnt2[1]],
                [refPnt3[0], refPnt3[1]],
                [refPnt4[0], refPnt4[1]]
            ])

            # pts_src is the four corners of src image
            pts_src = np.array([
                [0, 0],
                [srcImg.shape[1], 0],
                [srcImg.shape[1], srcImg.shape[0]],
                [0, srcImg.shape[0]]
            ])

            # Find the homography matrix H
            H, status = cv2.findHomography(pts_src, pts_dst)

            # Create a blank image for masking purpose
            blank = np.zeros((srcImg.shape[0], srcImg.shape[1]), np.uint8)

            # Perspective the masking image
            maskImg = cv2.warpPerspective(
                blank, H, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

            # Prespective the source image
            dstImg = cv2.warpPerspective(
                srcImg, H, (frame.shape[1], frame.shape[0]))

            # Crop out a region
            frame = cv2.bitwise_and(frame, frame, mask=maskImg)

            # Merge cropped region and background
            frame = cv2.add(frame, dstImg)

            cv2.imshow('frame', frame)
            # cv2.imshow('dstImg', dstImg)
            # cv2.imshow('maskImg', maskImg)

            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    g = cvApp()
    g.prespective_trans()
