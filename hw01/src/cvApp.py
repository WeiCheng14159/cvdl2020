import cv2
import numpy as np
from cameraCalib import CameraCalib


class cvApp():
    def __init__(self):
        # Camera caliberation object
        self.cameraObj = CameraCalib()
        # Default chess board image file index and file name
        self.fileIdx = 0
        self.filePath = self.cameraObj.get_img_dir() + "1.bmp"
        # OpenCV color code
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)

    # Q4
    # This function match SIFT features in two images
    def match_keypoints(self):
        # SIFT
        img1, img2, kp1, kp2, kp1_list, kp2_list, goodMatches = self.sift_match()
        # Draw top 7 matched key points on image combined
        combined = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodMatches[:7], None,
                                      self.RED, self.RED, None, cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Show image
        cv2.imshow('img', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # This function finds key points by SIFT algorithm
    def find_keypoints(self):
        # Where to store the output file path
        dstList = ["./Q4_Image/FeatureAerial1.jpg",
                   "./Q4_Image/FeatureAerial2.jpg"]
        # SIFT
        img1, img2, kp1, kp2, kp1_list, kp2_list, goodMatches = self.sift_match()
        # Draw SIFT key points in two images
        out1 = cv2.drawKeypoints(img1, kp1_list, None, self.RED)
        out2 = cv2.drawKeypoints(img2, kp2_list, None, self.RED)

        cv2.imwrite(dstList[0], out1)
        cv2.imwrite(dstList[1], out2)

        cv2.imshow('img', out1)
        cv2.waitKey(0)
        cv2.imshow('img', out2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # This function match two image by SIFT algorithm
    def sift_match(self):
        # Path to image to match
        srcList = ["./Q4_Image/Aerial1.jpg", "./Q4_Image/Aerial2.jpg"]
        # Read image in black white mode
        img1 = cv2.imread(srcList[0], cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(srcList[1], cv2.COLOR_BGR2GRAY)
        # Create SIFT descriptor
        sift = cv2.SIFT_create()
        # Compute keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # Match descriptor by FLANN matcher
        # FLANN parameters
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        # Apply the ratio test
        goodMatches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatches.append((m, n))
        # Sort good matching points
        goodMatches = sorted(goodMatches, key=lambda x: (
            x[0].distance + x[1].distance))
        # Store top 7 best match key points
        kp1_list = []
        kp2_list = []
        for m, n in goodMatches[:7]:
            kp1_list.append(kp1[m.queryIdx])
            kp2_list.append(kp2[m.trainIdx])

        return img1, img2, kp1, kp2, kp1_list, kp2_list, goodMatches

    # Q3
    # This function finds the disparity map between two images
    def find_disparity_map(self):
        # Read in images in black white
        imgL = cv2.imread("./Q3_Image/imL.png", cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread('./Q3_Image/imR.png', cv2.IMREAD_GRAYSCALE)
        # Stereo computation settings
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=21)
        # Compute disparity
        disparity = stereo.compute(imgL, imgR)
        # Normalize the disparity to 0-255
        disparity = cv2.normalize(
            disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imshow('img', disparity)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Q2
    # This function draw a tetrahedron on chess board
    def ar_draw_shape(self):
        # Caliberate the camera using a chess board
        self.cameraObj = CameraCalib(path="./Q2_Image/")
        self.cameraObj.calibrate()

        # vertices to draw on chess board
        vertices = np.float32(
            [[1, 1, 0], [5, 1, 0], [3, 3, -3], [3, 5, 0]]).reshape(-1, 3)

        print("Draw on chess board images in " + self.cameraObj.get_img_dir())
        for idx, fname in enumerate(self.cameraObj.get_img_path_list()):
            # Read in a chess board image
            img = cv2.imread(fname)
            # Convert
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            found, _ = cv2.findChessboardCorners(
                gray, (self.cameraObj.boardDim[0], self.cameraObj.boardDim[1]), None)
            # If found, draw on the chess board
            if found == True:
                # Project 3D world coordinate points onto 2D image plane
                imgPts, jacob = cv2.projectPoints(vertices, self.cameraObj.get_rvecs(
                    idx), self.cameraObj.get_tvecs(idx), self.cameraObj.get_mtx(), self.cameraObj.get_dist())
                # Draw a line between all two points in 2D image plane
                for i in range(0, vertices.shape[0]):
                    for j in range(i, vertices.shape[0]):
                        img = cv2.line(img, tuple(imgPts[i].ravel()), tuple(
                            imgPts[j].ravel()), self.GREEN, 5)

                cv2.imshow('img', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

    # Q1
    # This function finds a chess board in an image and caliberate a camera
    def find_corner(self):
        # Caliberate camera
        img2Show = self.cameraObj.calibrate()
        # Show marked chess board images
        for img in img2Show:
            cv2.imshow('img', img)
            cv2.waitKey(100)
        cv2.destroyAllWindows()

    # This function prints out the intrinsics parameter of a caliberated camera
    def find_intrinsic(self):
        print("Intrinsic Parameter of this camera:\n" +
              np.array2string(self.cameraObj.get_mtx()) + "\n")

    # This function prints out the distortion matrix of a caliberated camera
    def find_distortion(self):
        print("Distortion matrix of this camera:\n" +
              np.array2string(self.cameraObj.get_dist()) + "\n")

    # This function prints out the extrinsics parameter of a seleted image
    def find_extrinsic(self):
        # This is the target file index
        targetIdx = 0
        # Find the target file path
        for idx, fname in enumerate(self.cameraObj.get_img_path_list()):
            if fname == self.filePath:
                targetIdx = idx
                break

        print("Extrinsic Parameter of image " +
              self.filePath + " is\n" + np.array2string(self.cameraObj.get_ext(targetIdx)) + "\n")

    # Call back function to set the current file index
    def set_file_idx(self, idx):
        self.fileIdx = idx
        self.filePath = self.cameraObj.get_img_dir() + str(idx+1) + ".bmp"
