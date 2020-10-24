import cv2
import glob
import numpy as np


class CameraCalib():
    def __init__(self, path="./Q1_Image/"):
        # CameraCalib parameters
        # 3x3 intrinsics matrix
        self.mtx = None
        # Distortion matrix
        self.dist = None
        # 3x1 rotational matrix
        self.rvecs = None
        # 3x1 transition matrix
        self.tvecs = None

        # Path to caliberate images
        self.calibImgDir = path
        self.calibImgFileList = sorted(glob.glob(self.calibImgDir + "*.bmp"))
        # Chess board dimension
        self.boardDim = [11, 8]

    # Get the chess board dimension
    def get_chess_board_dim(self):
        return self.boardDim

    # Get caliberate image directory
    def get_img_dir(self):
        return self.calibImgDir

    # Get a filename list of caliberation images
    def get_img_path_list(self):
        return self.calibImgFileList

    # Get intrinsic parameters
    def get_mtx(self):
        return self.mtx

    # Get distortion matrix
    def get_dist(self):
        return self.distortion

    # Get rvecs
    def get_rvecs(self, idx):
        return self.rvecs[idx]

    # Get tvecs
    def get_tvecs(self, idx):
        return self.tvecs[idx]

    # Get extrinsic parameters
    def get_ext(self, idx):
        if(idx >= 0 and idx < len(self.rvecs)):
            # R is a 3x3 rotation matrix
            R = np.zeros((3, 3), dtype='float32')
            # T is a 3x1 transition vector
            T = np.zeros((3, 1), dtype='float32')
            # R_T is a 3x4 extrinsic matrix
            R_T = np.zeros((3, 4), dtype='float32')
            # rvecs is a 3x1 vector because the rotation matrix has only 3 DoF,
            # use cv2.Rodrigues function to convert to 3x3 matrix
            R, _ = cv2.Rodrigues(self.rvecs[idx])
            # tvecs is a 3x1 transition vector
            T = self.tvecs[idx]
            # Concatenate matrix and vector
            R_T = np.concatenate((R, T), axis=1)
            return R_T
        else:
            return None

    def calibrate(self):
        # Result images with anchor points marked on chess board image
        markedImgs = []
        # Subpixel refinement stop criteria
        subPixelStopCriteria = (cv2.TERM_CRITERIA_EPS +
                                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(self.boardDim[0]-1,self.boardDim[1]-1,0)
        chessBoardCoordinates = np.zeros(
            (self.boardDim[0]*self.boardDim[1], 3), np.float32)
        chessBoardCoordinates[:, :2] = np.mgrid[0:self.boardDim[0],
                                                0:self.boardDim[1]].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        world3dPnts = []  # 3d point in real world space
        img2dPnts = []  # 2d points in image plane.

        print("Find chess board images in " + self.get_img_dir())
        for fname in self.get_img_path_list():
            # Read in images
            img = cv2.imread(fname)
            # Convert to gray scale image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            found, corners = cv2.findChessboardCorners(
                gray, (self.boardDim[0], self.boardDim[1]), None)
            # If found, add object points, image points (after refining them)
            if found == True:
                # Add object points
                world3dPnts.append(chessBoardCoordinates)
                # Refine to at subPixel level
                cornersRefined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), subPixelStopCriteria)
                # Add image points
                img2dPnts.append(cornersRefined)
                # Draw and display the corners
                cv2.drawChessboardCorners(
                    img, (self.boardDim[0], self.boardDim[1]), cornersRefined, found)
                # Save the marked chess board image
                markedImgs.append(img)
            else:
                return None
        # Calibrate the camera
        ret, self.mtx, self.distortion, self.rvecs, self.tvecs = cv2.calibrateCamera(
            world3dPnts, img2dPnts, gray.shape[::-1], None, None)
        return markedImgs
