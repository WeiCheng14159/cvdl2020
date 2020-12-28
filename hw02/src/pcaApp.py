import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


class pcaApp():

    def __init__(self):

        # List of badge image files
        self.badgeFiles = sorted(glob.glob("./Q4_Image/" + "*.jpg"))

        # Badge shape
        self.imgShape = (100, 100, 3)

        # Number of PCA components
        self.numOfComp = 10

    def __pca_compute(self):

        # Image array
        self.X = []

        # Read in 34 images, each of size (100, 100, 3)
        for fname in self.badgeFiles:

            # Read image as float32
            img = cv2.imread(fname).astype(np.float32)

            # Reshape (100, 100, 3) to (30000, )
            vec = np.reshape(img, -1)

            # Append to list
            self.X.append(vec)

        # X is of size (34, 30000)
        self.X = np.array(self.X)

        # OpenCV assume float32 type image is of range 0.0 ~ 1.0
        self.X /= 255.0

        # Compute mean
        mean = np.mean(self.X, axis=0)

        # Reshape mean from (30000,) to (1, 30000)
        mean = np.expand_dims(mean, axis=0)

        # Mean image
        meanImg = np.reshape(mean, self.imgShape)

        # PCA
        # cv2.PCACompute2 will compute mean automatically
        mean, eigVec, eigVals = cv2.PCACompute2(
            data=self.X, mean=None, eigenvectors=None, eigenvalues=None, maxComponents=self.numOfComp)

        # Reconstructed image and original image
        reconImgList = []
        origImgList = []

        # Show the reconstructed images
        for vec in self.X:

            # vec reshape from (30000,) -> (1, 30000)
            vec = np.expand_dims(vec, axis=0)

            # Project the vector to eigenspace
            projEigVal = cv2.PCAProject(
                data=vec, mean=mean, eigenvectors=eigVec)

            # Back project the eigenvalues from eigenspace to original space
            reconVec = cv2.PCABackProject(
                data=projEigVal, mean=mean, eigenvectors=eigVec)

            # Reshape the (1,30000) vector to (100, 100, 3) OpenCV readable image
            reconImg = np.reshape(reconVec, self.imgShape)
            reconImgList.append(reconImg)

            # Original image
            origImg = np.reshape(vec, self.imgShape)
            origImgList.append(origImg)

        # Convert to numpy ndarray
        reconImgList = np.array(reconImgList)
        origImgList = np.array(origImgList)

        return reconImgList, origImgList

    def img_reconstruct(self):

        # PCA
        recon, orig = self.__pca_compute()

        # Show original and reconstructed images
        for i in range(34):
            cv2.imshow("orig", orig[i, :, :, :])
            cv2.imshow("recon", recon[i, :, :, :])
            cv2.waitKey(500)

        cv2.destroyAllWindows()

    def recon_error(self):

        # Recontruction error (RE)
        re = np.zeros(34)

        # PCA
        recon, orig = self.__pca_compute()

        # Show original and reconstructed images
        for i in range(34):

            # Convert origial and reconstructed image to gray scale
            origColor = orig[i, :, :, :]
            origGray = cv2.cvtColor(origColor, cv2.COLOR_BGR2GRAY)

            reconColor = recon[i, :, :, :]
            reconGray = cv2.cvtColor(reconColor, cv2.COLOR_BGR2GRAY)

            # Compute RE
            err = np.absolute(origGray - reconGray)
            re[i] = np.sum(err)

        print(re)


if __name__ == "__main__":
    g = pcaApp()
    g.recon_error()
