import matplotlib.pyplot as plt
from optparse import OptionParser
import numpy as np
import sys

# Get image path
def getOptions():
	parser = OptionParser()
	parser.add_option("-p", "--picture", dest="picture", action="store", help="specify the input picture path")
	parser.add_option("-c", "--count", dest="count", action="store", help="specify the required singular value count")
	options, _ = parser.parse_args()
	if not options.picture:
		sys.exit("-p/--picture is required!")
	return options


def compress(K,U,Sigma,VT,verbose=False):
	if verbose:
		print("Selected singular value count: {}".format(K))
		print("Before compression: {} {} {}".format(U.shape, Sigma.shape, VT.shape))
	U_K = U[:, :K]
	Sigma_K = np.eye(K)*Sigma[:K]   # Because Sigma matrix is a vector instead of a matrix, need to convert to matrix
	VT_K = VT[:K, :]
	if verbose:
		print("After  compression: {} {} {}".format(U_K.shape, Sigma_K.shape, VT_K.shape))
	return np.matmul(np.matmul(U_K,Sigma_K), VT_K)



if __name__ == "__main__":
	options = getOptions()

	# Read image
	pMatrix= np.array(plt.imread(options.picture))
	print("Original image shape: {}".format(pMatrix.shape))

	# Do SVD decomposition for each channel of RGB
	R, G, B = pMatrix[:,:,0], pMatrix[:,:,1], pMatrix[:,:,2]
	U_R,Sigma_R,VT_R = np.linalg.svd(R)
	U_G,Sigma_G,VT_G = np.linalg.svd(G)
	U_B,Sigma_B,VT_B = np.linalg.svd(B)

	# compress by top K singular values
	K = int(options.count) if options.count else 100
	R_new = compress(K,U_R,Sigma_R,VT_R,verbose=True)
	G_new = compress(K,U_G,Sigma_G,VT_G)
	B_new = compress(K,U_B,Sigma_B,VT_B)
	pMatrix_new = np.stack((R_new,G_new,B_new),2)  # Compose R,G,B channels back to complete image
	print("Compresses image shape: {}".format(pMatrix_new.shape))

	# show image after compress
	plt.imshow(pMatrix_new)
	plt.show()
	sys.exit()
