# deep_crs_him8
requirements:
'cv2'(opencv-python), 'numpy', 'xarray', 'scikit-learn', 'matplotlib', 'keras','tensorflow', 'scipy'

Dataset:
    Datasets explained in file dataset.txt
    URL: https://cloudstor.aarnet.edu.au/plus/s/Dau9GY3Y3IqobXQ
    Size: 43.5G

code: 
For usage please run any code as python example.py -h

PCA: PCA/pca.py
    This will calculate the PCA operator, reduce the dataset and calculate the distances 
    between bands. The result is a csv file.

Denoising: DN_OF_FrameNet/denoising.py
    This acts as a module for other packages but can also been executed generating a denoised version of the
    input dataset

Optical Flow: DN_OF_FrameNet/opticalflow.py
    This acts as a module for other packages but can also been executed performing opticalflow and showing the optical flow in the screen.

Frame prediction network: DN_OF_FrameNet/frame_pred.py
    This trains or test the frame prediction network.
    Files losses and utils are helpers of this network

Clustering: PrecNet/Clustering/clustering.py
    This acts as a module for other packages but can also been executing performing the clustering either using kmeans of meanshift of a dataset

Rain prediction network: PrecNet/rainpred.py
    This trains or test the rain prediction network.

Sharpening:
        This acts as a module for other packages but can also been executing performing sharpening of a dataset


Extra functions:

Others/image_creations.py extracts and save the images of a layer in a dataset into a folder
Others/video_creation.py creates a video from a layer in a dataset
