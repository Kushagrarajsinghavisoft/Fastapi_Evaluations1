import pickle

pca_path='datafiles/PCA_ds.pkl'
scaler_path = 'datafiles/scaled_file.pkl'
K_elbow_path = 'datafiles/Kelbow.pkl'
trained_model_path = 'datafiles/Agglomerative_Clustering_AC_model_file.pkl'

# Load the pca from the pickle file
with open(pca_path, 'rb') as f:
    pca_ds = pickle.load(f)

# Load the scaler from the pickle file
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(K_elbow_path, 'rb') as f:
    K_elbow = pickle.load(f)

# Load the trained model from the pickle file
with open(trained_model_path, 'rb') as f:
    model = pickle.load(f)

