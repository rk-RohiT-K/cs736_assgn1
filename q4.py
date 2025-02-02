import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from tqdm import tqdm
import h5py
def load_ct_image(path):
    """Load the CT image from mat file."""
    data = loadmat(path)
    # Assuming the image is stored in the first key of the mat file
    return data[list(data.keys())[3]]  # Skip '__header__', '__version__', '__globals__'

def extract_patches(image, patch_size=8, variance_threshold=1e-4):
    """
    Extract patches from image with sufficient variance.
    Returns: patches of shape (n_patches, patch_size*patch_size)
    """
    h, w = image.shape
    patches = []
    
    for i in range(0, h-patch_size+1, 4):  # Step size of 4 for efficiency
        for j in range(0, w-patch_size+1, 4):
            patch = image[i:i+patch_size, j:j+patch_size]
            if np.var(patch) > variance_threshold:
                patches.append(patch.flatten())
    
    return np.array(patches)

def initialize_dictionary(K, patch_size=8):
    """Initialize dictionary with random normalized atoms."""
    D = np.random.randn(patch_size*patch_size, K)
    return normalize(D, axis=0, norm='l2')

def optimize_coefficients(X, D, p, lambda_reg):
    """Optimize coefficients R given D using ISTA."""
    n_samples = X.shape[0]
    K = D.shape[1]
    R = np.zeros((K, n_samples))
    
    # ISTA parameters
    L = np.linalg.norm(D.T @ D, 2)  # Lipschitz constant
    eta = 1.0 / L
    
    for _ in range(100):  # Max iterations for coefficient update
        grad = D.T @ (D @ R - X.T)
        R_new = R - eta * grad
        
        # Soft thresholding for p=1, or other proximal operators for different p
        if p == 1:
            R = np.sign(R_new) * np.maximum(np.abs(R_new) - eta*lambda_reg, 0)
        else:
            # Approximate solution for other p values
            R = R_new * np.maximum(1 - eta*lambda_reg*p*np.abs(R_new)**(p-1), 0)
    
    return R

def learn_dictionary(X, K, p, max_iter=100):
    """Learn dictionary D using alternating minimization."""
    patch_size = int(np.sqrt(X.shape[1]))
    D = initialize_dictionary(K, patch_size)
    obj_values = []
    lambda_reg = 0.1  # Regularization parameter
    
    for iter in tqdm(range(max_iter)):
        # Fix D, optimize R
        R = optimize_coefficients(X, D, p, lambda_reg)
        
        # Fix R, optimize D
        D = normalize(X.T @ R.T, axis=0, norm='l2')
        
        # Calculate objective
        reconstruction = np.linalg.norm(X - (D @ R).T, 'fro')**2
        regularization = lambda_reg * np.sum(np.abs(R)**p)
        obj_values.append(reconstruction + regularization)
    
    return D, obj_values

def denoise_image(noisy_img, D, patch_size=8):
    """Denoise image using learned dictionary."""
    h, w = noisy_img.shape
    denoised = np.zeros_like(noisy_img)
    counts = np.zeros_like(noisy_img)
    
    for i in range(0, h-patch_size+1):
        for j in range(0, w-patch_size+1):
            patch = noisy_img[i:i+patch_size, j:j+patch_size].flatten()
            # Solve sparse coding problem
            r = optimize_coefficients(patch.reshape(1, -1), D, 0.8, 0.1)
            # Reconstruct patch
            reconstructed = (D @ r).reshape(patch_size, patch_size)
            denoised[i:i+patch_size, j:j+patch_size] += reconstructed
            counts[i:i+patch_size, j:j+patch_size] += 1
    
    return denoised / (counts + 1e-10)

def load_images_from_mat(mat_file_path):
    images = []
    try:
        # Open the .mat file
        with h5py.File(mat_file_path, 'r') as f:
            # Loop through all variables in the .mat file
            for key in f.keys():
                data = f[key][:]
                
                # Heuristic check: image data usually has 2 or 3 dimensions (height, width, [channels])
                if isinstance(data, np.ndarray) and (data.ndim == 2 or data.ndim == 3):
                    # Add the image data to the list
                    images.append(data)
                    print(f"Found image: {key} with shape {data.shape}")
        
        # Return all detected images
        return images
    except Exception as e:
        print(f"Error loading images from .mat file: {e}")
        return []

def main():
    # Load image
    img = load_images_from_mat('../data/assignmentImageDenoising_chestCT.mat')
    img = img[0]
    
    
    # Extract patches
    patches = extract_patches(img)
    print(f"Number of patches extracted: {len(patches)}")
    
    # Learn dictionaries for different p values
    p_values = [2.0, 1.6, 1.2, 0.8]
    dictionaries = []
    all_obj_values = []
    
    for p in p_values:
        print(f"\nLearning dictionary for p={p}")
        D, obj_values = learn_dictionary(patches, K=64, p=p)
        dictionaries.append(D)
        all_obj_values.append(obj_values)
        
        # Plot objective function
        plt.figure()
        plt.plot(obj_values)
        plt.title(f'Objective function vs iterations (p={p})')
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.show()
        
        # Plot dictionary atoms
        plt.figure(figsize=(10, 10))
        for k in range(64):
            plt.subplot(8, 8, k+1)
            plt.imshow(D[:, k].reshape(8, 8), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Dictionary atoms (p={p})')
        plt.show()
    
    # Denoising experiment
    # Add noise
    img_range = np.max(img) - np.min(img)
    noise_std = 0.1 * img_range
    noisy_img = img + np.random.normal(0, noise_std, img.shape)
    
    # Denoise using dictionary learned with p=0.8
    denoised_img = denoise_image(noisy_img, dictionaries[-1])
    
    # Show results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy')
    plt.subplot(133)
    plt.imshow(denoised_img, cmap='gray')
    plt.title('Denoised')
    plt.show()

if __name__ == "__main__":
    main() 