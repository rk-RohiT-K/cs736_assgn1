import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tqdm import tqdm
import h5py


def extract_patches(image, patch_size=8, variance_threshold=1e-4):
    """
    Extract patches from image with sufficient variance.
    Returns: patches of shape (n_patches, patch_size*patch_size)
    """
    h, w = image.shape
    patches = []
    
    for i in range(0, h-patch_size+1, 4):  
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
    
    L = np.linalg.norm(D.T @ D, 2) 
    eta = 1.0 / L
    
    for _ in range(100):  
        grad = D.T @ (D @ R - X.T)
        R_new = R - eta * grad
        
        if p == 1:
            R = np.sign(R_new) * np.maximum(np.abs(R_new) - eta*lambda_reg, 0)
        else:
            R = R_new * np.maximum(1 - eta*lambda_reg*p*np.abs(R_new)**(p-1), 0)
    
    return R

def learn_dictionary(X, K, p, max_iter=100):
    """Learn dictionary D using alternating minimization."""
    patch_size = int(np.sqrt(X.shape[1]))
    D = initialize_dictionary(K, patch_size)
    obj_values = []
    lambda_reg = 0.1 
    
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
            r = optimize_coefficients(patch.reshape(1, -1), D, 0.8, 0.1)
            reconstructed = (D @ r).reshape(patch_size, patch_size)
            denoised[i:i+patch_size, j:j+patch_size] += reconstructed
            counts[i:i+patch_size, j:j+patch_size] += 1
    
    return denoised / (counts + 1e-10)

def load_images_from_mat(mat_file_path):
    images = []
    try:
        with h5py.File(mat_file_path, 'r') as f:
            for key in f.keys():
                data = f[key][:]
                if isinstance(data, np.ndarray) and (data.ndim == 2 or data.ndim == 3):
                    images.append(data)
                    print(f"Found image: {key} with shape {data.shape}")
        return images
    except Exception as e:
        print(f"Error loading images from .mat file: {e}")
        return []

def main():
    img = load_images_from_mat('../data/assignmentImageDenoising_chestCT.mat')
    img = img[0]
    patches = extract_patches(img)
    print(f"Number of patches extracted: {len(patches)}")
    
    p_values = [2.0, 1.6, 1.2, 0.8]
    dictionaries = []
    all_obj_values = []
    
    for p in p_values:
        print(f"\nLearning dictionary for p={p}")
        D, obj_values = learn_dictionary(patches, K=64, p=p)
        dictionaries.append(D)
        all_obj_values.append(obj_values)
        
        plt.figure()
        plt.plot(obj_values)
        plt.title(f'Objective function vs iterations (p={p})')
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.show()
        
        plt.figure(figsize=(10, 10))
        for k in range(64):
            plt.subplot(8, 8, k+1)
            plt.imshow(D[:, k].reshape(8, 8), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Dictionary atoms (p={p})')
        plt.show()
    
    img_range = np.max(img) - np.min(img)
    noise_std = 0.1 * img_range
    noisy_img = img + np.random.normal(0, noise_std, img.shape)
    
    denoised_img = denoise_image(noisy_img, dictionaries[-1])
    
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