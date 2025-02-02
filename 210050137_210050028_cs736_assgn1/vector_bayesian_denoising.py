import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

class VectorBayesianDenoising:
    def __init__(self):
        self.prior_functions = {
            'squared_l2': self._squared_l2_prior,
            'l2': self._l2_prior,
            'huber_l1': self._huber_l1_prior
        }

    def _squared_l2_prior(self, u, gamma=None):
        """Prior A: Squared L2-norm of vector difference"""
        function_value = np.sum(np.sum(u**2, axis=-1))
        gradient = 2*u
        return function_value, gradient

    def _l2_prior(self, u, gamma=None):
        """Prior B: L2-norm of vector difference"""
        norms = np.sqrt(np.sum(u**2, axis=-1) + 1e-10)  
        function_value = np.sum(norms)
        # Gradient computation needs to account for vector nature
        gradient = u / norms[..., np.newaxis]
        return function_value, gradient

    def _huber_l1_prior(self, u, gamma):
        """Prior C: Huber-regularized L1-norm of vector difference"""
        vector_norms = np.sqrt(np.sum(u**2, axis=-1) + 1e-10)
        mask = vector_norms <= gamma
        
        function_value = np.sum(0.5*(vector_norms**2)*mask + 
                              (gamma*vector_norms - 0.5*gamma**2)*(~mask))
        
        scale = np.where(mask, 1.0, gamma/vector_norms)
        gradient = u * scale[..., np.newaxis]
        return function_value, gradient

    def _calculate_likelihood(self, x, y):
        """Gaussian likelihood for vector-valued data"""
        function_value = np.sum((x-y)**2)
        gradient = 2*(x-y)
        return function_value, gradient

    def _calculate_prior(self, x, prior_func, gamma):
        """Calculate prior using 4-neighbor system with wraparound for vector-valued data"""
        # Calculate differences with neighbors
        diff_up = x - np.roll(x, 1, axis=0)
        diff_down = x - np.roll(x, -1, axis=0)
        diff_left = x - np.roll(x, 1, axis=1)
        diff_right = x - np.roll(x, -1, axis=1)

        # Calculate prior values and gradients for each direction
        prior_values = []
        prior_grads = []
        for diff in [diff_up, diff_down, diff_left, diff_right]:
            value, grad = prior_func(diff, gamma)
            prior_values.append(value)
            prior_grads.append(grad)

        total_prior = sum(prior_values)
        total_grad = sum(prior_grads)
        return total_prior, total_grad

    def _calculate_posterior(self, x, y, prior_type, alpha, gamma):
        """Calculate posterior probability and its gradient"""
        prior_func = self.prior_functions[prior_type]
        likelihood, likelihood_grad = self._calculate_likelihood(x, y)
        prior, prior_grad = self._calculate_prior(x, prior_func, gamma)
        
        posterior = alpha*prior + (1-alpha)*likelihood
        posterior_grad = alpha*prior_grad + (1-alpha)*likelihood_grad
        
        return posterior, posterior_grad

    def denoise(self, noisy_img, clean_img=None, prior_type='squared_l2', 
                alpha=0.6, gamma=0.5, max_iter=150, save_dir=None):
        """
        Main denoising function for vector-valued images
        Args:
            noisy_img: Input noisy image [height, width, channels]
            clean_img: Ground truth image (optional)
            prior_type: 'squared_l2', 'l2', or 'huber_l1'
            alpha: Weight between prior and likelihood (0-1)
            gamma: Parameter for huber prior
            max_iter: Maximum number of iterations
            save_dir: Directory to save results
        """
        x = noisy_img.copy()
        y = noisy_img.copy()
        
        step_size = 1e-2
        min_step_size = 1e-8
        
        objective_values = []
        initial_posterior, _ = self._calculate_posterior(x, y, prior_type, alpha, gamma)
        objective_values.append(initial_posterior)
        
        for iteration in range(max_iter):
            posterior, posterior_grad = self._calculate_posterior(x, y, prior_type, alpha, gamma)
            x_new = x - step_size * posterior_grad
            new_posterior, _ = self._calculate_posterior(x_new, y, prior_type, alpha, gamma)
            
            if new_posterior < posterior:
                step_size *= 1.1
                x = x_new
            else:
                step_size *= 0.5
            
            objective_values.append(new_posterior)
            
            if step_size < min_step_size:
                break
        
        if clean_img is not None:
            final_rrmse = self._calculate_rrmse(clean_img, x)
            print(f'Final RRMSE: {final_rrmse:.5f}')
        
        if save_dir:
            self._save_results(noisy_img, x, clean_img, objective_values, 
                             prior_type, save_dir)
        
        return x, objective_values

    def _calculate_rrmse(self, A, B):
        """Calculate Relative Root Mean Square Error for vector-valued data"""
        return np.sqrt(np.sum((A-B)**2)/np.sum(A**2))

    def _save_results(self, noisy_img, denoised_img, clean_img, 
                     objective_values, prior_type, save_dir):
        """Save denoising results and plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(objective_values)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        plt.title(f'Objective Function Evolution ({prior_type} prior)')
        plt.savefig(os.path.join(save_dir, f'{prior_type}_objective.png'))
        plt.close()
        
        magnitude = lambda x: x
        
        plt.figure(figsize=(8, 8))
        im = plt.imshow(magnitude(noisy_img), cmap='jet')
        plt.colorbar(im)
        plt.title('Noisy Image')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{prior_type}_noisy.png'))
        plt.close()
        
        plt.figure(figsize=(8, 8))
        im = plt.imshow(magnitude(denoised_img), cmap='jet')
        plt.colorbar(im)
        plt.title(f'Denoised Image ({prior_type})')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{prior_type}_denoised.png'))
        plt.close()
        
        if clean_img is not None:
            plt.figure(figsize=(8, 8))
            im = plt.imshow(magnitude(clean_img), cmap='jet')
            plt.colorbar(im)
            plt.title('Ground Truth')
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'{prior_type}_ground_truth.png'))
            plt.close()

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
        print(f"Error loading images from {mat_file_path}: {e}")
        return []
if __name__ == '__main__':
    file =  load_images_from_mat('../data/assignmentImageDenoising_microscopy.mat')
    noisy_img = np.transpose(file[0],(1,2,0))
    clean_img = np.transpose(file[1],(1,2,0))

    def normalize_image(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    
    noisy_img_normalized = normalize_image(noisy_img)
    clean_img_normalized = normalize_image(clean_img)
    # fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    denoiser = VectorBayesianDenoising()
    # plt.colorbar(noisy_img, ax=axes[0])
    # plt.colorbar(clean_img, ax=axes[1])
    # plt.colorbar(noisy_img_normalized, ax=axes[2])
    # plt.colorbar(clean_img_normalized, ax=axes[3])

    # plt.imshow(noisy_img,cmap='gray')    
    # plt.show()
    # plt.close()
    # plt.imshow(noisy_img_normalized,cmap='gray')
    # plt.show()
    # plt.close()
    # plt.imshow(clean_img,cmap='gray')
    # plt.show()
    # plt.close()
    # plt.imshow(clean_img_normalized,cmap='gray')
    # plt.show()
    # plt.close()
    # Test all three priors
    for prior_type in ['squared_l2', 'l2', 'huber_l1']:
        denoised_img, objective_values = denoiser.denoise(
            noisy_img=noisy_img_normalized,
            clean_img=clean_img_normalized,
            prior_type=prior_type,
            alpha=0.6,
            gamma=0.5,
            max_iter=150,
            save_dir=f'denoising_results_microscopy_{prior_type}'
        ) 