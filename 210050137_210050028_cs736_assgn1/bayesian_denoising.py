import numpy as np
import matplotlib.pyplot as plt
import os

class BayesianDenoising:
    def __init__(self):
        self.prior_functions = {
            'quadratic': self._quadratic_prior,
            'huber': self._huber_prior,
            'discontinuity_adaptive': self._discontinuity_adaptive_prior
        }

    def _quadratic_prior(self, u, gamma=None):
        """g1(u) := |u|^2"""
        function_value = np.sum(np.abs(u)**2)
        gradient = 2*u
        return function_value, gradient

    def _huber_prior(self, u, gamma):
        """g2(u) := 0.5|u|^2 when |u| ≤ gamma and gamma|u|-0.5gamma^2 when |u| > gamma"""
        mask = np.abs(u) <= gamma
        function_value = np.sum(0.5*(np.abs(u)**2)*mask + 
                              (gamma*np.abs(u) - 0.5*gamma**2)*(~mask))
        gradient = u*mask + gamma*np.sign(u)*(~mask)
        return function_value, gradient

    def _discontinuity_adaptive_prior(self, u, gamma):
        """g3(u) := gamma|u|-gamma^2*log(1 + |u|/gamma)"""
        function_value = np.sum(gamma*np.abs(u) - gamma**2*np.log(1 + np.abs(u)/gamma))
        gradient = gamma*u/(gamma + np.abs(u))
        return function_value, gradient

    def _calculate_likelihood(self, x, y):
        """Gaussian likelihood with alpha=1"""
        function_value = np.sum(np.abs(x-y)**2)
        gradient = 2*(x-y)
        return function_value, gradient

    def _calculate_prior(self, x, prior_func, gamma):
        """Calculate prior using 4-neighbor system with wraparound"""
        diff_up = x - np.roll(x, 1, axis=0)
        diff_down = x - np.roll(x, -1, axis=0)
        diff_left = x - np.roll(x, 1, axis=1)
        diff_right = x - np.roll(x, -1, axis=1)

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

    def denoise(self, noisy_img, clean_img=None, prior_type='quadratic', 
                alpha=0.6, gamma=0.5, max_iter=150, save_dir=None):
        """
        Main denoising function
        Args:
            noisy_img: Input noisy image
            clean_img: Ground truth image (optional, for RRMSE calculation)
            prior_type: 'quadratic', 'huber', or 'discontinuity_adaptive'
            alpha: Weight between prior and likelihood (0-1)
            gamma: Parameter for non-quadratic priors
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
        """Calculate Relative Root Mean Square Error"""
        return np.sqrt(np.sum((A-B)**2)/np.sum(A**2))

    def _save_results(self, noisy_img, denoised_img, clean_img, 
                     objective_values, prior_type, save_dir):
        """Save denoising results and plots"""
        os.makedirs(save_dir, exist_ok=True)
        plt.figure()
        plt.plot(objective_values)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        plt.title(f'Objective Function Evolution ({prior_type} prior)')
        plt.savefig(os.path.join(save_dir, f'{prior_type}_objective.png'))
        plt.close()
        
        plt.imsave(os.path.join(save_dir, 'noisy.png'), noisy_img, cmap='gray')
        plt.imsave(os.path.join(save_dir, f'{prior_type}_denoised.png'), 
                  denoised_img, cmap='gray')
        if clean_img is not None:
            plt.imsave(os.path.join(save_dir, 'ground_truth.png'), 
                      clean_img, cmap='gray')

from scipy.io import loadmat

if __name__ == '__main__':
    data_path = '../data/assignmentImageDenoising_brainMRIslice.mat'
    data = loadmat(data_path)
    noisy_img = data['brainMRIsliceNoisy']
    clean_img = data['brainMRIsliceOrig']

    def normalize(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    noisy_img = normalize(noisy_img)
    clean_img = normalize(clean_img)


    denoiser = BayesianDenoising()
    print("Original Image RRMSE:", denoiser._calculate_rrmse(clean_img, noisy_img))
    denoised_img, objective_values = denoiser.denoise(
        noisy_img=noisy_img,
        clean_img=clean_img, 
        prior_type='huber',  
        alpha= 0.7926919367283951,  
        gamma=0.8*0.005791666666666667, 
        max_iter=150,
        save_dir='denoising_results_brain' 
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(clean_img, cmap='gray')
    plt.title('Clean Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(denoised_img, cmap='gray')
    plt.title('Denoised Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
