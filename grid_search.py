import numpy as np
import bayesian_denoising as bd
import sys

def grid_search_parameters(noisy_img, clean_img, function_type='quadratic', tolerance=1e-4, max_iterations=10):
    gamma_value_lower_bound = 1e-4
    gamma_value_upper_bound = 1.0
    beta_value_lower_bound = 1e-4
    beta_value_upper_bound = 1.0
    
    rrmse_final_min = float('inf')
    beta_opt = None
    gamma_opt = None
    
    denoiser = bd.BayesianDenoising()
    
    total_iterations = 0
    
    # Grid search with refinement
    for i in range(max_iterations):
        current_iteration_rrmse = float('inf')
        
        for gamma in np.linspace(gamma_value_lower_bound, gamma_value_upper_bound, 10):
            for beta in np.linspace(beta_value_lower_bound, beta_value_upper_bound, 10):
                total_iterations += 1
                
                denoised_img, _ = denoiser.denoise(
                    noisy_img=noisy_img,
                    clean_img=clean_img,
                    prior_type=function_type,
                    alpha=beta,
                    gamma=gamma
                )
                
                rrmse_final = np.sqrt(np.mean((denoised_img - clean_img)**2)) / np.sqrt(np.mean(clean_img**2))
                
                current_iteration_rrmse = min(current_iteration_rrmse, rrmse_final)
                if rrmse_final < rrmse_final_min:
                    rrmse_final_min = rrmse_final
                    beta_opt = beta
                    gamma_opt = gamma
                    print(f'RMSE: {rrmse_final_min:.6f} | Beta: {beta_opt:.6f} | Gamma: {gamma_opt:.6f}')
                
                # Update progress
                print(f'Ops: {total_iterations}')
        gamma_value_lower_bound = max(gamma_opt - gamma_opt/2, 0.00001)
        gamma_value_upper_bound = min(gamma_opt + gamma_opt/2, 1.0)
        
        beta_value_upper_bound = min(beta_opt + beta_opt/2, 1.0)
        beta_value_lower_bound = max(beta_opt - beta_opt/2, 0.00001)
        
    
    return beta_opt, gamma_opt, rrmse_final_min

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
    
    beta_opt, gamma_opt, best_rrmse = grid_search_parameters(noisy_img, clean_img, function_type='huber')
    
    print(f"Optimal parameters found:")
    print(f"Beta: {beta_opt}")
    print(f"Gamma: {gamma_opt}")
    print(f"Best RRMSE: {best_rrmse}")