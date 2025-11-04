import os
import sys
import numpy as np


if __name__ == "__main__":
    # Add the parent directory to Python path
    latent_abc_smc_dir = "/home/jp4474/latent-abc-smc"
    if latent_abc_smc_dir not in sys.path:
        sys.path.insert(0, latent_abc_smc_dir)
    
    os.chdir(latent_abc_smc_dir)
    from systems import SpatialSIR3D
    
    generations = np.load("/home/jp4474/latent-abc-smc/spatialSIR3D_d128_ed64_6_8_4_8_vae_mask_0.15_beta_0.1/spatial_sir_inference_pairwise_cosine.npz", allow_pickle=True)['generations']
    last_generation = generations[-1]
    particles = last_generation['particles']

    time_space = np.linspace(0, 15, num=100, endpoint=True)
    spatial_abc = SpatialSIR3D(time_space=time_space,)

    inferred_data = np.zeros((1000, 3, time_space.shape[0]))

    for i in range(1000):
        particle = particles[i]

        aggregated = np.zeros((10, 3, time_space.shape[0]))

        for j in range(10):
            data = spatial_abc.simulate(particle)[0]
            proportion = data.sum(axis=(2, 3)) / (80 * 80)
            aggregated[j] = proportion

        inferred_data[i] = aggregated.mean(axis=0)

    np.savez_compressed("aggregated_results_0.01.npz", inferred_data=inferred_data)

    print("End of script")