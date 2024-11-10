import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import pickle

model_name = 'HalfCheetah-v5_163840.pkl'
with open(f"../results/{model_name}", 'rb') as f:
    model = pickle.load(f)

for individual, individual_id in model['population']:
    env = RecordVideo(gym.make('HalfCheetah-v5', render_mode='rgb_array'), video_folder="videos", name_prefix=f"HalfCheetah-v5-{model_name}-{str(individual_id)[:4]}")

    if individual_id in model['fitnesses']:
        fitness = model['fitnesses'][individual_id]
        print(f"Evaluating individual {str(individual_id)[:4]} with fitness {fitness}.")

        obs, _ = env.reset()

        # Each individual gets 1000 timesteps of interaction
        for _ in range(1000):
            action = individual.predict(obs)[:6]
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                continue

    env.close()