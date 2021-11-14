import numpy as np
import pandas as pd
import seaborn as sns
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise import KNNBasic
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt

reader = Reader(line_format='user item rating timestamp', sep=',', 
    rating_scale=(0.5, 5.0), skip_lines=1)

ratings = Dataset.load_from_file("/Users/dorymauretour/Hw5ML/ratings_small.csv", 
    reader=reader)

PMF = SVD(biased=False) # probabalistic matrix factorization
UCF = KNNBasic(sim_options={'user_based': True}) # user-based collaborative filtering
ICF = KNNBasic(sim_options={'user_based': False}) # item-based collaborative filtering

PMF_results = cross_validate(PMF, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(PMF_results)

UCF_results = cross_validate(UCF, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(UCF_results)

ICF_results = cross_validate(ICF, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(ICF_results)

data = [[np.mean(UCF_results['test_rmse']), np.mean(UCF_results['test_mae'])],
        [np.mean(ICF_results['test_rmse']), np.mean(ICF_results['test_mae'])],
        [np.mean(PMF_results['test_rmse']), np.mean(PMF_results['test_mae'])]]

df = pd.DataFrame(data, columns=['RMSE', 'MAE'], index=['UCF', 'ICF', 'PMF'])

print(df)
fig = df.plot.bar(title='RMSE and MAE Comparison of UCF, ICF, and PMF', ylim=(0.5, 1.05))
plt.show()

UCF_cosine = KNNBasic(sim_options={'name': 'cosine', 'user_based': True}) # user-based collaborative filtering using cosine similarity
UCF_msd = KNNBasic(sim_options={'name': 'MSD', 'user_based': True}) # user-based collaborative filtering using mean squared difference
UCF_pearson = KNNBasic(sim_options={'name': 'pearson', 'user_based': True}) # user-based collaborative filtering using Pearson correlation coefficient

ICF_cosine = KNNBasic(sim_options={'name': 'cosine', 'user_based': False}) # item-based collaborative filtering using cosine similarity
ICF_msd = KNNBasic(sim_options={'name': 'MSD', 'user_based': False}) # item-based collaborative filtering using mean squared difference
ICF_pearson = KNNBasic(sim_options={'name': 'pearson', 'user_based': False}) # item-based collaborative filtering using Pearson correlation coefficient

UCF_cosine_results = cross_validate(UCF_cosine, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(UCF_cosine_results)

UCF_msd_results = cross_validate(UCF_msd, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(UCF_msd_results)

UCF_pearson_results = cross_validate(UCF_pearson, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(UCF_pearson_results)

ICF_cosine_results = cross_validate(ICF_cosine, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(ICF_cosine_results)

ICF_msd_results = cross_validate(ICF_msd, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(ICF_msd_results)

ICF_pearson_results = cross_validate(ICF_pearson, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(ICF_pearson_results)

UCF_sim_data = [[np.mean(UCF_cosine_results['test_rmse']), np.mean(UCF_cosine_results['test_mae'])],
                [np.mean(UCF_msd_results['test_rmse']), np.mean(UCF_msd_results['test_mae'])],
                [np.mean(UCF_pearson_results['test_rmse']), np.mean(UCF_pearson_results['test_mae'])]]

UCF_sim_df = pd.DataFrame(UCF_sim_data, columns=['RMSE', 'MAE'], index=['Cosine', 'MSD', 'Pearson'])

ICF_sim_data = [[np.mean(ICF_cosine_results['test_rmse']), np.mean(ICF_cosine_results['test_mae'])],
                [np.mean(ICF_msd_results['test_rmse']), np.mean(ICF_msd_results['test_mae'])],
                [np.mean(ICF_pearson_results['test_rmse']), np.mean(ICF_pearson_results['test_mae'])]]

ICF_sim_df = pd.DataFrame(ICF_sim_data, columns=['RMSE', 'MAE'], index=['Cosine', 'MSD', 'Pearson'])

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4))
fig.suptitle('Similarity Measures Comparison')
plt.show()

print('UCF sim data\n', UCF_sim_df, '\n')
UCF_sim_df.plot.bar(ax=ax1, ylim=(0.5, 1.05))
ax1.title.set_text('User-Based Collaborative Filtering')
plt.show()

print('ICF sim data\n', ICF_sim_df, '\n')
ICF_sim_df.plot.bar(ax=ax2, ylim=(0.5, 1.05))
ax2.title.set_text('Item-Based Collaborative Filtering')
plt.show()

UCF_cosine = KNNBasic(sim_options={'name': 'cosine', 'user_based': True}) # user-based collaborative filtering using cosine similarity
UCF_msd = KNNBasic(sim_options={'name': 'MSD', 'user_based': True}) # user-based collaborative filtering using mean squared difference
UCF_pearson = KNNBasic(sim_options={'name': 'pearson', 'user_based': True}) # user-based collaborative filtering using Pearson correlation coefficient

ICF_cosine = KNNBasic(sim_options={'name': 'cosine', 'user_based': False}) # item-based collaborative filtering using cosine similarity
ICF_msd = KNNBasic(sim_options={'name': 'MSD', 'user_based': False}) # item-based collaborative filtering using mean squared difference
ICF_pearson = KNNBasic(sim_options={'name': 'pearson', 'user_based': False}) # item-based collaborative filtering using Pearson correlation coefficient

UCF_k_results = []

for k in range(10, 101, 10):
    UCF_k = KNNBasic(k=k, sim_options={'user_based': True})
    result = cross_validate(UCF_k, ratings, measures=['RMSE', 'MAE'], cv=5)
    UCF_k_results.append([np.mean(result['test_rmse']), np.mean(result['test_mae'])])
    
ICF_k_results = []

for k in range(10, 101, 10):
    ICF_k = KNNBasic(k=k, sim_options={'user_based': False})
    result = cross_validate(ICF_k, ratings, measures=['RMSE', 'MAE'], cv=5)
    ICF_k_results.append([np.mean(result['test_rmse']), np.mean(result['test_mae'])])

UCF_k_df = pd.DataFrame(UCF_k_results, columns=['RMSE', 'MAE'], index=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
ICF_k_df = pd.DataFrame(ICF_k_results, columns=['RMSE', 'MAE'], index=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 4))
fig.suptitle('Number of Neighbors Comparison')
plt.show()

print('UCF k data\n', UCF_k_df, '\n')
UCF_k_df.plot.bar(ax=ax1, ylim=(0.5, 1.05))
ax1.set_xlabel('number of neighbors')
ax1.title.set_text('User-Based Collaborative Filtering')
plt.show()

print('ICF k data\n', ICF_k_df, '\n')
ICF_k_df.plot.bar(ax=ax2, ylim=(0.5, 1.05))
ax2.set_xlabel('number of neighbors')
ax2.title.set_text('Item-Based Collaborative Filtering')
plt.show()




