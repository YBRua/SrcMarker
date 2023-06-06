import os
import matplotlib.pyplot as plt

from collections import defaultdict

if __name__ == '__main__':
    DATASET_DIR = './data/github_c'
    n_samples = defaultdict(int)

    for author_subdir_base in os.listdir(DATASET_DIR):
        author_subdir = os.path.join(DATASET_DIR, author_subdir_base)
        assert os.path.isdir(author_subdir)
        for source_file in os.listdir(author_subdir):
            file_path = os.path.join(author_subdir, source_file)
            assert os.path.isfile(file_path)
            n_samples[author_subdir_base] += 1

    for author, n in n_samples.items():
        print(f'{author}: {n}')

    print(f'Max samples: {max(n_samples.values())}')
    print(f'  From author: {max(n_samples, key=n_samples.get)}')
    print(f'Min samples: {min(n_samples.values())}')
    print(f'  From author: {min(n_samples, key=n_samples.get)}')

    author_list = list(n_samples.keys())
    n_samples_list = [n_samples[author] for author in author_list]

    fig, ax = plt.subplots(1, 1)
    ax.hist(n_samples_list, bins=15)
    ax.set_xlabel('# Samples')
    ax.set_ylabel('# Authors')
    ax.set_title('Distribution of # Samples per Author')
    plt.show()
