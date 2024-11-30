import os
import pandas as pd
from nltk.corpus import stopwords
import re
import string
import random

def preprocess_tweet(tweet):
    # Remove tweet ID and timestamp (assuming they are at the beginning)
    tweet = re.sub(r'^\d+\s+\d+:\d+:\d+\s+', '', tweet)

    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)

    # Remove hashtag symbols
    tweet = re.sub(r'#', '', tweet)

    # Remove URLs
    tweet = re.sub(r'http\S+|www.\S+', '', tweet)

    # Convert to lowercase
    tweet = tweet.lower()

    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    tweet = tweet.strip()
    tweet = re.sub(r'\s+', ' ', tweet)

    # Optionally, remove stopwords
    stop_words = set(stopwords.words('english'))
    words = tweet.split()
    words = [word for word in words if word not in stop_words]

    return set(words)


def jaccard_distance(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1  # Define distance as 1 if both sets are empty
    return 1 - len(intersection) / len(union)


class KMeansJaccard:
    def __init__(self, K, max_iterations=100):
        self.K = K
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = [[] for _ in range(K)]

    def fit(self, data):
        # Initialize centroids randomly
        self.centroids = random.sample(data, self.K)

        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}")
            # Assign clusters
            new_clusters = [[] for _ in range(self.K)]
            for tweet in data:
                distances = [jaccard_distance(tweet, centroid) for centroid in self.centroids]
                min_distance_index = distances.index(min(distances))
                new_clusters[min_distance_index].append(tweet)

            # Check for convergence
            if new_clusters == self.clusters:
                print("Convergence reached.")
                break
            self.clusters = new_clusters

            # Update centroids
            new_centroids = []
            for idx, cluster in enumerate(self.clusters):
                if not cluster:
                    # If a cluster is empty, reinitialize its centroid
                    new_centroid = random.choice(data)
                else:
                    # Choose the tweet with the minimum total distance to all others in the cluster
                    min_distance = float('inf')
                    new_centroid = cluster[0]
                    for candidate in cluster:
                        total_distance = sum(jaccard_distance(candidate, other) for other in cluster)
                        if total_distance < min_distance:
                            min_distance = total_distance
                            new_centroid = candidate
                new_centroids.append(new_centroid)
            self.centroids = new_centroids

    def get_clusters(self):
        return self.clusters


def load_data(file_path):
    tweets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tweets.append(line)
    return tweets


def compute_sse(clusters, centroids):
    sse = 0
    for idx, cluster in enumerate(clusters):
        centroid = centroids[idx]
        for tweet in cluster:
            distance = jaccard_distance(tweet, centroid)
            sse += distance ** 2
    return sse


def main():
    # Specify the path to your data file
    data_file = 'usnewshealth.txt'  # Change this to your chosen file
    if not os.path.exists(data_file):
        print(f"Data file {data_file} does not exist.")
        return

    print("Loading data...")
    raw_tweets = load_data(data_file)
    print(f"Total tweets loaded: {len(raw_tweets)}")

    print("Preprocessing tweets...")
    preprocessed_tweets = [preprocess_tweet(tweet) for tweet in raw_tweets]
    print("Preprocessing completed.")

    # Define different K values
    K_values = [5, 10, 15, 20, 25]
    results = []

    for K in K_values:
        print(f"\nClustering with K={K}")
        kmeans = KMeansJaccard(K=K, max_iterations=100)
        kmeans.fit(preprocessed_tweets)
        clusters = kmeans.get_clusters()
        centroids = kmeans.centroids
        sse = compute_sse(clusters, centroids)

        # Record cluster sizes
        cluster_sizes = [len(cluster) for cluster in clusters]

        # Prepare result entry
        result_entry = {
            'K': K,
            'SSE': sse,
            'Cluster Sizes': cluster_sizes
        }
        results.append(result_entry)

        print(f"Finished clustering for K={K}")
        print(f"SSE: {sse}")
        print(f"Cluster Sizes: {cluster_sizes}")

    # Create results table
    results_df = pd.DataFrame(results)
    print("\nClustering Results:")
    print(results_df)


if __name__ == "__main__":
    main()
