import os
import pandas as pd
from nltk.corpus import stopwords
import re
import string
import random

def preprocess_tweet(tweet):
    tweet = re.sub(r'^\d+\s+\d+:\d+:\d+\s+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'http\S+|www.\S+', '', tweet)
    tweet = tweet.lower()
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = tweet.strip()
    tweet = re.sub(r'\s+', ' ', tweet)
    stop_words = set(stopwords.words('english'))
    words = tweet.split()
    words = [word for word in words if word not in stop_words]

    return set(words)


def jaccard_distance(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1
    return 1 - len(intersection) / len(union)


class KMeansJaccard:
    def __init__(self, K, max_iterations=100):
        self.K = K
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = [[] for _ in range(K)]

    def fit(self, data):
        self.centroids = random.sample(data, self.K)

        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}")
            new_clusters = [[] for _ in range(self.K)]
            for tweet in data:
                distances = [jaccard_distance(tweet, centroid) for centroid in self.centroids]
                min_distance_index = distances.index(min(distances))
                new_clusters[min_distance_index].append(tweet)

            if new_clusters == self.clusters:
                print("Convergence reached.")
                break
            self.clusters = new_clusters

            new_centroids = []
            for idx, cluster in enumerate(self.clusters):
                if not cluster:
                    new_centroid = random.choice(data)
                else:
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
    data_file = 'usnewshealth.txt'
    if not os.path.exists(data_file):
        print("Cannot read file usnewshealth.txt!")
        return

    print("Load data")
    raw_tweets = load_data(data_file)
    print(f"Tweets loaded: {len(raw_tweets)}")

    print("Preprocessing tweets")
    preprocessed_tweets = [preprocess_tweet(tweet) for tweet in raw_tweets]

    K_values = [5, 10, 15, 20, 25]
    results = []

    for K in K_values:
        print(f"\nClustering with K={K}")
        kmeans = KMeansJaccard(K=K, max_iterations=100)
        kmeans.fit(preprocessed_tweets)
        clusters = kmeans.get_clusters()
        centroids = kmeans.centroids
        sse = compute_sse(clusters, centroids)

        cluster_sizes = [len(cluster) for cluster in clusters]

        result_entry = {
            'K': K,
            'SSE': sse,
            'Cluster Size': cluster_sizes
        }
        results.append(result_entry)

        print(f"Value of K={K}")
        print(f"SSE: {sse}")
        print(f"Cluster Size: {cluster_sizes}")

    results_df = pd.DataFrame(results)
    print("\nClustering Table:")
    print(results_df)


if __name__ == "__main__":
    main()
