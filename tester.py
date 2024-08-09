from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Example text data
texts = ["And don'T send more of those armored super-cops! they're useless! the perps here have ray-guns and boomerangs, yeah, you heard me...",
         "you are evil",
         "text for character 3"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(texts).toarray()

# Example character and panel tags
character_tags = [
    ["solo", "1boy", "male focus", "bald", "facial hair", "motor vehicle", "beard", "motorcycle"],
    ["1girl", "solo", "blonde hair", "ponytail", "uniform", "hairclip", "blue eyes"],
    ["1boy", "male focus", "solo", "black hair", "goggles", "short hair", "goggles on head", "closed eyes", "upper body", "muscular male", "pectorals", "muscular", "chef", "white jacket", "large pectorals", "speech bubble", "outdoors", "sideburns"],
    ["1boy", "male focus", "solo", "upper body", "looking at viewer", "black hair", "sunglasses", "glasses", "white jacket", "v-shaped eyebrows", "short hair", "closed mouth"],
    ["1boy", "male focus", "solo", "facial hair", "shirt", "green eyes", "blue shirt", "stubble", "teeth", "brown hair", "wide-eyed", "blonde hair", "clenched teeth", "short hair", "looking at viewer"],
    ["solo", "1boy", "facial hair", "male focus", "black background", "profile", "collared shirt", "shirt", "brown hair", "blue shirt", "short hair"],
    ["1boy", "male focus", "glasses", "beard", "facial hair", "speech bubble", "solo", "shirt", "round eyewear", "english text", "upper body", "collared shirt", "orange hair", "red shirt"],
    ["1boy", "male focus",  "facial hair", "solo", "blonde hair", "green eyes", "stubble", "portrait", "teeth", "looking at viewer", "speech bubble"],
    ["1boy", "male focus", "solo", "gloves", "black gloves", "brown hair", "glasses", "upper body", "short hair", "goggles", "holding"],
    ["male focus", "1boy", "facial hair", "solo", "shirt", "speech bubble", "pants", "standing", "hands in pockets", "furry", "red shirt", "fat man", "fat", "furry male", "glasses", "beard", "english text", "bald", "mustache", "pocket", "collared shirt", "black pants"],
    ["1boy", "male focus", "facial hair", "glasses", "beard", "solo", "round eyewear", "blue eyes", "speech bubble", "colored skin", "shirt", "portrait", "teeth", "orange shirt"],
    ["1boy", "male focus", "solo", "shirt", "orange hair", "pants", "arms behind back", "orange shirt", "from behind", "walking", "full body", "facial hair", "blurry", "short sleeves", "indoors", "red shirt"],
    ["solo", "1boy", "box", "lab coat", "male focus", "standing", "cardboard box", "long sleeves", "gun", "holding", "weapon", "from behind", "paper", "full body", "holding gun"],
    ["gloves", "solo", "glasses", "black gloves", "1boy", "male focus", "brown hair", "adjusting eyewear"],
    ["facial hair", "male focus", "glasses", "bara", "mature male", "1boy", "short hair", "muscular male", "beard", "from side", "muscular", "solo", "mustache", "shirt", "thick mustache", "close-up", "thick eyebrows", "old man", "red hair", "orange hair", "upper body"],
    ["glasses", "beard", "facial hair", "male focus", "blue eyes", "1boy", "english text", "adjusting eyewear", "speech bubble", "mustache", "orange hair"],
    ["english text", "1boy", "male focus", "speech bubble", "black hair", "upper body", "looking at viewer", "solo", "red-framed eyewear", "smile", "closed mouth", "glasses", "eyewear on head"],
    ["male focus", "1boy", "solo", "facial hair", "goggles", "thick eyebrows", "black hair", "goggles on head", "short hair", "heart", "stubble", "sideburns", "dark-skinned male", "headset"],
    ["solo", "1boy", "male focus", "indoors", "lab coat", "leaning forward", "black hair", "gloves", "long sleeves", "black gloves"],
    ["english text", "1boy", "male focus", "goggles", "speech bubble", "clenched teeth", "teeth", "solo", "black hair", "goggles on head", "headset", "white border", "border"],
    ["male focus", "1boy", "solo", "walking", "from behind", "pants", "shirt", "full body", "blurry", "indoors", "facing away", "short hair"],
    ["solo", "blurry", "male focus", "1boy", "lab coat", "from behind", "blurry background", "standing", "pants", "brown hair", "depth of field", "short hair"],
    ["1girl", "blurry", "solo", "depth of field", "skirt", "from behind", "blurry background", "standing", "long sleeves"],
    ["1boy", "male focus", "solo", "coat", "black hair", "pants", "standing", "short hair", "white coat", "from behind", "border", "black pants", "motor vehicle"]
]
panel_tags = [["UrbanSetting", "Night","Alleyway","Fence","Building","Windows","Standing","Moody","Gritty","Noir","Crime","BoldLines","Shading","Dramatic"],
              ["Waiting", "Thinking","Seated","Indoor","Window","Home","Office","Partially Open Blinds","Wall Mounted Lighting"],
              ["Desk", "Chair","Minimalist Environment","Office Setting"]]

# One-hot encoding for character tags
mlb_char_tags = MultiLabelBinarizer()
character_tag_features = mlb_char_tags.fit_transform(character_tags)

# One-hot encoding for panel tags
mlb_panel_tags = MultiLabelBinarizer()
panel_tag_features = mlb_panel_tags.fit_transform(panel_tags)

# Normalize features
text_features = normalize(text_features)
character_tag_features = normalize(character_tag_features)
panel_tag_features = normalize(panel_tag_features)

# Define weights
character_tag_weight = 1.0  # Example weight
panel_tag_weight = 1.0      # Example weight

# Apply weights
weighted_character_tag_features = character_tag_features * character_tag_weight
weighted_panel_tag_features = panel_tag_features * panel_tag_weight

# Combine all features into a single feature vector for each character
#combined_features = np.hstack((weighted_character_tag_features, weighted_panel_tag_features))

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_clusters = kmeans.fit_predict(character_tag_features)

# DBSCAN clustering
dbscan = DBSCAN(eps=2000, min_samples=1)
dbscan_clusters = dbscan.fit_predict(character_tag_features)

from sklearn.cluster import AgglomerativeClustering

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5)  # You can change n_clusters as needed
agglo_clusters = agglo.fit_predict(character_tag_features)

print("Agglomerative Clustering assignments:", agglo_clusters)

from sklearn.cluster import SpectralClustering

# Spectral Clustering
spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors')
spectral_clusters = spectral.fit_predict(character_tag_features)

print("Spectral Clustering assignments:", spectral_clusters)

from sklearn.cluster import AffinityPropagation

# Affinity Propagation
affinity = AffinityPropagation(random_state=42)
affinity_clusters = affinity.fit_predict(character_tag_features)

print("Affinity Propagation assignments:", affinity_clusters)

from sklearn.metrics.pairwise import cosine_similarity

# Cosine Similarity
similarity_matrix = cosine_similarity(character_tag_features)

print("Cosine Similarity Matrix:", similarity_matrix)

from sklearn.mixture import GaussianMixture

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=42)
gmm_clusters = gmm.fit_predict(character_tag_features)

print("Gaussian Mixture Model assignments:", gmm_clusters)

from sklearn.cluster import Birch

# Birch Clustering
birch = Birch(n_clusters=5)
birch_clusters = birch.fit_predict(character_tag_features)
#Birch Clustering assignments: [3 2 1 1 4 4 3 4 1 3 3 0 0 1 3 3 1 1 1 1]
#Birch Clustering assignments: [2 3 0 0 4 4 2 4 0 2 2 1 1 0 2 2 0 0 0 0 1 1 1 1]
#Birch Clustering assignments: [2 3 5 1 4 4 2 4 1 2 2 0 0 1 2 2 1 5 1 5 0 0 0 0]
print("Birch Clustering assignments:", birch_clusters)



# Print the cluster assignments
print("K-Means Cluster assignments:", kmeans_clusters)
print("DBSCAN Cluster assignments:", dbscan_clusters)
