from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def calculatePCAResults(feature, n_components):
    '''
    feature: feature vector
    n_components: 차원 축소 후 차원의 수
    '''
    PCA = PCA(n_components=n_components)
    PCA.fit(feature)
    return PCA.transform(feature)

def calcuateTSNEResults(feature, n_components):
    ''''
    feature: feature vector
    n_components: 차원 축소 후 차원의 수
    '''
    return TSNE(n_components=n_components).fit_transform(feature)
