# Author: Sining Sun , Zhanheng Yang
from utils import *
import numpy as np
import scipy.cluster.vq as vq


num_gaussian = 5
num_iterations = 5
targets = ['Z','1','2','3','4','5','6','7','8','9','0']

class GMM:
    def __init__(self, D, K=5):
        assert(D>0)
        self.dim = D
        self.K = K
        #Kmeans Initial
        self.mu , self.sigma , self.pi = self.kmeans_initial()

    def kmeans_initial(self):
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)]
        for (l,d) in zip(labels,data):
            clusters[l].append(d)
        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c)*1.0 / len(data) for c in clusters])
        return mu , sigma , pi
    
    def gaussian(self , x , mu , sigma):
        """Calculate gaussion probability.
    
            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        D=x.shape[0]
        #求sigma行列式
        det_sigma = np.linalg.det(sigma)
        #求sigma逆矩阵
        inv_sigma = np.linalg.inv(sigma + 0.0001)
        mahalanobis = np.dot(np.transpose(x-mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x-mu))
        const = 1/((2*np.pi)**(D/2))
        return const * (det_sigma)**(-0.5) * np.exp(-0.5 * mahalanobis)
    
    def calc_log_likelihood(self , X):
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model 
        """

        log_llh = 0.0
        """
            FINISH by YOUSELF
        """
        result = np.zeros((X.shape[0],self.K))
        for i in range(self.K):
           for j in range(X.shape[0]):
                result[j,i] = self.pi[i]*self.gaussian(X[j],self.mu[i],self.sigma[i])
        log_llh = np.sum(np.log(np.sum(result,axis=1)))
        return log_llh

    def em_estimator(self , X):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model
            X (1937,39)
            mu (5,39)
            sigma (5,39,39)
            post_grob (1937,5)
        """
        log_llh = 0.0
        """ 
            FINISH by YOUSELF
        """
        #print(X.shape)
        #print(np.array(self.mu).shape)
        #print(np.array(self.sigma).shape)
        data_num = X.shape[0]
        all_prob = np.zeros((X.shape[0], self.K))
        for i in range(self.K):
            for j in range(X.shape[0]):
                all_prob[j,i] = self.pi[i]*self.gaussian(X[i],self.mu[i], self.sigma[i])
        #计算后验概率
        post_prob = all_prob/np.sum(all_prob,axis=0)
        #print(post_prob.shape)
        #重新估计参数
        self.pi = np.sum(post_prob,axis=0)/data_num
        for i in range(self.K):
            self.mu[i] = np.average(X, axis=0, weights=post_prob[:, i])
            COV = np.zeros((self.dim,self.dim))
            for j in range(data_num):
                #变成列向量
                res = (X[j]-self.mu[i]).reshape(-1,1)
                COV += post_prob[j,i]*np.dot(res, res.T)
            self.sigma[i] = COV/ np.sum(post_prob[:,i])
        log_llh = self.calc_log_likelihood(X)
        return log_llh


def train(gmms, num_iterations = num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
    return gmms

def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian) #Initial model
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()


