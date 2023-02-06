import numpy as np

def pinball(u, alpha):
    u = np.array(u)
    return alpha*u - (u < 0) * u

def conformalAdaptStable(betas, alpha, gammas, sigma=1/500, eta=2.8):
    T = len(betas)
    k = len(gammas)

    alphaSeq = np.zeros(T) + alpha
    errSeqAdapt = np.zeros(T)
    errSeqFixed = np.zeros(T)
    gammaSeq = np.zeros(T)
    meanAlphaSeq = np.zeros(T)
    meanErrSeq = np.zeros(T)
    meanGammas = np.zeros(T)
    
    expertAlphas  = np.zeros(k) + alpha
    expertWs  = np.ones(k)
    curExpert = np.random.choice(np.arange(k))
    expertCumulativeLosses = np.zeros(k)
    expertProbs = np.zeros(k) + 1/k

    for t in range(0, T):
        alphat = expertAlphas[curExpert]
        alphaSeq[t] = alphat
        errSeqAdapt[t] = int(alphat > betas[t])
        errSeqFixed[t] = int(alpha > betas[t])
        gammaSeq[t] = gammas[curExpert]
        meanAlphaSeq[t] = np.sum(expertProbs * expertAlphas)
        meanErrSeq[t] = int(meanAlphaSeq[t] > betas[t])
        meanGammas[t] = np.sum(expertProbs*gammas)

        expertLosses = pinball(betas[t] - expertAlphas, alpha)
        
        expertAlphas = expertAlphas + gammas * (alpha - np.array(expertAlphas > betas[t], dtype=int))

        if (eta < np.inf):
            expertBarWs = expertWs * np.exp(-eta*expertLosses)

            expertNextWs = (1-sigma)*expertBarWs/sum(expertBarWs) + sigma/k
            
            expertProbs = expertNextWs/sum(expertNextWs)
            curExpert = np.random.choice(np.arange(k), p=expertProbs)
            expertWs = expertNextWs
        else:
            expertCumulativeLosses = expertCumulativeLosses + expertLosses
            curExpert = np.where(expertCumulativeLosses == min(expertCumulativeLosses))[0][0]
    return alphaSeq, errSeqAdapt, errSeqFixed, gammaSeq, meanAlphaSeq, meanErrSeq, meanGammas