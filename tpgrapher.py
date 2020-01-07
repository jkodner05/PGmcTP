import numpy as np
from math import log, factorial
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BOTH_GEN = 0
NO_GEN = 1
C1_GEN = 2
C2_GEN = 3

CYAN = '#3d85c6'
BLACK = '#111111'
GOLD = '#f1c232'
RED = '#e06666'

def get_label(c, l):  
    return mpatches.Patch(color=c, label=l) 

def plot_regionprobs(nops, bothps, c1ps, c2ps, title, labels):
    def plotline(data, c, w):
        line = plt.plot([100*i for i in data])
        plt.setp(line, color=c, linewidth=w, drawstyle="steps-mid", fillstyle="bottom")

    axes = plt.gca()
    axes.set_ylim([0,100])
    axes.set_xlim([0,len(nops)])
    plotline(nops, CYAN, 3)
    plotline(bothps, BLACK, 3)
    plotline(c1ps, GOLD, 3)
    plotline(c2ps, RED, 3)
    plt.title(title, fontsize=15)
    plt.ylabel("Generalization Probability", fontsize=12)
    plt.xlabel("Combined Vocab Size", fontsize=12)
    plt.legend(handles=[get_label(CYAN,labels[0]),get_label(BLACK,labels[3]),get_label(GOLD,labels[1]),get_label(RED,labels[2])],fontsize=15,loc="best")
    plt.savefig(title.replace(" ","").replace("+","").replace("&","-").replace("#","") + ".png")
    plt.show()

def prepdisp(mat):
    return 255-(255*mat).astype(np.uint8)

def plot_matrix(mat, title, labels, c=None, axislabels=["",""]):
    axes = plt.gca()
    axes.set_ylim([-0.5,mat.shape[0]-0.5])
    axes.set_xlim([-0.5,mat.shape[1]-0.5])
    plt.imshow(mat, interpolation="nearest", cmap=c)
#    if labels:
#        if len(labels) == 4:
#            plt.legend(handles=[get_label("cyan",labels[0]),get_label("blue",labels[1]),get_label("gold",labels[2]),get_label("red",labels[3])],fontsize=15, bbox_to_anchor=(2, 1))
#        elif len(labels) == 3:
#            plt.legend(handles=[get_label("blue",labels[0]),get_label("gold",labels[1]),get_label("red",labels[2])],fontsize=15,loc="best", bbox_to_anchor=(1, 0.5))
    plt.title(title, fontsize=15)
    plt.xlabel(axislabels[0], fontsize=12)
    plt.ylabel(axislabels[1], fontsize=12)
    plt.tight_layout()
    plt.savefig(title.replace(" ","").replace("+","").replace("&","-") + ".png")
    plt.show()

def binom(n,k):
    return float(factorial(n)) / (factorial(k) * factorial(n-k))

def hypergeometric(n,k,N,K):
    #n - total known
    #k - totak c1 known
    #N - total possible
    #K - total c1 possible
    return binom(K,k) * binom(N-K,n-k) / float(binom(N,n))

def tp(n, e):
    N = n+e
    if N > 1:
#        print N, e, N/log(N), e < N / log(N)
        return e < N / log(N)
    return False

def calc_matrices(maxc1, maxc2):
    
    tpmat = np.zeros((maxc1+1, maxc2+1))
    hypergmat = np.zeros((maxc1+1, maxc2+1))
    cummat = np.zeros((maxc1+1, maxc2+1))

    for c1 in range(0,maxc1+1):
        for c2 in range(0,maxc2+1):
#            print hypergeometric(c1+c2,c1,maxc1+maxc2,maxc1)
            hypergmat[c1,c2] = hypergeometric(c1+c2,c1,maxc1+maxc2,maxc1)
            c1gen = tp(c1,c2)
            c2gen = tp(c2,c1)
            if not c1gen and not c2gen:
                tpmat[c1, c2] = NO_GEN
            elif c1gen and c2gen:
                tpmat[c1, c2] = BOTH_GEN
            elif c1gen:# and c1/float(c2) < 2 and c2/float(c1)/2:
                tpmat[c1, c2] = C1_GEN
            elif c2gen:# and c1/float(c2) < 2 and c2/float(c1)/2:
                tpmat[c1, c2] = C2_GEN

#    cummat[np.where(tpmat != NO_GEN)] = 1
#    cummat = np.multiply(cummat,hypergmat)
    return tpmat, hypergmat


def display_matrices(tpmat, hypergmat, cummat, c1, c2, title, labels, axes):
    plot_matrix(tpmat, "%s: Tolerance Principle States" % title, labels, axislabels=axes)
    plot_matrix(1-hypergmat, "%s: Development Path Probabilities" % title, None, c="gray", axislabels=axes)
    plot_matrix(1-cummat,  "%s: Generalization Probabilities" % title, labels[1:], c="gray", axislabels=axes)
#    plt.imshow(tpmat, interpolation="nearest")
#    plt.imsave("tpmat_%s_%s.png"%(c1,c2), tpmat, interpolation="nearest")
#    plt.show()
#    plt.imshow(1-hypergmat, cmap="gray", interpolation="nearest")
#    plt.show()
#    plt.imshow(1-cummat, cmap="gray", interpolation="nearest")
#    plt.show()

def get_region_massmat(tpmat, hypergmat, region):
    massmat = np.zeros(tpmat.shape)
    massmat[np.where(tpmat == region)] = 1
    massmat = np.multiply(massmat,hypergmat/np.sum(hypergmat))
    return massmat

def get_massmat(tpmat, hypergmat, include_nogen=False):

    massmat = np.zeros((tpmat.shape[0], tpmat.shape[1], 3))
    nogenmassmat = get_region_massmat(tpmat, hypergmat, NO_GEN)
    bothgenmassmat = get_region_massmat(tpmat, hypergmat, BOTH_GEN)
    c1genmassmat = get_region_massmat(tpmat, hypergmat, C1_GEN)
    c2genmassmat = get_region_massmat(tpmat, hypergmat, C2_GEN)
    maxprob = max(np.max(c1genmassmat), max(np.max(c2genmassmat),np.max(bothgenmassmat)))
#    massmat[:,:,0] = tpmat
#    massmat[:,:,1] = tpmat
#    massmat[:,:,2] = tpmat/5
    massmat[:,:,1] += c2genmassmat/maxprob/1.4
    massmat[:,:,2] += c2genmassmat/maxprob/1.4
    massmat[:,:,0] += c2genmassmat/maxprob/8
    massmat[:,:,0] += bothgenmassmat/maxprob/1.2
    massmat[:,:,1] += bothgenmassmat/maxprob/1.2
    massmat[:,:,2] += bothgenmassmat/maxprob/1.2
    massmat[:,:,2] += c1genmassmat/maxprob/1.2
    massmat[:,:,1] += c1genmassmat/maxprob/8

    if include_nogen:
        massmat[:,:,0] += nogenmassmat/maxprob/2
        massmat[:,:,1] += nogenmassmat/maxprob/5

#    plt.imshow(255-(massmat*255).astype(np.uint8), interpolation="nearest")
#    plt.show()
#    plt.imshow(tpmat)
#    plt.show()
    return massmat
    

def get_mass(tpmat, hypergmat, region):
    massmat = np.zeros(tpmat.shape)
    massmat[np.where(tpmat == region)] = 1
    massmat = np.multiply(massmat,hypergmat/np.sum(hypergmat))
    return np.sum(massmat)


def get_masses(tpmat, hypergmat):
    mass = np.zeros((tpmat.shape[0], tpmat.shape[1], 3))
    nogenmass = get_mass(tpmat, hypergmat, NO_GEN)
    bothgenmass = get_mass(tpmat, hypergmat, BOTH_GEN)
    c1genmass = get_mass(tpmat, hypergmat, C1_GEN)
    c2genmass = get_mass(tpmat, hypergmat, C2_GEN)
    return nogenmass, bothgenmass, c1genmass, c2genmass


def display_quantmat(massmat):
    plt.imshow(255-(255*np.round(massmat/2,1)*2).astype(np.uint8), interpolation="nearest")
    plt.show()
#    threshmat = np.zeros(massmat.shape)
#    threshmat[np.where(massmat > thresh)] = 255
#    plt.imshow(255-255*(threshmat/threshmat).astype(np.uint8), interpolation="nearest")
#    plt.show()

def block_except(mat, exceptlist, blocker=1):
    blockmat = mat.copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i + j not in exceptlist:
                blockmat[i,j] = blocker
    return blockmat


def get_regionprobs_by_n(tpmat, hypergmat):
    totalv = np.sum(tpmat.shape)
    nops = []
    bothps = []
    c1ps = []
    c2ps = []
    for i in range(totalv):
        tpblock = block_except(tpmat, [i], 5)
        hypergblock = block_except(hypergmat, [i], 0.0)
        nop, bothp, c1p, c2p = get_masses(tpblock, hypergblock)
        nops.append(nop)
        bothps.append(bothp)
        c1ps.append(c1p)
        c2ps.append(c2p)
    return nops, bothps, c1ps, c2ps

def tpmatrix(c1, c2, title, labels, axes):
    print "\n", title
    tpmat, hypergmat = calc_matrices(c1, c2)
#    hypergmat[4,4] = 1
#    tpmat[4,4] = 10
    cummat = get_massmat(tpmat, hypergmat)
    nogenmass, bothgenmass, c1genmass, c2genmass = get_masses(tpmat, hypergmat)
    print "No Gen Mass", nogenmass
    print "Both Gen Mass", bothgenmass
    print "C1 Gen Mass", c1genmass, "\tSquared", (c1genmass)**2
    print "C2 Gen Mass", c2genmass, "\tSquared", (c2genmass)**2
    print "Both + C1 Gen Mass", bothgenmass+c1genmass, "\tSquared", (bothgenmass+c1genmass)**2
    print "Both + C2 Gen Mass", bothgenmass+c2genmass, "\tSquared", (bothgenmass+c2genmass)**2
    print "All Mass", nogenmass+bothgenmass+c1genmass+c2genmass

    display_matrices(1-get_massmat(tpmat,np.ones(tpmat.shape),include_nogen=True), hypergmat, cummat, c1, c2, title, labels, axes)
#    display_quantmat(cummat)

    nops, bothps, c1ps, c2ps = get_regionprobs_by_n(tpmat, hypergmat)
    plot_regionprobs(nops, bothps, c1ps, c2ps, "%s: Productivity by Vocab Size" % title, labels)

#    plt.imshow(prepdisp(block_except(cummat, [7])), interpolation="nearest")
#    plt.show()
#    tpblock = block_except(tpmat, [7],100)
#    hypergblock = block_except(hypergmat, [7],0)
#    print_masses(tpblock, hypergblock)


    

tpmatrix(16, 28, "Classes IV & V", ["Separate","IV Prod. in IV+V","V Prod. in IV+V","Either Prod."], ["Class V Vocab Size","Class IV Vocab Size"])
tpmatrix(52, 44, "Classes III & IV+V", ["Separate","III Prod. in III+IV+V","IV+V Prod. in III+IV+V","Either Prod."], ["Class IV+V Vocab Size","Class III Vocab Size"])
#tpmatrix(28, 29, "Class VI & VII vs VI+VII", ["V & VI","V -> V+VI","VI -> V+VI","V or VI -> V+VI"])
tpmatrix(44, 29, "Classes IV+V & VI", ["Separate","IV+V Prod. in IV+V+VI","VI Prod. in IV+V+VI","Either Prod."], ["Class VI Vocab Size","Class IV+V Vocab Size"])
