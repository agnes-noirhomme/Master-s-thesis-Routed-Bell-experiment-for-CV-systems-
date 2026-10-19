# Description: For a given target implementations at short range (pchsh) and long range (psteer), this code computes the maximum etaBL for which the target correlations with given close range efficiencies etaA and etaBS and a given amount of local noise nu admit an SRQ model at a given NPA relaxation level.
# Inputs: 
# - etaA_list = list of short range efficiencies
# - chsh = Boolean: True = CHSH implementation, False = BB84 correlations
#  - noise = Boolean: True = local noise, False = no noise
#  - nus = list of local visibilities
#  - binning = Boolean: True = Bin no click outcomes, False = Keep no-click outcomes
#  - LEVEL = NPA level
# Outputs: etaBL_list
# Import the necessary packages
import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
import math
import qutip as qtp
from sympy import Integer
from scipy.special import erf, erfc
from scipy.optimize import differential_evolution, minimize
################################################################################
##################### Parmenant functions -- do not change #####################
################################################################################


# Get observed correlations given ideal strategies pchsh (short range) and psteer (long range)

def get_p_obs(pchsh, psteer,binning):
    nZ = 2
    nX_max = max(nX, nX2)
    nY_max = max(nY, nY2)
    p_obs = np.zeros((nA, nB, nX_max, nY_max, nZ), dtype=object)

    if binning:
        for a in range(nA):
            for b in range(nB):
                for x in range(nX):
                    for y in range(nY):         # Bob_S: nY=3 measurements
                        p_obs[a,b,x,y,0] = pchsh[a,b,x,y] 
        for a in range(nA):
            for b in range(nB):
                for x in range(nX2):
                    for y2 in range(nY2):       # Bob_L: nY2=2 measurements
                        p_obs[a,b,x,y2,1] = psteer[a,b,x,y2] 
    return p_obs


def get_equalities(p_obs, binning,P):
    equalities = []
    k = 0 if binning else 1

    ###### Bob_S: short path, nY=3 measurements → NPA indices {0,1,2} ######
    for x in range(nX):
        for y in range(nY):               # y in {0,1,2} → Bs0,Bs1,Bs2
            for a in range(nA+k):
                for b in range(nB+k):
                    equalities += [P([a,b],[x,y]) - p_obs[a,b,x,y,0]]

    ###### Bob_L: long path, nY2=2 measurements → NPA indices {3,4} ######
    for x in range(nX):
        for y2 in range(nY2):             # 3 measures bob short puis apres mesurs bob long -> decalage de 3 
            for a in range(nA+k):
                for b in range(nB+k):
                    equalities += [P([a,b],[x,y2+nY]) - p_obs[a,b,x,y2,1]]

    return equalities

def objective():
    return -etaBL #we want to maximize over -nu 



# Get target correlations given local noise nu
def get_p_target(nu): 

    #implementation of the probabilities for chsh: short path or steer: long path 
    pchshq = np.zeros((nA,nB,nX,nY),dtype=object) #short path ->non local model 
    psteerq= np.zeros((nA,nB,nX2,nY2),dtype=object) #long path -> local model 
    pchsh= np.zeros((nA,nB,nX,nY),dtype=object)  #encode probabilitites with noise  
    psteer= np.zeros((nA,nB,nX2,nY2),dtype=object) #encode probabilities with noise

    #Psteer #local model in long path
    psteerq=np.load("/Users/a.blkmn/mémoire/13:4/2002/normal/certifyinglongrange/xxx-xxn/xxx-xxn-xn/long.npy")
    #PCHSH ie short path:
    pchshq=np.load("/Users/a.blkmn/mémoire/13:4/2002/normal/certifyinglongrange/xxx-xxn/xxx-xxn-xn/short.npy")

    #noise
        #pchsh 
    
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                   # pchsh[a, b, x, y] = nu * pchshq[a, b, x, y] + ((1-nu)/4)
                    pchsh[a, b, x, y] = pchshq[a, b, x, y]

    for x in range(nX2):
        for y in range(nY2):
            for a in range(nA):
                for b in range(nB):
                   # psteer[a, b, x, y] = nu * psteerq[a, b, x, y] + ((1-nu)/4)  
                    pA = np.sum(psteerq[a, :, x, 0])  #find P(a)=trace(rho_A*A_a|x) with y the choice i want (due to no signalling)
                   # #nu*proba+((1-nu)/2)*P(a|x)
                    psteer[a, b, x, y] = (nu * psteerq[a, b, x, y]+ (1 - nu) * pA * (1/2))
                    


    ####Define Marginals#### 
    [pAchsh, pAsteer]=[np.zeros((nA,nX),dtype=object), np.zeros((nA,nX2),dtype=object)]
    [pBchsh, pBsteer]=[np.zeros((nB,nY),dtype=object), np.zeros((nB,nY2),dtype=object)]
    #no signaling constraints 
    for a in range(nA):
        for x in range(nX):
            pAchsh[a,x] = np.sum(pchsh[a,:,x,0])

    for a in range(nA):
        for x2 in range(nX2):
            pAsteer[a,x2] = np.sum(psteer[a,:,x2,0])

    for b in range(nB):
        for y in range(nY):
            pBchsh[b,y] = np.sum(pchsh[:,b,0,y])
    for b in range(nB):
        for y2 in range(nY2):
            pBsteer[b,y2] = np.sum(psteer[:,b,0,y2])
    return pchsh, psteer, pAchsh, pAsteer, pBchsh, pBsteer


nA=2 # 2 OUTCOMES 
nB=2
nX=3 # 3 mesures alice 
nX2=3 # 3 mesures alice
nY=3# 3 mesures short bob
nY2=2#  mesures long bob



#########################################################
########### Only need to change these inputs ############
#########################################################

noise = True # if True, compute maximum etaBL for noise, otherwise for no noise
binning = True # if True, compute maximum etaBL for binning (keep no click as an existing outcome), otherwise for no binning

LEVEL = 3 #NPA relaxation level

####################################
####################################
####################################

#### Main code ####
 

etaBL=1
#def objective(x):
#theta3,theta4,theta5=x
P = ncp.Probability([nA,nA,nA],[nB,nB,nB,nB,nB]) #defines non commuting operators 
substitutions = P.substitutions #substitutions used in ncpol2sdpa    -> non commuting operators satisfies projectors relations now                         
[A0,A1,A2,Bs0,Bs1,Bs2,Bl0,Bl1] = P.get_all_operators() #define with name those projectors: 3 operators for Alice and Bob short and 2 for Bob long 
    #A_{a=0|x=0} et A_{a=1|x=0}=1-A_{a=0|x=0} -> we only keep A_{a=0|x=0}=A_0
    #same for A_{a=0|x=1} etc ->we only keep A_{a=0|x=1}=A_1
substitutions.update({Bl0*Bl1:Bl1*Bl0}) #Bob long path -> bob's operator in long path commute 

nu = ncp.generate_variables('nu', 1)[0] #generate variable nu for noise
sdp = ncp.SdpRelaxation(ncp.flatten([nu,P.get_all_operators()]), normalized=True, verbose=1) 


obj = -nu  #objective function

[pchsh, psteer, pAchsh, pAsteer, pBchsh, pBsteer] = get_p_target(nu)
p_obs = get_p_obs(pchsh, psteer,binning) #p_obs contains all probabilities
equalities = get_equalities(p_obs,binning,P) #constraints for moment matrix needing to have our probabilities inside
sdp.get_relaxation(level = LEVEL,
                       substitutions = substitutions,
                       objective=obj,
                       momentequalities= equalities)#,
                        #inequalities = [etaBL <=1],
                        #extramonomials=P.get_extra_monomials('AB'))
  
sdp.solve()
print(sdp.primal)



# (CVXPY) Jul 13 09:49:30 AM:   Problem status  : PRIMAL_AND_DUAL_FEASIBLE
# (CVXPY) Jul 13 09:49:30 AM:   Solution status : OPTIMAL
# (CVXPY) Jul 13 09:49:30 AM:   Primal.  obj: -1.0000001074e+00   nrm: 2e+00    Viol.  con: 1e-08    var: 0e+00    barvar: 0e+00  
# (CVXPY) Jul 13 09:49:30 AM:   Dual.    obj: -1.0000000922e+00   nrm: 6e+00    Viol.  con: 0e+00    var: 9e-09    barvar: 2e-08  
# -------------------------------------------------------------------------------
#                                     Summary                                    
# -------------------------------------------------------------------------------
# (CVXPY) Jul 13 09:49:30 AM: Problem status: optimal
# (CVXPY) Jul 13 09:49:30 AM: Optimal value: -1.000e+00
# (CVXPY) Jul 13 09:49:30 AM: Compilation took 3.837e+01 seconds
# (CVXPY) Jul 13 09:49:30 AM: Solver (including time spent in interface) took 3.525e+01 seconds
# -1.000000092182832



# (CVXPY) Jul 13 09:52:00 AM:   Problem status  : PRIMAL_AND_DUAL_FEASIBLE
# (CVXPY) Jul 13 09:52:00 AM:   Solution status : OPTIMAL
# (CVXPY) Jul 13 09:52:00 AM:   Primal.  obj: -1.0000000462e+00   nrm: 2e+00    Viol.  con: 7e-09    var: 0e+00    barvar: 0e+00  
# (CVXPY) Jul 13 09:52:00 AM:   Dual.    obj: -1.0000000401e+00   nrm: 6e+00    Viol.  con: 0e+00    var: 4e-09    barvar: 8e-09  
# -------------------------------------------------------------------------------
#                                     Summary                                    
# -------------------------------------------------------------------------------
# (CVXPY) Jul 13 09:52:00 AM: Problem status: optimal
# (CVXPY) Jul 13 09:52:00 AM: Optimal value: -1.000e+00
# (CVXPY) Jul 13 09:52:00 AM: Compilation took 3.846e+01 seconds
# (CVXPY) Jul 13 09:52:00 AM: Solver (including time spent in interface) took 3.570e+01 seconds
# -1.000000040068345