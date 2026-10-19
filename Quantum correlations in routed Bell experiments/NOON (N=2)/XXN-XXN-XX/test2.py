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

def get_p_obs(etaA,etaBL, psteer, pAsteer, pBsteer,binning):
    p_obs = np.zeros((nA, nB, nX2, nY2), dtype=object)
    if binning:
        for a in range(nA):
            for b in range(nB):
                for x in range(nX2):
                    for y2 in range(nY2):       
                        p_obs[a, b, x, y2] = etaA * etaBL * psteer[a, b, x, y2] + (1 - etaA) * etaBL * pBsteer[b, y2] * (1-a) +\
                                            etaA * (1 - etaBL) * pAsteer[a, x] * (1-b) + (1 - etaA) * (1 - etaBL)*(1-a)*(1-b) 
    
    return p_obs


def get_equalities(p_obs, binning):
    equalities = []
    k = 0 if binning else 1

    ###### Bob_L: long path, nY2=2 measurements → NPA indices {3,4} ######
    for x in range(nX):
        for y2 in range(nY2):             # 3 mesures short bob puis apres mesures de bob long -> decallage de 3=nY
            for a in range(nA+k):
                for b in range(nB+k):
                    equalities += [P([a,b],[x,y2]) - p_obs[a,b,x,y2]]

    return equalities

def objective():
    return -etaBL #we want to maximize over -nu 



# Get target correlations given local noise nu
def get_p_target(nu): 

    psteerq= np.zeros((nA,nB,nX2,nY2),dtype=object) #long path -> local model 

    psteer= np.zeros((nA,nB,nX2,nY2),dtype=object) #encode probabilities with noise

    #Psteer #local model in long path
    psteerq=np.load("/Users/a.blkmn/mémoire/13:4/2002/normal/certifyinglongrange/xxn-xxn-xx/long.npy")


    for x in range(nX2):
        for y in range(nY2):
            for a in range(nA):
                for b in range(nB):
                     psteer[a, b, x, y] = nu * psteerq[a, b, x, y] + ((1-nu)/4)  
                    


    ####Define Marginals#### 
    pAsteer = np.zeros((nA,nX2), dtype=object)
    pBsteer = np.zeros((nB,nY2), dtype=object)
    #no signaling constraints 
    for a in range(nA):
        for x2 in range(nX2):
            pAsteer[a,x2] = np.sum(psteer[a,:,x2,0])
    for b in range(nB):
        for y2 in range(nY2):
            pBsteer[b,y2] = np.sum(psteer[:,b,0,y2])
    return psteer, pAsteer, pBsteer


nA=2 # 2 OUTCOMES 
nB=2
nX=3 # 3 mesures alice 
nX2=3 # mesures alice
nY2=2# 3 mesures long bob



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

nu=1.0
etaBL = ncp.generate_variables('etaBL',1)[0]
P = ncp.Probability([nA,nA,nA],[nB,nB]) #defines non commuting operators : 3 op for Alice, and 6 for Bob (3 short-3 long)
substitutions = P.substitutions #substitutions used in ncpol2sdpa    -> non commuting operators satisfies projectors relations now                         
[A0,A1,A2,Bl0,Bl1] = P.get_all_operators() #define with name those projectors: 3 operators for Alice and Bob short and 2 for Bob long 
    #A_{a=0|x=0} et A_{a=1|x=0}=1-A_{a=0|x=0} -> we only keep A_{a=0|x=0}=A_0
    #same for A_{a=0|x=1} etc ->we only keep A_{a=0|x=1}=A_1
#substitutions.update({Bl0*Bl1:Bl1*Bl0})
sdp = ncp.SdpRelaxation(ncp.flatten([etaBL,P.get_all_operators()]), normalized=True, verbose=1) 
obj = objective ()  #objective function
etaA=1
[psteer, pAsteer, pBsteer] = get_p_target(nu)
p_obs = get_p_obs(etaA, etaBL, psteer,  pAsteer, pBsteer,binning)  #p_obs contains all probabilities
equalities = get_equalities(p_obs,binning) #constraints for moment matrix needing to have our probabilities inside
sdp.get_relaxation(level = LEVEL,
                       substitutions = substitutions,
                       objective=obj,
                       momentequalities= equalities,#,
                        inequalities = [etaBL <=1])
                        #extramonomials=P.get_extra_monomials('AB'))
  
sdp.solve()
print(sdp.primal)


# (CVXPY) Jul 11 04:40:12 PM:   Problem status  : PRIMAL_AND_DUAL_FEASIBLE
# (CVXPY) Jul 11 04:40:12 PM:   Solution status : OPTIMAL
# (CVXPY) Jul 11 04:40:12 PM:   Primal.  obj: -9.9999998487e-01   nrm: 1e+00    Viol.  con: 5e-09    var: 0e+00    barvar: 0e+00  
# (CVXPY) Jul 11 04:40:12 PM:   Dual.    obj: -9.9999997840e-01   nrm: 5e+00    Viol.  con: 0e+00    var: 3e-09    barvar: 3e-09  
# -------------------------------------------------------------------------------
#                                     Summary                                    
# -------------------------------------------------------------------------------
# (CVXPY) Jul 11 04:40:12 PM: Problem status: optimal
# (CVXPY) Jul 11 04:40:12 PM: Optimal value: -1.000e+00
# (CVXPY) Jul 11 04:40:12 PM: Compilation took 3.060e-01 seconds
# (CVXPY) Jul 11 04:40:12 PM: Solver (including time spent in interface) took 2.023e-01 seconds
# -0.9999999783986157