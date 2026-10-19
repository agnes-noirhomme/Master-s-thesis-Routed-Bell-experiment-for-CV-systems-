import numpy as np
from math import sqrt, log2, log, pi, cos, sin,acos,asin
from scipy.optimize import linprog
import itertools
from scipy.stats import multivariate_normal
import math
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import basinhopping
from scipy.optimize import shgo
from scipy.optimize import dual_annealing
from itertools import permutations

#number of outcomes 
k_A = 2
k_B = 2
#number of measurements 
n_A =3
n_B =2
def nosignalling(p,precision=1e-15): #check if the probabilities are non signalling
    k_A,k_B,n_A,n_B=p.shape # creates an "array" of the dimension of 2 outcomes and n_A,n_B measurements 
    #no signalling for Alice: 
    for b in range(2): #2 outcomes 
        for y in range (n_B): #number of measurements of Bob 
            for x in range (n_A): # number of measurements of Alice
                for xprime in range(n_A): # number of measurements of Alice
                    if (x != xprime):
                        test1=np.sum(p[:,b,x,y])
                        test2=np.sum(p[:,b,xprime,y])
                        if(abs((test1-test2))>precision):
                            print("problem of no signalling for Alice")
                            return False 

    #no signaling for Bob
    for a in range(2): #2 outcomes 
        for x in range (n_A): #number of measurements of Alice 
            for y in range (n_B): # number of measurements of Bib
                for yprime in range(n_B): # number of measurements of Bob
                    if (y != yprime):
                        test1=np.sum(p[a,:,x,y])
                        test2=np.sum(p[a,:,x,yprime])
                        if((test1-test2)>precision):
                            print("problem of no signalling for Bob")
                            return False 

    return True

def negativity(p): # to find negatuve probabilties 
    k_A,k_B,n_A,n_B=p.shape
    for a in range(k_A):
        for b in range(k_B):
            for x in range(n_A):
                for y in range(n_B):
                    if (p[a,b,x,y]<-1e-15):
                        print("error of negativity")
                        return False 
    return True 


def normalised(p,precision=1e-15):
    k_A,k_B,n_A,n_B=p.shape
    for a in range(k_A):
        for b in range(k_B):
            for x in range(n_A):
                for y in range(n_B):
                    if ((abs(np.sum(p[:,:,x,y])-1))>precision) :
                        print("normalisation error")
                        print(np.sum(p[:,:,x,y]))
                        return False        

    return True
def generate_local_deterministic_strategies(n_A, k_A, n_B, k_B):
    """
    Generate all local deterministic strategies (correlations) in a bipartite Bell scenario.
    
    Parameters:
    -----------
    n_A : int
        Number of inputs for Alice
    k_A : int
        Number of outputs for Alice
    n_B : int
        Number of inputs for Bob
    k_B : int
        Number of outputs for Bob
    
    Returns:
    --------
    list of dict
        Each dict contains:
        - 'alice_strategy': tuple of Alice's outputs for each input
        - 'bob_strategy': tuple of Bob's outputs for each input
        - 'correlation': 4D numpy array p(a,b|x,y) of shape (k_A, k_B, n_A, n_B)
    
    Notes:
    ------
    Total number of strategies: k_A^n_A * k_B^n_B
    """
    
    strategies = {} #empty dictionnary (dictionnary="list" but elements are accessed with a "key" that can be a number/string/etc (see: "counter:correlation")) compared to a list)
    
    # Generate all possible deterministic strategies for Alice
    # Each strategy is a function x -> a, represented as a tuple of k_A choices
    alice_strategies = list(itertools.product(range(k_A), repeat=n_A)) #creates all lambda_A=(a_1,...,a_M_A)
    
    # Generate all possible deterministic strategies for Bob
    bob_strategies = list(itertools.product(range(k_B), repeat=n_B)) #creates all lambda_B=(b_1,...,b_M_A)
    
     # For each pair of strategies, create the correlation D_(alpha,beta)(a,b|x,y)= deta_a X delta_b ie the deterministic strategy
    counter = 0
    for alice_strat in alice_strategies:
        for bob_strat in bob_strategies:
            # alice_strat[x] is Alice's output for input x
            # bob_strat[y] is Bob's output for input y
            
            # Create correlation matrix p(a,b|x,y)
            correlation = {} #empty dictionnary 
            for x in range(n_A):
                for y in range(n_B):
                    for a in range(k_A):
                        for b in range(k_B):
                            if a == alice_strat[x] and b == bob_strat[y]: #if use of deterministic strategies 
                                correlation[a, b, x, y] = 1 #delta function =1
                            else: #no use of determinisitic strategies 
                                correlation[a, b, x, y] = 0 #delta function =0 if not the strategy used 
            
            strategies.update({counter: correlation}) #add the D_(alpha,beta)(a,b|x,y)= deta_a X delta_bin for the 2^3x2^3= 64 strategies, in the dictionary :{correlation_counter,...}={correlation_1,correlation_2,...} 
            counter += 1 #count number of strategies 
    
    return strategies



# def lin_prog(points,p, random_probability_distribution, LPmethod='highs'):
#     """
#     This is the dual formulation.
#     Checks if p is inside convex hull of points
#     if objective >= 1 , then p is inside the convex hull.
#     if objective <1 , then p is outside.
#     """
#     ##### Using scipy (runs much faster) ###########
#     if type(points[0]) == dict:
#         points = np.array([list(point.values()) for point in points])

#     if type(p) == dict:    
#         p = np.array(list(p.values()))
    
#     if type(random_probability_distribution) == dict:
#         random_probability_distribution = np.array(list(random_probability_distribution.values()))
    
#     n_points = len(points)
#     n_dim = len(p)
#     c = np.r_[1,np.zeros(n_points)]
#     aux = np.r_[points.T,np.ones((1,n_points))].T
#     iden = random_probability_distribution
#     first_col=np.r_[iden-p,0]
#     A=np.r_[[first_col],aux]
#     b = np.r_[iden, 1]
#     all_bounds = [(-1,1) for j in range(n_dim)]+[(None,None)] 
#     """
#     To be precise, there are no bounds at all on the dual variables. However, having the bound (-1,1) on the elements of the witness helps improve numerical stability. Note that the last element is the value of the witness on the point p, hence no bound on it. The solution is then checked with the primal solution to make sure that these bounds did not alter the solution. 
#     """

#     lp = linprog(b, A_ub=-A, b_ub=-c,  method = LPmethod , bounds = all_bounds, options= {'presolve':True, 'disp':False} )

#     objective_value = lp.fun
#     witness = -lp.x[:-1]
#     witness_dictionary = convert_vector_to_dictionary(witness)
#     maximum_value_of_witness_for_elements_in_points = lp.x[-1]
#     value_of_witness_for_p = 1-objective_value+lp.x[-1]
#     # check_instability = max(abs(max(lp.x)),abs(min(lp.x)))
#     # print("max value encoutered in dual LP",check_instability)
#     #print("successfull ran LP")
#     return  objective_value, witness_dictionary, maximum_value_of_witness_for_elements_in_points,value_of_witness_for_p



def in_hull(points, q):
    ##if points/q is a list/collection of dictionnary -> conversion to array/matrix
    # example: points = [{"x": 1, "y": 2},{"x": 3, "y": 4}] , then points[0] gives {"x": 1, "y": 2}=ONE dictionnary, points[1]={"x": 3, "y": 4}= ONE dictionnary etc ie it is a list of dictionnaries 
    # -> this is then converted into an array: {"x": 1, "y": 2} -> [1,2]
    if type(points[0]) == dict: 

        points = np.array([list(points[j].values()) for j in range(len(points))])

    if type(q) == dict:    #same here 
        q = np.array(list(q.values())) 
    #points and q are in an "array" now: 
    points=np.array(points) 
    q=np.array(q)

    n_points = len(points) #number of elements in "points" ie deterministic points
    n_dim = len(q) #dimension of q points (all probabilities)


    #variable to max/minimise in the linear program
    #-> example: min -x_0+4x_1 (subject to things) gives c=[-1,4], here we want to maximize nu ie minimize -nu -> c=[-1,0,...] (see linearprog in python)
    c = np.r_[-1,np.zeros(n_points)]  #np.r ="add element successively in the array/matrix"
    

    aux = np.r_[points.T,np.ones((1,n_points))].T
    iden=iden = np.sum(points,axis=0)/(len(points))

    first_col=np.r_[iden-q,0]# first column of the matrix is 1-P_q and the rest is 0
    A=np.r_[[first_col],aux].T 
    b = np.r_[iden, 1] #b column: contains the identity and the last element (from sum_j p_j=1) is 1

    lp = linprog(c, A_eq=A, b_eq=b , method ='highs', options= {'presolve':True, 'disp':False} )
    
    # check_instability = max(abs(max(lp.x)),abs(min(lp.x)))
    # print("max value encoutered in primal LP",check_instability)
    return -lp.fun



#definition of my probabilities 
def probxx00(theta1,theta3,z):#P(a=b=1|XX)
    fonction=(math.erf(z)**2)-(1/(np.sqrt(np.pi)))*math.exp(-(z**2))*math.erf(z)*(z+2*(z**3))+2*math.cos(2*(theta1-theta3))*math.exp(-2*(z**2))*(z**2)*(1/(np.pi))
    return fonction
def probxx10(z): #P(a=1|X)-> ind on angles
    fonction=math.erf(z)-(1/(2*np.sqrt(np.pi)))*math.exp(-(z**2))*(2*(z**3)+z)
    return fonction

def q(z,theta1,theta2,theta3,theta4): #creates the probabilities in the set Q (that is either inside or outside the set of local model)
    q = np.zeros((2, 2,3,2))

    #00: X_theta1-X_theta3: 
    #definitions for X-X measurement using PDF and CDF 
    probaxx00=probxx00(theta1,theta3,z)
    probaxx200=probxx10(z)
    q[1,1,0,0]=probaxx00
    q[1,0,0,0]=probaxx200-q[1,1,0,0]
    q[0,1,0,0]=q[1,0,0,0]
    q[0,0,0,0]=1-q[1,0,0,0]-q[0,1,0,0]-q[1,1,0,0]

    probaxx201=probxx10(z)
    probaxx01=probxx00(theta1,theta4,z)
    q[1,1,0,1]=probaxx01
    q[1,0,0,1]=probaxx200-q[1,1,0,1]
    q[0,1,0,1]=q[1,0,0,1]
    q[0,0,0,1]=1-q[1,0,0,1]-q[0,1,0,1]-q[1,1,0,1]

    probaxx10=probxx00(theta2,theta3,z)
    q[1,1,1,0]=probaxx10
    q[1,0,1,0]=probaxx200-q[1,1,1,0]
    q[0,1,1,0]=q[1,0,1,0]
    q[0,0,1,0]=1-q[1,0,1,0]-q[0,1,1,0]-q[1,1,1,0]

    probaxx11=probxx00(theta2,theta4,z)
    q[1,1,1,1]=probaxx11
    q[1,0,1,1]=probaxx200-q[1,1,1,1]
    q[0,1,1,1]=q[1,0,1,1]
    q[0,0,1,1]=1-q[1,0,1,1]-q[0,1,1,1]-q[1,1,1,1]




    #N-X_theta3-
    q[1,1,2,0] =(1/2)*math.erf(z)
    q[1,0,2,0] =(1/2)-q[1,1,2,0]
    q[0,1,2,0] =probaxx200-q[1,1,2,0]
    q[0,0,2,0] = 1 - q[1,1,2,0] - q[1,0,2,0] - q[0,1,2,0]

#    #N-X_theta4->
    q[1,1,2,1] =(1/2)*math.erf(z)
    q[1,0,2,1] =(1/2)-q[1,1,2,1]
    q[0,1,2,1] =probaxx200-q[1,1,2,1]
    q[0,0,2,1] = 1 - q[1,1,2,1] - q[1,0,2,1] - q[0,1,2,1]



    return q


def q_array_to_dict(q_array): #translate an "array" to a dictionnary -> in_hull use two dictionnaries q (our probabilities) and points 
    q_dict = {} #empty dictionnary 
    for x in range(n_A):
        for y in range(n_B):
            for a in range(k_A):
                for b in range(k_B):
                    q_dict[a, b, x, y] = q_array[a, b, x, y] #dicitionnary contains elements of the array
    return q_dict



bounds=[(0,2*np.pi),(0,2*np.pi,)]

points = generate_local_deterministic_strategies(n_A, k_A, n_B, k_B) # generate deterministic local points
theta1=0.0
z=0.8332826383058412
theta2=1.3016915440568797

""" def objective(x):
        theta3,theta4=x
        probabilities=q(z,theta1,theta2,theta3,theta4) # probabitlies into dictionnaire to use in_hull function to find nu value 
        probadictionnaire=q_array_to_dict(probabilities)
        nu=in_hull(points,probadictionnaire)
        return abs(1-nu) #return nu smallest to nu=1 ie at boundary of locality

seeds=range(30) # 30 seeds 
bestvalue=None
bestnu=-1
bestparam=None
for seed in seeds:
    result = differential_evolution(objective,bounds,seed=seed,maxiter=500,popsize=15,tol=1e-8)
    theta3,theta4 = result.x # keep  parameters 
    probabilities = q(z, theta1, theta2,theta3,theta4)
    probadictionnaire = q_array_to_dict(probabilities)
    nu = in_hull(points, probadictionnaire) #compute nu for each seed of the optimiser 
    if(nu>bestnu):
        print(nu,theta3,theta4)
        bestnu=nu
        bestparam=result.x

print(bestnu)
print(bestparam)
theta3=bestparam[0]
theta4=bestparam[1] """
theta3=5.952746609925483
theta4=1.283953524560014
probab=q(z,theta1,theta2,theta3,theta4) # probabitlies into dictionnaire to use in_hull function to find nu value 
probadict=q_array_to_dict(probab)
nu=in_hull(points,probadict)
print(nu) 
print(nosignalling(probab,1e-15))
print(normalised(probab,1e-15))
print(negativity(probab))
np.save("/Users/a.blkmn/mémoire/13:4/2002/normal/certifyinglongrange/xxn-xxn-xx/long.npy", probab)

#1.0 5.952746609925483 1.283953524560014 -> OK here 
#1.0000251053033218 2.808188855465241 4.4432841801240475
#1.0000251053033218
#[2.80818886 4.44328418]
#1.0000251053033218
##True
#True
#True







