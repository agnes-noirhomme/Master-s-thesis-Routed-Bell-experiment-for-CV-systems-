import numpy as np
from math import sqrt, log2, log, pi, cos, sin,acos,asin
from scipy.optimize import linprog
import itertools
from scipy.stats import multivariate_normal
import math
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from itertools import permutations
#number of outcomes 
k_A = 2 
k_B = 2
#number of measurements 
n_A =3
n_B = 3


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
                    if (p[a,b,x,y]<0):
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

#def lin_prog(points,p, random_probability_distribution, LPmethod='highs'):
#     """
#     This is the dual formulation.
#     Checks if p is inside convex hull of points
#     if objective >= 1 , then p is inside the convex hull.
#     if objective <1 , then p is outside.
#     """
#     ##### Using scipy (runs much faster) ###########
#if type(points[0]) == dict:
    #points = np.array([list(point.values()) for point in points])

#if type(p) == dict:    
   # p = np.array(list(p.values()))
    
#if type(random_probability_distribution) == dict:
 #   random_probability_distribution = np.array(list(random_probability_distribution.values()))
    
#n_points = len(points)
#n_dim = len(p)
#c = np.r_[1,np.zeros(n_points)]
#aux = np.r_[points.T,np.ones((1,n_points))].T
#iden = random_probability_distribution
#first_col=np.r_[iden-p,0]
#A=np.r_[[first_col],aux]
#b = np.r_[iden, 1]
#all_bounds = [(-1,1) for j in range(n_dim)]+[(None,None)] 
#     """
#     To be precise, there are no bounds at all on the dual variables. However, having the bound (-1,1) on the elements of the witness helps improve numerical stability. Note that the last element is the value of the witness on the point p, hence no bound on it. The solution is then checked with the primal solution to make sure that these bounds did not alter the solution. 
#     """

#lp = linprog(b, A_ub=-A, b_ub=-c,  method = LPmethod , bounds = all_bounds, options= {'presolve':True, 'disp':False} )

#objective_value = lp.fun
#witness = -lp.x[:-1]
#witness_dictionary = convert_vector_to_dictionary(witness)
#maximum_value_of_witness_for_elements_in_points = lp.x[-1]
#value_of_witness_for_p = 1-objective_value+lp.x[-1]
#     # check_instability = max(abs(max(lp.x)),abs(min(lp.x)))
#print("max value encoutered in dual LP",check_instability)
#print("successfull ran LP")
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

    #linear program: min c^Tx subject to Ax=b etc 
    #variable to max/minimise in the linear program: min c^Tx 
    #-> example: min -x_0+4x_1 (subject to things) gives c=[-1,4], here we want to maximize nu ie minimize -nu -> c=[-1,0,...] (see linearprog in python)
    c = np.r_[-1,np.zeros(n_points)] 
    aux = np.r_[points.T,np.ones((1,n_points))].T #gives for each column (deterministic points(=D_1), 1,..., 1)^T
    iden = np.sum(points,axis=0)/(len(points)) #iden=identity of all of deterministic points -> iden=(1/number of det points)*sum det points 
    first_col=np.r_[iden-q,0]# first column of the matrix is 1-P_q and the rest is 0
    A=np.r_[[first_col],aux].T #add the first column and the other columns with deterministic points in the whole A matrix
    b = np.r_[iden, 1] #b column: contains the identity and the last element (from sum_j p_j=1) is 1


    #function linprog in python 
    lp = linprog(c, A_eq=A, b_eq=b , method ='highs', options= {'presolve':True, 'disp':False} )
    # check_instability = max(abs(max(lp.x)),abs(min(lp.x)))
    # print("max value encoutered in primal LP",check_instability)
    return -lp.fun





def semidefpos(matrice, tol=1e-15):
    valeurs_propres, _ = np.linalg.eigh(matrice)
    return np.all(valeurs_propres >= -tol) #valeur propre positive a une precision pres de 10^-16 

#definition of my probabilities 
def probxx(r,theta1,theta3):
    mu = [0, 0]
    rho00 = np.tanh(2*r)*cos(theta1+theta3)
    sigma = [np.sqrt(np.cosh(2*r)), np.sqrt(np.cosh(2*r))]
    cov = [[sigma[0]**2, rho00*sigma[0]*sigma[1]],[rho00*sigma[0]*sigma[1], sigma[1]**2]  ]
    if not semidefpos(cov, tol=1e-15):   #si pas definie semi positive 
        raise ValueError("La matrice de covariance n'est pas semi-définie positive")
    return multivariate_normal(mean=mu, cov=cov, allow_singular=True) #definie semi positive a notre precision 




def q(z,r,theta1,theta2,theta3,theta4): #creates the probabilities in the set Q (that is either inside or outside the set of local model)
    q = np.zeros((2, 2,3,3))

    #00: X_theta1-X_theta3: 
    #definitions for X-X measurement using PDF and CDF 
    probaxx00=probxx(r,theta1,theta3)
    q[1,1,0,0]=probaxx00.cdf([z, z])- probaxx00.cdf([-z, z])- probaxx00.cdf([z, -z])+ probaxx00.cdf([-z, -z])
    q[1,0,0,0]=math.erf(z/(np.sqrt(2*np.cosh(2*r))))-q[1,1,0,0]
    q[0,1,0,0]=q[1,0,0,0]
    q[0,0,0,0]=1-q[1,0,0,0]-q[0,1,0,0]-q[1,1,0,0]
   
    probaxx01=probxx(r,theta1,theta4)
    q[1,1,0,1]=probaxx01.cdf([z, z])- probaxx01.cdf([-z, z])- probaxx01.cdf([z, -z])+ probaxx01.cdf([-z, -z])
    q[1,0,0,1]=math.erf(z/(np.sqrt(2*np.cosh(2*r))))-q[1,1,0,1]
    q[0,1,0,1]=q[1,0,0,1]
    q[0,0,0,1]=1-q[1,0,0,1]-q[0,1,0,1]-q[1,1,0,1]

    probaxx10=probxx(r,theta2,theta3)
    q[1,1,1,0]=probaxx10.cdf([z, z])- probaxx10.cdf([-z, z])- probaxx10.cdf([z, -z])+ probaxx10.cdf([-z, -z])
    q[1,0,1,0]=math.erf(z/(np.sqrt(2*np.cosh(2*r))))-q[1,1,1,0]
    q[0,1,1,0]=q[1,0,1,0]
    q[0,0,1,0]=1-q[1,0,1,0]-q[0,1,1,0]-q[1,1,1,0]

    probaxx11=probxx(r,theta2,theta4)
    q[1,1,1,1]=probaxx11.cdf([z, z])- probaxx11.cdf([-z, z])- probaxx11.cdf([z, -z])+ probaxx11.cdf([-z, -z])
    q[1,0,1,1]=math.erf(z/(np.sqrt(2*np.cosh(2*r))))-q[1,1,1,1]
    q[0,1,1,1]=q[1,0,1,1]
    q[0,0,1,1]=1-q[1,0,1,1]-q[0,1,1,1]-q[1,1,1,1]



    #N-X_theta3
    q[0,0,2,0] = math.erfc(z/(np.sqrt(2)))*(1/(np.cosh(r)**2))
    q[1,0,2,0] = math.erfc(z/(np.sqrt(2*np.cosh(2*r))))-q[0,0,2,0]
    q[0,1,2,0] =(1/np.cosh(r)**2)-q[0,0,2,0]
    q[1,1,2,0] = 1-q[0,0,2,0]-q[1,0,2,0]-q[0,1,2,0]

    #N-X_theta4

    q[0,0,2,1] =  math.erfc(z/(np.sqrt(2)))*(1/(np.cosh(r)**2))
    q[1,0,2,1] = math.erfc(z/(np.sqrt(2*np.cosh(2*r))))-q[0,0,2,1]
    q[0,1,2,1] =(1/np.cosh(r)**2)-q[0,0,2,1]
    q[1,1,2,1] = 1-q[0,0,2,1]-q[1,0,2,1]-q[0,1,2,1]


    #independent on angle measurement 
    # X_theta1-N
    q[0,0,0,2] = math.erfc(z/(np.sqrt(2)))*(1/(np.cosh(r)**2))
    q[1,0,0,2] =(1/np.cosh(r)**2)-q[0,0,0,2]
    q[0,1,0,2] =math.erfc(z/(np.sqrt(2*np.cosh(2*r))))-q[0,0,0,2]
    q[1,1,0,2] = 1- q[0,1,0,2]-q[1,0,0,2]-q[0,0,0,2]
    
    #X_theta2-N
    q[0,0,1,2] = math.erfc(z/(np.sqrt(2)))*(1/(np.cosh(r)**2))
    q[1,0,1,2] =(1/np.cosh(r)**2)-q[0,0,1,2]
    q[0,1,1,2] =math.erfc(z/(np.sqrt(2*np.cosh(2*r))))-q[0,0,1,2]
    q[1,1,1,2] = 1- q[0,1,1,2]-q[1,0,1,2]-q[0,0,1,2]


    ##N-N 
    q[0,0,2,2]=(1/np.cosh(r)**2)
    q[1,0,2,2]=0
    q[0,1,2,2]=0
    q[1,1,2,2]=1- q[0,1,2,2]-q[1,0,2,2]-q[0,0,2,2]


    return q


def I3322(q_test,x1,x2,x3,y1,y2,y3):#definition of I3322
    valeur1=q_test[1,1,x1,y1]+q_test[1,1,x1,y2] +q_test[1,1,x1,y3]+q_test[1,1,x2,y1]+q_test[1,1,x2,y2]-q_test[1,1,x2,y3]+q_test[1,1,x3,y1]-q_test[1,1,x3,y2]
    P_A1=q_test[1,0,x1,y1]+q_test[1,1,x1,y1]
    P_B1=q_test[0,1,x1,y1]+q_test[1,1,x1,y1]
    P_B2=q_test[0,1,x1,y2]+q_test[1,1,x1,y2]
    valeur2=P_A1+2*P_B1+P_B2
    return valeur1- valeur2

def allI3322(q_test,x1,x2,x3,y1,y2,y3):
    best_valeur=-float("inf")
    for p in permutations([x1, x2, x3]):
        for q in permutations([y1, y2, y3]):
            valeur=I3322(q_test,p[0],p[1],p[2],q[0],q[1],q[2])
            if valeur>best_valeur:
                best_valeur=valeur
    return best_valeur


def q_array_to_dict(q_array): #translate an "array" to a dictionnary -> in_hull use two dictionnaries q (our probabilities) and points 
    q_dict = {} #empty dictionnary 
    for x in range(3):
        for y in range(3):
            for a in range(2):
                for b in range(2):
                    q_dict[a, b, x, y] = q_array[a, b, x, y] #dicitionnary contains elements of the array
    return q_dict


theta1=0.0

bounds = [
    (0.1, 10),   #z         
    (0.1,10),   #r
    (0,2*np.pi),  #theta2      
    (0,2*np.pi),    #theta3  
    (0,2*np.pi),   #theta4
]
#OPTIMISATION OF NU 

points = generate_local_deterministic_strategies(n_A, k_A, n_B, k_B) # generate deterministic local points
z=1.2137626724530066
r=1.1804768563678283
theta1=0.0
theta2=2.3725503167045034
theta3=5.8255180261002986
theta4=4.7123888313345113
theta1=0.0
probabilities=q(z,r,theta1,theta2,theta3,theta4) # probabitlies into dictionnaire to use in_hull function to find nu value 
probadictionnaire=q_array_to_dict(probabilities)
nu=in_hull(points,probadictionnaire)
print(nu)
print(nosignalling(q(z,r,theta1,theta2,theta3,theta4)))
#test normalisation
print(normalised(q(z,r,theta1,theta2,theta3,theta4)))
#test negative probabilities
print(negativity(q(z,r,theta1,theta2,theta3,theta4)))



np.save("/Users/a.blkmn/mémoire/13:4/tmsv/normal/certifying lon range/xxn-xxn-xx/short.npy",probabilities)

# 0.9744067826671602
# True
# True
# True






