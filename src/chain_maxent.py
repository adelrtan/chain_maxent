# Import libraries
import re
import numpy as np
import pandas as pd
from scipy import optimize
import warnings
warnings.filterwarnings("ignore")


# Read UR-SR features
def get_idLs_XCons(file):    
    # Read .csv file to DataFrame
    data_cons = pd.read_csv(file)
    
    # Get list of constraints
    cons_l = data_cons.columns[3:]
    
    # Convert to dict. 
    # key: triples. value: list of relevant transitions for each triple
    cons_d = {tuple(data_cons.iloc[i, 0:3]): list(data_cons.iloc[i, 3:]) for i in range(len(data_cons))}
    
    # Get design matrix 1, X_cons
    # ordered by triples_id_l
    triples_id_l = []
    X_cons = []
    for i in cons_d:
        triples_id_l.append(i)
        X_cons.append(cons_d[i]) 
    X_cons = np.array(X_cons)
    
    return triples_id_l, cons_l, X_cons


# Read WORD-UR features
def get_compLs_XTrans(file, triples_id_l):    
    # Read .csv file to DataFrame
    data_trans = pd.read_csv(file)
    
    # Get list of components
    comp_l = data_trans.columns[2:]
    
    # Convert to dict. 
    # key: triples. value: list of relevant transitions for each triple
    trans_d = {tuple(data_trans.iloc[i, 0:2]): list(data_trans.iloc[i, 2:]) for i in range(len(data_trans))}
         
    # Get compact design matrix 2, X_trans
    # ordered by doubles_id_l
    doubles_id_l = []
    X_trans = []
    for i in trans_d:
        doubles_id_l.append(i)
        X_trans.append(trans_d[i]) 
    X_trans = np.array(X_trans)
    
    # Get full design matrix 3, X_trans_full
    # ordered by triples_id_l
    X_trans_full = [trans_d[i,j] for i,j,k in triples_id_l]    
    X_trans_full = np.array(X_trans_full)
    
    return comp_l, doubles_id_l, X_trans, X_trans_full 


# Read WORD-SR frequencies
def get_wdSRCount_d(file):    
    # Read .csv file to DataFrame
    data_counts = pd.read_csv(file)
    
    # Convert to dict 'af_wdSR_d' 
    af_wdSR_d = {tuple(data_counts.iloc[i, 0:2]): data_counts.iloc[i, 2] for i in range(len(data_counts))}
    
    return af_wdSR_d


# Model: WORD-UR submodule
def get_p_wdUR(X_trans, theta):  
    # Component params
    num_comp = X_trans.shape[1]
    theta_trans = theta[:num_comp]
    
    # Get e^((wd,UR)-score) for each (wd,UR)-pair
    wdUR_score = np.exp(np.dot(X_trans, theta_trans))
    
    # Get p(wd,UR) for each triple
    partFunc = np.sum(wdUR_score)
    p_wdUR_arr = wdUR_score / partFunc
    
    return p_wdUR_arr


# Model: UR-SR submodule
def cdnProb_SR_gUR(triples_id_l, X_cons, theta):
    ### 0 eHarmony ###
    # Get e^harmony for each (UR,SR) pair
    # eHarmony is (m,)-vector.
    num_constr = X_cons.shape[1]
    theta_w = np.array(theta)[-num_constr:]
    eHarmony = np.exp(-1 * np.dot(X_cons, theta_w))
        
    ### 1 Partition Function ###
    # Get partition function for each UR
    partFunc = []
    for i in triples_id_l:
        # Initialize the accumulator
        accumulator = 0
        for j, k in enumerate(triples_id_l):
            # If the (wd,UR) match
            if (i[0], i[1]) == (k[0], k[1]):
                # Add its eHarmony to the accumulator
                accumulator += eHarmony[j]
        # Store partition function for that particular UR to partFunc_d
        partFunc.append(accumulator)
    partFunc = np.array(partFunc)
                    
    ### 2 Conditional Probability ###
    # Get conditional probability of SR given UR -- cProb_SR_gUR
    cProb_SR_gUR_arr = eHarmony / partFunc
                
    return cProb_SR_gUR_arr


# Model: Full model
def jtProb_wdURSR(triples_id_l, doubles_id_l, X_cons, X_trans, theta):
    # Get p(SR|wd,UR)
    cProb_SR_arr = cdnProb_SR_gUR(triples_id_l, X_cons, theta)

    # Get p(wd,UR)
    p_wdUR_arr = get_p_wdUR(X_trans, theta)
    p_wdUR_d = {doubles_id_l[i]: p_wdUR_arr[i] for i in range(len(doubles_id_l))}
    p_wdUR_full_arr = [p_wdUR_d[i,j] for i,j,k in triples_id_l]
    
    # Get p(wd,UR,SR)
    jtProb_wdURSR_arr = cProb_SR_arr * p_wdUR_full_arr
    
    return jtProb_wdURSR_arr


# Expectation
def get_expectation(triples_id_l, doubles_id_l, X_cons, X_trans, theta,
                    aFreq_wdSR_d):
    # Joint probabilites of (wd,UR,SR)-triples
    jProb_triples = jtProb_wdURSR(triples_id_l, doubles_id_l, X_cons, X_trans, theta)
    
    # Get new expectations
    expt_arr = []
    for i, j in enumerate(triples_id_l):
        accumulator = 0
        for k, l in enumerate(triples_id_l):
            # If (wd,SR) match
            if (j[0], j[2]) == (l[0], l[2]):
                # Add joint prob to denominator accumulator
                accumulator += jProb_triples[k]
        # Calculate cdnProb(UR|wd,SR)
        p_ur_gwdSR = jProb_triples[i] / accumulator
        # Calculate new expectation: cdnProb(UR|wd,SR) * absFreq(wd,SR)
        new_expectation = p_ur_gwdSR * aFreq_wdSR_d[(j[0], j[2])]
        expt_arr.append(new_expectation)
    
    expt_arr = np.array(expt_arr)
    
    return expt_arr


# Regularized loss function
def obj_fn(triples_id_l, doubles_id_l, X_cons, X_trans_full, X_trans, theta,
           expt_arr, lamb):
    # Log likelihood
    log_likelihood = np.dot(expt_arr,
                     np.log(jtProb_wdURSR(triples_id_l, doubles_id_l,
                                          X_cons, X_trans, theta)))
    
    # Regulatization term for all params
    reg_term = (-1/2) * np.dot(lamb, (theta ** 2))
    
    # Old regularization: same lambda for all params
    # reg_term = (-lamb)/2 * np.sum(theta ** 2)
    
    # Regularization term only for constraint params
    # To use when morpheme-transition params are normalized locally
    # num_constr = X_cons.shape[1]
    # reg_term = (-lamb)/2 * np.sum(theta[-num_constr:] ** 2)

    # Objective function
    obj_fn = log_likelihood + reg_term

    return obj_fn


# Gradient for WORD-UR params
def gradient_trans(X_trans_full, X_trans, theta, expt_arr):
    # First term
    first_term = np.dot(expt_arr, X_trans_full)
    
    # Second term
    wdUR_prob = get_p_wdUR(X_trans, theta)
    second_term = np.dot(expt_arr, \
                         (np.broadcast_to(np.dot(wdUR_prob, X_trans), \
                                          X_trans_full.shape)))
    
    # Gradient_trans
    grad_trans = first_term - second_term
    grad_trans = np.squeeze(grad_trans)
    
    return grad_trans


# Gradient for UR-SR params
def gradient_cons(triples_id_l, X_cons, theta, expt_arr):
    # First term
    first_term = -1 * np.dot(expt_arr, X_cons)
    
    # Second term
    # Make dictionary "index_bucket" with unique (wd,UR)-pairs as keys
    # and for values: a list of the indices of the triples
    # that have that (wd,UR)-pair
    index_bucket = {}
    for i, triple in enumerate(triples_id_l):
        if (triple[0], triple[1]) not in index_bucket:
            index_bucket[(triple[0], triple[1])] = [i]
        else:
            index_bucket[(triple[0], triple[1])].append(i)
    
    cProb_full = cdnProb_SR_gUR(triples_id_l, X_cons, theta)
    
    value_bucket = {}
    for i in index_bucket:
        myIndices = index_bucket[i]
        cProb_temp = [cProb_full[j] for j in myIndices]
        X_temp = np.array([X_cons[j] for j in myIndices])
        value_bucket[i] = np.dot(X_temp.T, cProb_temp)
    
    value_arr = []
    for triple in triples_id_l:
        for pair in value_bucket:
            if (triple[0], triple[1]) == pair:
                value_arr.append(value_bucket[pair])
    value_arr = np.array(value_arr)
    
    second_term = -1 * np.dot(expt_arr, value_arr)
    
    # Gradient_cons
    grad_cons = first_term - second_term
    grad_cons = np.array(grad_cons)
    
    return grad_cons


# Regularized gradients for all params
def gradient(triples_id_l, doubles_id_l, 
             X_cons, X_trans_full, X_trans, theta, 
             expt_arr, lamb):
    # Make non-regularized gradients
    grad_trans = gradient_trans(X_trans_full, X_trans, theta, expt_arr)
    grad_cons = gradient_cons(triples_id_l, X_cons, theta, expt_arr)    
    
    # Non-regularized gradients combined
    grad_noReg = np.concatenate((grad_trans, grad_cons))
    
    # Regularization term
    reg_term = -lamb * theta
    
    # Regularization term when morpheme-transition params are normalized locally
    # num_comp = X_trans.shape[1]
    # reg_term[:num_comp] = 0
    
    # Full gradients
    grad = grad_noReg + reg_term
    
    return grad


# Bounds for params
def bounds(X_cons, X_trans):
    # Get number of components & number of constraints
    num_comp = X_trans.shape[1]
    num_constr = X_cons.shape[1]
    
    # Make bounds for component params
    comp_bounds = [(np.NINF, np.inf) for i in range(num_comp)]
    # Make bounds for constraint params
    constr_bounds = [(0., np.inf) for i in range(num_constr)]
    
    # Concatenate both bounds
    bounds = np.array(comp_bounds + constr_bounds)
    
    return bounds


# Maximization: The "M-step" of expectation-maximization
def update_theta(triples_id_l, doubles_id_l, comp_l,
                 X_cons, X_trans_full, X_trans,
                 theta, expt_arr, lamb):
    # 0. Objective function: log-likelihood
    def f(theta, triples_id_l, doubles_id_l, X_cons, X_trans_full, X_trans, expt_arr, lamb):
        # Change sign because this method is for minimization
        objFn = -1 * obj_fn(triples_id_l, doubles_id_l, X_cons, X_trans_full, X_trans, theta, expt_arr, lamb)
        return objFn
    
    # 1. First partial derivative of objective function
    def jac(theta, triples_id_l, doubles_id_l, X_cons, X_trans_full, X_trans, expt_arr, lamb):
        # Change sign because this method is for minimization
        grad = -1 * gradient(triples_id_l, doubles_id_l, X_cons, X_trans_full, X_trans, theta, expt_arr, lamb)
        return grad
    
    # 2. Bounds
    bnds = bounds(X_cons, X_trans)
    
    # 3. Result
    res = optimize.minimize(fun=f,
                            x0=theta,
                            args=(triples_id_l,
                                  doubles_id_l, 
                                  X_cons,
                                  X_trans_full,
                                  X_trans,
                                  expt_arr,
                                  lamb),
                            method='L-BFGS-B',
                            jac=jac,
                            bounds=bnds)

    
    return res


# Train one model: Run expectation-maximization
def run_em(triples_id_l, doubles_id_l, comp_l,
           X_cons, X_trans_full, X_trans,
           theta, aFreq_wdSR_d,
           lamb, num_iter, threshold,
           verbose):
    # log_likelihood at starting weights
    expt_arr = get_expectation(triples_id_l, doubles_id_l,
                               X_cons, X_trans,
                               theta, aFreq_wdSR_d)

    old_logLike = -1 * obj_fn(triples_id_l, doubles_id_l, X_cons, X_trans_full, X_trans,
                              theta, expt_arr, lamb=lamb)
    
    # Iterate over num_iter
    for i in range(num_iter):
        
        # Get Expectation
        expt_arr = get_expectation(triples_id_l, doubles_id_l,
                                   X_cons, X_trans,
                                   theta, aFreq_wdSR_d)

        # Update parameters to "maximize" log likelihood
        # Update theta
        result = update_theta(triples_id_l, doubles_id_l, comp_l,
                              X_cons, X_trans_full, X_trans,
                              theta, expt_arr, lamb)
        theta = result.x
    
        # At regular intervals perform visual check that log-likelihood is decreasing
        if (i % 1 == 0):
            # Print log-likelihood & gradients.
            new_logLike = result.fun
            if verbose == True:
                print(f"Log-likelihood after iteration {i+1} is {new_logLike:.5f}")
                print(f"Gradients are {result.jac}. Weights are {theta}")            
            
            # Check change in log-likelihood
            diff_logLike = old_logLike - new_logLike
            # If difference in log-likelihod smaller than threshold
            if abs(diff_logLike) < threshold:
                
                # Print exit conditions & exit
                if verbose == True:
                    print("")
                    print(f"Exited expectation-maximization after iteration {i+1}")
                    print(f"Log-likelihood after iteration {i+1} is {new_logLike:.5f}")
                    print(f"Gradients are {result.jac}.")
                    print(f"Weights are {theta}")
                break
                
            else:
                # Update the old log-likelihood
                 old_logLike = new_logLike
    
    # Return theta, p(wd,UR), new_logLike
    return theta, new_logLike


# Train multiple randomly-initialized models
def run_randomInit(triples_id_l, doubles_id_l, comp_l,
                   X_cons, X_trans_full, X_trans,
                   aFreq_wdSR_d,
                   num_runs,
                   lamb, num_iter, threshold,
                   verbose, save_runs,
                   floor): 
    
    # Initialize best_logLike
    best_logLike = np.inf
    
    if save_runs == True:
        best_initTheta = []
        best_learnedTheta = []
        best_logLike = []
    
    while len(best_logLike) < num_runs :        
        # Initialize params
        comp_params = np.random.uniform(low=0.0, high=100.0, size=X_trans.shape[1])
        constr_params = np.random.uniform(low=0.1, high=5.0, size=X_cons.shape[1])
        init_theta = np.concatenate((comp_params, constr_params))
        
        # Run EM
        learned_theta, new_logLike = run_em(triples_id_l, doubles_id_l, comp_l,
                                            X_cons, X_trans_full, X_trans,
                                            init_theta, aFreq_wdSR_d,
                                            lamb, num_iter, threshold,
                                            verbose)
        
        # If saving all runs close to floor
        if save_runs == True:
            # If likelihood is very close to floor
            if new_logLike < floor:
                # Append its initial weights, learned weights & objFn to relevant lists
                best_initTheta.append(init_theta)
                best_learnedTheta.append(learned_theta)
                best_logLike.append(new_logLike)
        
        # If only saving the ONE best run
        else:
            # If likelihood is best one thus far
            if new_logLike < best_logLike:
                # Update best initial params
                best_initTheta = init_theta
            
                # Update best learned params
                best_learnedTheta = learned_theta
            
                # Update best log-likelihood
                best_logLike = new_logLike
            
                print(f"Best objective function is {best_logLike}")
                print(f"Best starting weights are {best_initTheta}")
                print(f"Learned weights are {best_learnedTheta}")
            
    return best_initTheta, best_learnedTheta, best_logLike


# Train models
def train_models(ursr_file, wdur_file, wdsr_freq_file,
                 num_runs, lamb=None, 
                 save_runs=True, data_frame=False,
                 num_iter=1500, threshold=0.0001,
                 verbose=False, floor=np.inf,
				 save_data=True, save_data_frame=False):

    # 1. Read input files
    triples_id_l, cons_l, X_cons = get_idLs_XCons(ursr_file)
    comp_l, doubles_id_l, X_trans, X_trans_full = get_compLs_XTrans(wdur_file, triples_id_l)
    aFreq_wdSR_d = get_wdSRCount_d(wdsr_freq_file)
      
    # 2. Regularization
    # If no regularization specified
    if lamb == None:
        # Set lambda to 0 for all params
        num_params = X_cons.shape[1] + X_trans.shape[1]
        lamb = np.zeros((num_params,))
    # Else
    else:
        # Use user-specified lambda
        lamb = lamb
        
    
    # 3. Train models
    init_theta, trained_theta, logLike = run_randomInit(triples_id_l, doubles_id_l, comp_l,
                                                        X_cons, X_trans_full, X_trans,
                                                        aFreq_wdSR_d,
                                                        num_runs=num_runs,
                                                        lamb=lamb, num_iter=num_iter, threshold=threshold,
                                                        verbose=verbose, save_runs=save_runs,
                                                        floor=floor)
    
    # 4. Save data to .npy
	if save_data == True:
        np.save('trained_weights.npy', trained_theta, allow_pickle=True)
        np.save('initial_weights.npy', init_theta, allow_pickle=True)
        np.save('log_likelihood_trained.npy', logLike, allow_pickle=True)
    
	# 5. (Make DataFrame)
    if data_frame == True:
        feature_names = comp_l.to_list() + cons_l.to_list()
        trained_theta = pd.DataFrame(trained_theta, columns=feature_names)
        # 5b. (Save DataFrame to .csv)
        if save_data_frame == True:
            trained_theta.to_csv('trained_weights.csv')
    else:
        trained_theta = trained_theta
    
    return init_theta, trained_theta, logLike

    
# Predict probabilities
def predict_probabilities(ursr_file, wdur_file, trained_theta, data_frame=False, 
                          save_data=True, save_data_frame=False):
    # 1. Read input files
    triples_id_l, cons_l, X_cons = get_idLs_XCons(ursr_file)
    comp_l, doubles_id_l, X_trans, X_trans_full = get_compLs_XTrans(wdur_file, triples_id_l)
    
    # 2. Predict probabilities
    # Initialize pred_probs list
    pred_probs = []
    
    if type(trained_theta) is not list:
        trained_theta = list(trained_theta.values)
    
    for i in trained_theta:
        pred_prob = jtProb_wdURSR(triples_id_l, doubles_id_l, X_cons, X_trans, theta=i)
        pred_probs.append(pred_prob)
     
    # 3. Save data to .npy
    if save_data == True:
        np.save('predicted_probs.npy', pred_probs, allow_pickle=True)
     
    # 4. (Make DataFrame)
    if data_frame == True:
        pred_probs = pd.DataFrame(pred_probs, columns = triples_id_l)     
        # 4b. (Save DataFrame to .csv)
        if save_data_frame == True:
            pred_probs.to_csv('predicted_probs.csv')		
    else:
        pred_probs = pred_probs
    
    return pred_probs 


# A small example:
# 1. Optimize models on example training data
# Training input files
ursr_train_file = '../data/ursr_feature_matrix_train.csv'
wdur_train_file = '../data/wdur_feature_matrix_train.csv'
wdsr_freq_file = '../data/wdsr_freq_train.csv'

num_runs = 30	# Number of models to train

start_wts, trained_wts, log_like = train_models(ursr_train_file, wdur_train_file, 
                                                wdsr_freq_file,
                                                num_runs=num_runs, data_frame=True,
                                                save_data_frame=True)

print(f"The DataFrame of trained weights for {num_runs} models:\n")
print(trained_wts)


# 2. Evaluate models on test data
# Test input files	
ursr_test_file = '../data/ursr_feature_matrix_test.csv'
wdur_test_file = '../data/wdur_feature_matrix_test.csv'

predicted_probs = predict_probabilities(ursr_test_file, wdur_test_file, trained_wts, 
                                        data_frame=True, save_data_frame=True)

print(f"The DataFrame of predicted probabilities for test items ({num_runs} models):\n")
print(predicted_probs)
			