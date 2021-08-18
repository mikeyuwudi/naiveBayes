import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    # Implement your algorithm and return 
    state = {}
    N = matrix.shape[1]
    
    ############################
    # Implement your code here #
    num_spam = 0
    spam_arr, ham_arr = [], []
    for i, num in enumerate(category):
        if num == 1: 
            num_spam += 1
            spam_arr.append(i)
        else:
            ham_arr.append(i)
    p_spam = num_spam / len(category) 
    state["p_spam"] = p_spam
    
    mu_spam, mu_ham = [], []
    total_spam_word, total_ham_word = 0, 0
    for i in spam_arr:
        total_spam_word += sum(matrix[i, :])
    for i in ham_arr:
        total_ham_word += sum(matrix[i, :])
    for i in range(N):
        spam_num, ham_num = 0, 0
        for j in spam_arr:
            spam_num += matrix[j, i]
        for k in ham_arr:
            ham_num += matrix[k, i]
        mu_spam.append((spam_num + 1) / (total_spam_word + N))
        mu_ham.append((ham_num + 1) / (total_ham_word + N))
    
    state["mu_spam"] = mu_spam
    state["mu_ham"] = mu_ham
    ############################
    
    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    output = np.zeros(matrix.shape[0])
    
    ############################
    # Implement your code here #
    p_spam = state["p_spam"]
    mu_spam = state["mu_spam"]
    mu_ham = state["mu_ham"]
    
    for i in range (matrix.shape[0]):
        ln_p_spam, ln_p_ham = 0, 0
        for j in range (matrix.shape[1]):
            if matrix[i, j] != 0:
                ln_p_spam += np.log(mu_spam[j]) + np.log(p_spam) - (np.log(mu_spam[j]*p_spam + mu_ham[j]*(1 - p_spam)))
                ln_p_ham += np.log(mu_ham[j]) + np.log(1 - p_spam) - (np.log(mu_spam[j]*p_spam + mu_ham[j]*(1 - p_spam)))
        if ln_p_spam >= ln_p_ham: output[i] = 1
    ############################
    
    return output

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: {:2.4f}%'.format(100*error))
     
def indToken(state, tokenlist):
    mu_spam = state["mu_spam"]
    mu_ham = state["mu_ham"]
    ratio, token = [], []
    for i in range(len(mu_spam)):
        ratio.append(mu_spam[i] / mu_ham[i])
    for i in range(5):
        ind = np.argmax(ratio)
        token.append(tokenlist[ind])
        ratio[ind] = -1
    return token

# define this function just to return a value instead of printing message
def evaluate2(output, label):
    return (output != label).sum() * 1. / len(output)

def predictMany():
    # Load all files and train
    data50, tokenlist, category50 = readMatrix('q4_data/MATRIX.TRAIN.50')
    data100, tokenlist, category100 = readMatrix('q4_data/MATRIX.TRAIN.100')
    data200, tokenlist, category200 = readMatrix('q4_data/MATRIX.TRAIN.200')
    data400, tokenlist, category400 = readMatrix('q4_data/MATRIX.TRAIN.400')
    data800, tokenlist, category800 = readMatrix('q4_data/MATRIX.TRAIN.800')
    data1400, tokenlist, category1400 = readMatrix('q4_data/MATRIX.TRAIN.1400')
    
    state50 = nb_train(data50, category50)
    state100 = nb_train(data100, category100)
    state200 = nb_train(data200, category200)
    state400 = nb_train(data400, category400)
    state800 = nb_train(data800, category800)
    state1400 = nb_train(data1400, category1400)
    
    data = [data50, data100, data200, data400, data800, data1400]
    state_arr = [state50, state100, state200, state400, state800, state1400]
    cate_arr = [category50, category100, category200, category400, category800, category1400]
    pred_err = []
    
    for i in range(len(state_arr)):
        pred = nb_test(data[i], state_arr[i])
        err = evaluate2(pred, cate_arr[i])
        pred_err.append(err)
    
    plt.figure(1)
    plt.plot([50, 100, 200, 400, 800, 1400],pred_err,marker='o',linestyle='-',linewidth=3)
    plt.xlabel('data size')
    plt.ylabel('prediction error %')
    plt.title("Data size vs. Error")
    plt.grid(True)


def main():
    # Note1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    # Note3: The shape of the data matrix (document matrix): (number of emails) by (number of tokens)

    # Load files
    dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')

    # Train
    state = nb_train(dataMatrix_train, category_train)

    # Test and evluate
    prediction = nb_test(dataMatrix_test, state)
    evaluate(prediction, category_test)
    
    # (b) Find 5 tokens that have the highest positive value
    token = indToken(state, tokenlist)
    print("5 tokens:", token)    
    
    # (c)
    predictMany()
    print("By the results, the data size = 100 and 200 gives the best classification error.")
    
if __name__ == "__main__":
    main()
        
