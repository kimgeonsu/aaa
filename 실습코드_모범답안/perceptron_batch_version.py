import numpy as np 

def step_function(x): #step function 정의
    if x >= 0:
        return 1.0
    else:
        return -1.0

np.random.seed(20)
X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float64) #논리 AND 연산 수행을 위한 학습 데이터
y = np.array([-1, 1, 1, 1], dtype=np.float64) #논리 AND 연산을 위한 목표값
W = np.random.randn(len(X[0]) + 1) #페셉트론 가충지 초기화, W[0]: bias에 관한 가중치 
#W = np.zeros(len(X[0])) # 퍼셉트론 가중치 값
alpha = 0.2
epoch = 20

def perceptron_learning(X, Y, W, alpah=0.2, epoch=10): # perceptron 학습 함수
    #X0 = np.ones((4,1))
    #X = np.concatenate([X0, X], axis=1) # bias를 x[:, 0]에 할당하고 1로 입력함 
    for e in range(epoch):
        print(f"the number of epoch{e}\n")
        error_samples = []
        for j in range(len(X)):
            predict = step_function(np.dot(np.r_[1, X[j]],W)) # bias를 x[:, 0]에 할당하고 1로 입력함 
            if(Y[j] != predict):
                error_samples.append((np.r_[1,X[j]],Y[j])) # bias를 x[:, 0]에 할당하고 1로 입력함    
        
        # 틀린 sample에 관해 에러와 입력벡터의 곱의 합을 구함
        sum = np.zeros(3,dtype=np.float64) 
        for x, y in error_samples:
            sum = sum + y*x
        
        # 가중치 업데이트
        for i in range(len(W)): 
            W[i] = W[i] + alpha*sum[i]
        #W = W + alpha*sum
        print(f"변경된 가중치:[{W[0]},{W[1]}, {W[2]}]\n")
        print("++++++++++++++++++++++++++++++++\n")
    return W

def perceptron_inference(X,W):
    #X0 = np.ones((4,1),dtype=np.float64)
    #X = np.concatenate([X0, X], axis=1)
    inferences = []
    for x in X:
        inferences.append((x, step_function(np.dot(np.r_[1,x],W))))
    return inferences

l_W = perceptron_learning(X,y,W,alpha,epoch)
outputs = perceptron_inference(X, l_W)

for x, y in outputs:
    print(f"Input:[{x[0]}, {x[1]}] --> Prediction:{y}\n")
        