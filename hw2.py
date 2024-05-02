import numpy as np
import time
import matplotlib.pyplot as plt

# (1)
np.random.seed(seed=1)  # 난수를 고정
N = 1000  # 데이터의 수
K = 3  # 분포의 수
Y = np.zeros((N, 3), dtype=np.uint8)
X = np.zeros((N, 2))
Mu = np.array([[-2.0, -2.0], [1.5, 2.0], [2, -1.5]])  # 분포의 중심
Sig = np.array([[0.9, 0.9], [0.8, 0.7], [0.9, 1.0]])  # 분포의 분산
Pi = np.array([0.35, 0.7, 1.0])  # 각 분포에 대한 비율
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            Y[n, k] = 1
            break
    for k in range(2):
        X[n, k] = np.random.randn() * Sig[Y[n, :] == 1, k] + \
            Mu[Y[n, :] == 1, k]


X_train = X[:600]
X_validation = X[600:800]
X_test = X[800:]
Y_train = Y[:600]
Y_validation = Y[600:800]
Y_test = Y[800:]

print(Y_train)

# (2)
P = 8  # 은닉 노드
C = 3  # 출력 노드
D = 2  # 입력 노드

# (3)


def Sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = Sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Sigmoid(Z2)
    return Z1, A1, Z2, A2


def backward_propagation(X, y, Z1, A1, Z2, A2, W1, W2):
    m = len(X)

    dZ2 = A2 - y  # 오차
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * Sigmoid(Z1) * (1 - Sigmoid(Z1))
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0)

    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


def initialize_weights(D, P, c):
    W1 = np.random.randn(D, P)
    b1 = np.zeros(P)
    W2 = np.random.randn(P, c)
    b2 = np.zeros(c)
    return W1, b1, W2, b2


def trainingMLP(X_train, y_train, X_valid, y_valid, num_epochs, batch_size, learning_rate):
    # 초기 가중치 초기화
    W1, b1, W2, b2 = initialize_weights(D, P, C)

    train_errors = []
    valid_errors = []
    best_valid_error = float('inf')
    best_params = None

    for epoch in range(num_epochs):
        # 미니배치 스톡 캐스트 경사 하강법
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward propagation
            Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2)

            # Backward propagation
            dW1, db1, dW2, db2 = backward_propagation(
                X_batch, y_batch, Z1, A1, Z2, A2, W1, W2)

            # 가중치 업데이트
            W1, b1, W2, b2 = update_parameters(
                W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # 훈련 및 검증 세트의 에러 계산
        _, _, _, train_pred = forward_propagation(X_train, W1, b1, W2, b2)
        train_error = mse_loss(train_pred, y_train)
        train_errors.append(train_error)

        _, _, _, valid_pred = forward_propagation(X_valid, W1, b1, W2, b2)
        valid_error = mse_loss(valid_pred, y_valid)
        valid_errors.append(valid_error)

        # 현재까지의 최적 모델 저장
        if valid_error < best_valid_error:
            best_valid_error = valid_error
            best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

        print("Epoch {}/{} - Train Error: {}, Valid Error: {}".format(epoch +
              1, num_epochs, train_error, valid_error))
    return train_errors, valid_errors, best_params


num_epochs = 100
learning_rate = 0.01
batch_sizes = [8, 16, 32]
for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    start_time = time.time()
    train_errors, valid_errors, best_params = trainingMLP(
        X_train, Y_train, X_validation, Y_validation, num_epochs, batch_size, learning_rate)
    end_time = time.time()

    W1, b1, W2, b2 = best_params

    _, _, _, test_pred = forward_propagation(X_test, W1, b1, W2, b2)
    test_error = mse_loss(test_pred, Y_test)

    print(f"Test Error: {test_error}")
    print(f"Training time: {end_time - start_time} seconds")

    plt.plot(range(1, num_epochs + 1), train_errors, label='Train Error')
    plt.plot(range(1, num_epochs + 1), valid_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Error')
    plt.title('Training and Validation Error')
    plt.legend()
    plt.show()

    best_W1, best_b1, best_W2, best_b2 = best_params

    _, _, _, test_pred = forward_propagation(
        X_test, best_W1, best_b1, best_W2, best_b2)

    predicted_classes = np.argmax(test_pred, axis=1)

    accuracy = np.mean(predicted_classes == Y_test.argmax(axis=1))

    print(f"Test Set Accuracy: {accuracy}")
