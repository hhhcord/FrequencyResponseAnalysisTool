import numpy as np

def compute_optimal_v(n, u, y):
    # データ数の取得
    N = len(y)-1

    # Y の構築
    Y = np.array(y[n:]).reshape(-1, 1)

    # X_1 の構築
    X_1 = np.zeros((N-n+1, n))
    for i in range(N-n+1):
        for j in range(n):
            X_1[i, j] = y[n-1+i-j]

    # X_2 の構築
    X_2 = np.zeros((N-n+1, n+1))
    for i in range(N-n+1):
        for j in range(n+1):
            X_2[i, j] = u[n+i-j]
    
    # X の構築
    X = np.hstack((X_1, X_2))

    # 行列が正則かチェック
    if np.linalg.det(X.T @ X) == 0:
        raise ValueError("行列が正則ではありません。")

    # 最適化問題を解く
    v = np.linalg.inv(X.T @ X) @ X.T @ Y

    # vからv1とv2を抽出
    v1 = v[:n]
    v2 = v[n:]

    # 係数ベクトルaの生成
    a = np.zeros(n+1)
    for i in range(n+1):
        if i==0:
            a[i] = 1
        else:
            a[i] = v1[i-1]

    # 係数ベクトルbの生成
    b = np.zeros(n+1)
    for i in range(n+1):
        b[i] = v2[i]

    print('v>>')
    print(v)

    return a, b

# 使用例
n = 3  # n の値
u = [1, 2, 2, 3, 1, 1, 2, 8]  # u ベクトル
y = [1, 3, 5, 7, 9, 11, 13, 15]  # y ベクトル

try:
    a, b = compute_optimal_v(n, u, y)
    print(a)
    print(b)
except ValueError as e:
    print(e)
