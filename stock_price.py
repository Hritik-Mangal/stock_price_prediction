def create_square_matrix(n):
    return [[1000 if i == j else 0 for j in range(n)] for i in range(n)]

def create_column_matrix(rows):
    return [[0] for _ in range(rows)]

def create_design_matrix(file_path):
    design_matrix = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:  # Skipping header
                values = line.strip().split(',')
                row = [1] + [float(value) for value in values[:4]] + [float(values[-1])]  # Bias term included
                design_matrix.append(row)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit()
    return design_matrix

def mat_mult(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(A[0]))) for j in range(len(B[0]))] for i in range(len(A))]

def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

def matrix_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scalar_multiply_matrix(scalar, M):
    return [[scalar * M[i][j] for j in range(len(M[0]))] for i in range(len(M))]

def iterative_update(design_matrix, P, W):
    for i, row in enumerate(design_matrix):
        beta = [[float(val)] for val in row[:-1]]  # Feature vector as column matrix
        y = float(row[-1])  # Actual output

        P_beta = mat_mult(P, beta)  # Compute P * beta
        betaT = transpose(beta)
        denominator = max(1e-6, 1 + mat_mult(betaT, P_beta)[0][0])  # Prevent division errors
        K = scalar_multiply_matrix(1 / denominator, P_beta)  # Kalman gain
        error = y - mat_mult(betaT, W)[0][0]  # Prediction error

        # Update weights
        W = [[W[i][0] + K[i][0] * error] for i in range(len(W))]
        P = matrix_sub(P, mat_mult(K, mat_mult(betaT, P)))  # Update covariance matrix
    return W, P

def predict_next_opening_price(closing_price, volume, prev_high, prev_low, weights):
    """
    Predicts the next day's opening stock price based on input parameters and learned weights.
    """
    input_vector = [[1], [closing_price], [volume], [prev_high], [prev_low]]  # Bias term included
    return mat_mult(transpose(weights), input_vector)[0][0]

# Load data
file_path = "C:\\Projects\\Stock Market Price Prediction\\stock_prices_next_open.csv"
d = 5  # 4 parameters + 1 bias term
P0 = create_square_matrix(d)
W0 = create_column_matrix(d)

design_matrix = create_design_matrix(file_path)
final_weights, final_P = iterative_update(design_matrix, P0, W0)

# Get user input
closing_price = float(input("Enter Closing Price (Rs): "))
volume = float(input("Enter Volume: "))
prev_high = float(input("Enter Previous Day High (Rs): "))
prev_low = float(input("Enter Previous Day Low (Rs): "))

# Predict next opening price
predicted_price = predict_next_opening_price(closing_price, volume, prev_high, prev_low, final_weights)
print(f"Predicted Opening Price: INR {predicted_price}")