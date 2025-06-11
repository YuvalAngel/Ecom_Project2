import csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class RecommenderLSBias:
    """
    Bias-Only Recommender System using Least Squares (Normal Equations) to find biases.

    This model predicts a rating r_hat for a user u and item i as:
    r_hat = global_mean + user_bias_u + item_bias_i

    Hyperparameters:
    - lambda_reg: Regularization strength for user and item biases.
                  Note: This is applied as lambda_reg * I to A.T @ A.
    """
    def __init__(self, lambda_reg=1.0):
        self.lambda_reg = lambda_reg
        self.r_avg = 0.0
        self.b_u = {} # User biases
        self.b_i = {} # Item biases
        self._user_to_idx = {} # Map user_id to an integer index
        self._item_to_idx = {} # Map item_id to an integer index
        self._idx_to_user = [] # Map integer index back to user_id
        self._idx_to_item = [] # Map integer index back to item_id

    def fit(self, train_data):
        """
        Trains the bias-only model by solving a linear least squares problem
        with L2 regularization.

        Args:
            train_data (list): A list of tuples (user_id, item_id, rating).
        """
        if not train_data:
            print("Warning: train_data is empty. Cannot fit model.")
            return

        # 1. Compute global mean
        self.r_avg = np.mean([r for _, _, r in train_data])

        # 2. Map user and item IDs to contiguous integer indices
        current_user_idx = 0
        current_item_idx = 0
        for u, i, _ in train_data:
            if u not in self._user_to_idx:
                self._user_to_idx[u] = current_user_idx
                self._idx_to_user.append(u)
                current_user_idx += 1
            if i not in self._item_to_idx:
                self._item_to_idx[i] = current_item_idx
                self._idx_to_item.append(i)
                current_item_idx += 1

        num_users = len(self._user_to_idx)
        num_items = len(self._item_to_idx)
        num_biases = num_users + num_items # Total number of parameters in 'b'

        # 3. Construct the A matrix and c vector for the system A.T @ A @ b = A.T @ c
        # A is a sparse matrix: rows are ratings, columns are biases (user_0...user_N, item_0...item_M)
        # Each row of A will have two '1's: one for the user bias, one for the item bias.
        rows = []
        cols = []
        data = []
        c_vector = []

        for row_idx, (u, i, r) in enumerate(train_data):
            u_idx = self._user_to_idx[u]
            i_idx = self._item_to_idx[i]

            # Contribution to A for user bias
            rows.append(row_idx)
            cols.append(u_idx)
            data.append(1.0)

            # Contribution to A for item bias
            rows.append(row_idx)
            cols.append(num_users + i_idx) # Item biases come after all user biases
            data.append(1.0)

            # Corresponding entry in c vector (r - r_avg)
            c_vector.append(r - self.r_avg)

        # Create sparse matrix A
        A = sp.csc_matrix((data, (rows, cols)), shape=(len(train_data), num_biases))
        c = np.array(c_vector)

        # 4. Formulate and solve the normal equations: (A.T @ A + lambda_reg * I) @ b = A.T @ c
        print("Solving normal equations...")
        ATA = A.T @ A

        # Add regularization term: lambda_reg * I (identity matrix)
        # We only add to the diagonal elements where biases are.
        # Create a sparse identity matrix for regularization
        reg_matrix = sp.eye(num_biases, format='csc') * self.lambda_reg
        ATA_reg = ATA + reg_matrix

        ATc = A.T @ c

        # Solve the linear system
        # spsolve is suitable for sparse matrices
        biases_flat = spsolve(ATA_reg, ATc)

        # 5. Distribute the solved biases back to b_u and b_i dictionaries
        for u_id, idx in self._user_to_idx.items():
            self.b_u[u_id] = biases_flat[idx]

        for i_id, idx in self._item_to_idx.items():
            self.b_i[i_id] = biases_flat[num_users + idx]

        print("Model fitting complete.")

    def predict(self, u, i):
        """
        Predicts the rating for a given user and item using learned biases.

        Args:
            u: User ID.
            i: Item ID.

        Returns:
            float: Predicted rating, clipped between 1.0 and 5.0.
        """
        # Get user bias, default to 0.0 if user is unseen
        user_bias = self.b_u.get(u, 0.0)
        # Get item bias, default to 0.0 if item is unseen
        item_bias = self.b_i.get(i, 0.0)

        # Calculate full prediction
        pred = self.r_avg + user_bias + item_bias

        # Clip the prediction to the valid rating range (e.g., 1 to 5)
        return float(np.clip(pred, 1.0, 5.0))

    def mse(self, data):
        """
        Calculates the Mean Squared Error (MSE) on the given dataset.

        Args:
            data (list): A list of tuples (user_id, item_id, actual_rating).

        Returns:
            float: The calculated MSE.
        """
        if not data:
            return 0.0 # Avoid division by zero if data is empty

        squared_errors = []
        for u, i, r in data:
            pred = self.predict(u, i)
            squared_errors.append((r - pred) ** 2)
        return np.mean(squared_errors)
    


def load_train_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            user, item, rating = row
            data.append((user, item, float(rating)))
    return data


def load_test_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            user, item = row
            data.append((user, item))
    return data



def save_predictions(test_data, model, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        # Write header (optional)
        writer.writerow(['user_id', 'item_id', 'predicted_rating'])
        for user, item in test_data:
            pred = model.predict(user, item)
            writer.writerow([user, item, pred])



def solve(train_path='train.csv', test_path='test.csv', pred_path='pred1.csv'):
    # Load training and test data
    train_data = load_train_data(train_path)
    test_data = load_test_data(test_path)

    print("Training RecommenderLSBias model...")
    rec = RecommenderLSBias(lambda_reg=1.0) # Keep lambda_reg at 1.0 as per your constraint
    rec.fit(train_data)


    # Compute MSE on training set
    mse_train = rec.mse(train_data)
    with open('mse.txt', 'w') as f:
        f.write(f"{mse_train}\n")
    print(f"Train MSE for Task 1: {mse_train}")

    # Predict ratings for test set
    preds = [
        float(np.clip(rec.predict(u, i), 1.0, 5.0))
        for u, i in test_data
    ]

    # Write predictions to CSV
    with open(pred_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user id', 'item id', 'rating'])
        for (u, i), r in zip(test_data, preds):
            writer.writerow([u, i, r])

    return preds, mse_train


if __name__ == "__main__":
    solve()
