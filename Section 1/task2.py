import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds # Import svds for Singular Value Decomposition
import csv

class Recommender:
    """
    A recommender class that approximates the user-item rating matrix using SVD.
    This version focuses on:
    - Building the sparse user-item rating matrix.
    - Applying Singular Value Decomposition (SVD) for dimensionality reduction.
    - Predicting ratings based on the SVD-approximated matrix.
    """
    def __init__(self, k=10):
        self.k = k
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = []
        self.idx_to_item = []

        self.n_users = 0
        self.n_items = 0

        self.rating_matrix = None # Original sparse rating matrix
        self.R_hat = None         # SVD-approximated dense rating matrix
        self.global_mean = 0.0    # Store global mean for prediction fallback

    def fit(self, train_data):
        if not train_data:
            print("Warning: train_data is empty. Cannot build rating matrix.")
            return

        # 1. Map user and item IDs to contiguous integer indices
        print("Mapping user and item IDs...")
        for u, i, _ in train_data:
            if u not in self.user_to_idx:
                self.user_to_idx[u] = self.n_users
                self.idx_to_user.append(u)
                self.n_users += 1
            if i not in self.item_to_idx:
                self.item_to_idx[i] = self.n_items
                self.idx_to_item.append(i)
                self.n_items += 1

        print(f"Found {self.n_users} unique users and {self.n_items} unique items.")

        # 2. Compute global mean (will be used for centering and cold-start prediction)
        self.global_mean = np.mean([r for _, _, r in train_data])

        # 3. Create the sparse user-item rating matrix (LIL for efficient building)
        print("Populating rating matrix...")
        R_lil = sp.lil_matrix((self.n_users, self.n_items))
        
        # We need the original rating matrix for MSE calculation later
        self.rating_matrix = sp.lil_matrix((self.n_users, self.n_items))

        for u_id, i_id, rating in train_data:
            u_idx = self.user_to_idx[u_id]
            i_idx = self.item_to_idx[i_id]
            self.rating_matrix[u_idx, i_idx] = rating
            R_lil[u_idx, i_idx] = rating

        # Convert to sparse format for efficient SVD
        R_csr = R_lil.tocsr()
        self.rating_matrix = self.rating_matrix.tocsr()

        print("Rating matrix built successfully.")

        # 4. Perform Truncated SVD (Stage 2)
        print(f"Performing SVD with k={self.k} latent factors...")
        
        U, s, Vt = svds(R_csr, k=self.k)

        # Reconstruct the approximated matrix R_hat.
        self.R_hat = U @ np.diag(s) @ Vt

        print("SVD approximation complete.")

    def predict(self, u, i, clip=True):
        """
        Predicts the rating for a given user and item using the SVD-approximated matrix.
        Handles cold-start (unseen users/items).
        """
        u_idx = self.user_to_idx.get(u)
        i_idx = self.item_to_idx.get(i)

        # Handle cold-start: if user or item not seen in training, return global mean
        if u_idx is None or i_idx is None:
            return self.global_mean # Fallback for unseen users/items
        
        # Predict from the SVD-approximated matrix
        pred = self.R_hat[u_idx, i_idx]

        # Clip predictions to the valid rating range (1.0 to 5.0)
        return float(np.clip(pred, 1.0, 5.0) if clip else pred)

    def mse(self, data):
        """
        Calculates the Mean Squared Error (MSE) on the given dataset.
        This uses the predict method, which in turn uses the SVD-approximated matrix.
        """
        if not data:
            return 0.0

        squared_errors = []
        for u, i, r in data:
            pred = self.predict(u, i, clip=False)
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
        writer.writerow(['user id', 'item id', 'rating'])
        for user, item in test_data:
            pred = model.predict(user, item)
            writer.writerow([user, item, pred])


def solve(train_path='train.csv', test_path='test.csv', pred_path='pred2.csv'):
    # Load training and test data using your existing helper functions
    train_data = load_train_data(train_path)
    test_data = load_test_data(test_path)


    # Instantiate the Recommender class for SVD
    print("Initializing Recommender (SVD) model...")
    rec = Recommender(k=10) # Set k=10 as required

    # Fit the model (This will build the matrix and perform SVD)
    rec.fit(train_data)

    # Compute MSE on training set (Stage 3 part 1)
    mse_train = rec.mse(train_data)
    
    # Append MSE to mse.txt
    with open('mse.txt', 'a') as f:
        f.write(f"{mse_train}")
    print(f"Train MSE for Task 2 (SVD k=10): {mse_train}")

    # Predict ratings for test set (Stage 3 part 2)
    preds = [
        float(np.clip(rec.predict(u, i), 1.0, 5.0))
        for u, i in test_data
    ]

    # Save predictions to pred2.csv using the helper function
    save_predictions(test_data, rec, pred_path)
    print(f"Predictions saved to {pred_path}")

    return preds, mse_train


if __name__ == '__main__':
    solve()
