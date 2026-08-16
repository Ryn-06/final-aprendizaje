from sklearn.linear_model import LogisticRegression

def construir_modelo(input_dim=5):
    return LogisticRegression(C=0.1, random_state=42)