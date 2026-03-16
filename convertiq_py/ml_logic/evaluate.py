from sklearn.metrics import roc_auc_score

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict_proba(X_test)[:,1]

    score = roc_auc_score(y_test, y_pred)

    return score