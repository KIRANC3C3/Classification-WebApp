import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve


def plot_confusion_matrix(model, x_test, y_test, display_labels):
    cm = confusion_matrix(y_test, model.predict(x_test), labels=display_labels)
    st.text("Confusion Matrix:")
    st.write(cm)


def plot_roc_curve(model, x_test, y_test):
    try:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
        st.text("ROC Curve:")
        st.line_chart(pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr}))
    except AttributeError as e:
        st.warning("This classifier does not support probability estimates for ROC curve.")
def plot_precision_recall_curve(model, x_test, y_test):
    try:
        precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(x_test)[:, 1])
        st.text("Precision-Recall Curve:")
        st.line_chart(pd.DataFrame({'Precision': precision, 'Recall': recall}))
    except AttributeError as e:
        st.warning("This classifier does not support probability estimates for Precision-Recall curve.")




def plot_metrics(metrics_list, model, x_test, y_test, display_labels):
    if 'Confusion Matrix' in metrics_list:
        plot_confusion_matrix(model, x_test, y_test, display_labels)

    if 'ROC Curve' in metrics_list:
        plot_roc_curve(model, x_test, y_test)

    if 'Precision - Recall Curve' in metrics_list:
        plot_precision_recall_curve(model, x_test, y_test)


def main():
    st.title("Classification Web Application")
    st.sidebar.title("Classification Web Application")
    st.markdown("Upload your CSV file and perform classification ")
    st.sidebar.markdown("Upload your CSV file and perform classification ")

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        label = LabelEncoder()
        for col in df.columns:
            df[col] = label.fit_transform(df[col])

        st.subheader("User-Uploaded Dataset:")
        st.write(df)

        y = df.type
        x = df.drop(columns=['type'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox("Choose Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest Classification"))

        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Choose Model Hyperparameters:")
            C = st.sidebar.number_input("C (Strength of Regularization)", 0.01, 100.0, step=0.01, key="C")
            kernel = st.sidebar.radio("Kernel for SVM Classification", ("rbf", "linear"), key="kernel")
            gamma = st.sidebar.radio("Kernel Coefficient (Gamma)", ("scale", "auto"), key='gamma')

            metrics = st.sidebar.multiselect("Choose the Metrics to be plotted",
                                             ("Precision - Recall Curve", "ROC Curve", "Confusion Matrix"))

            if st.sidebar.button("Classify, Get Results!"):
                st.subheader("Support Vector Machine (SVM) Classifier Results: ")
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_predicted = model.predict(x_test)
                st.write("Accuracy", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_predicted).round(2))
                st.write("Recall: ", recall_score(y_test, y_predicted).round(2))
                st.write("f1 score: ", f1_score(y_test, y_predicted).round(2))
                st.write("ROC: Area Under the Curve: ", roc_auc_score(y_test, y_predicted).round(2))

                plot_metrics(metrics, model, x_test, y_test, display_labels=[0, 1])

        elif classifier == 'Logistic Regression':
            st.sidebar.subheader("Choose Model Hyperparameters:")
            C = st.sidebar.number_input("C (Strength of Regularization)", 0.01, 100.0, step=0.01, key="C_log")
            max_iter = st.sidebar.slider("Maximum Number of Iterations: ", 100, 500, key='max_iter')

            metrics = st.sidebar.multiselect("Choose the Metrics to be plotted",
                                             ("Precision - Recall Curve", "ROC Curve", "Confusion Matrix"))

            if st.sidebar.button("Classify, Get Results!"):
                st.subheader("Logistic Regression Classifier Results: ")
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_predicted = model.predict(x_test)
                st.write("Accuracy", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_predicted).round(2))
                st.write("Recall: ", recall_score(y_test, y_predicted).round(2))
                st.write("f1 score: ", f1_score(y_test, y_predicted).round(2))
                st.write("ROC: Area Under the Curve: ", roc_auc_score(y_test, y_predicted).round(2))

                plot_metrics(metrics)

        elif classifier == 'Random Forest Classification':
            st.sidebar.subheader("Choose Model Hyperparameters:")
            n_estimators = st.sidebar.number_input("The number of trees in the forest:", 1, 20, step=1, key='n_estimators')
            max_depth = st.sidebar.number_input("The maximum Depth of Each Tree: ", 1, 20, step=1, key='max_depth')
            bootstrap = st.sidebar.radio("Bootstrap Samples when building Trees", ('True', 'False'), key='bootstrap')

            metrics = st.sidebar.multiselect("Choose the Metrics to be plotted",
                                             ("Precision - Recall Curve", "ROC Curve", "Confusion Matrix"))

            if st.sidebar.button("Classify, Get Results!"):
                st.subheader("Random Forest Classifier Results: ")
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                               n_jobs=-1)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_predicted = model.predict(x_test)
                st.write("Accuracy", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_predicted).round(2))
                st.write("Recall: ", recall_score(y_test, y_predicted).round(2))
                st.write("f1 score: ", f1_score(y_test, y_predicted).round(2))
                st.write("ROC: Area Under the Curve: ", roc_auc_score(y_test, y_predicted).round(2))

                plot_metrics(metrics)

    else:
        st.sidebar.info("Please upload a CSV file.")

if __name__ == '__main__':
  main()
