import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from sklearn.preprocessing import normalize
import pickle
import tensorflow as tf
import networkx as nx
# Load Logistic Regression model and results
with open('lr_model_results.pkl', 'rb') as f:
    results = pickle.load(f)
with open('mc_model_results.pkl', 'rb') as f:
    results1 = pickle.load(f)

# Load your session data
session = pd.read_csv('data/session.csv')
def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded_sequences[i] = np.array(seq[:maxlen])
        else:
            padded_sequences[i, :len(seq)] = np.array(seq)
    return padded_sequences
def onegrams(seq):
    o=[]
    for s in seq:
        o.append(str(s))
    o=[s for s in o if '5' not in s]
    freq_ngrams = Counter(o)
    nfreq_onegrams = {s: count / len(o) for s, count in freq_ngrams.items()}
    return nfreq_onegrams

def twograms(seq):
    t=[]
    for i in range(len(seq) - 1):
        t.append(''.join(map(str,seq[i:i+2])))
    t=[s for s in t if '5' not in s]
    freq_ngrams = Counter(t)
    nfreq_twograms = {s: count / len(t) for s, count in freq_ngrams.items() if count>0}
    return nfreq_twograms

def create_hvg(seq):
    n = len(seq)
    hvg = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if all(seq[k] < min(seq[i], seq[j]) for k in range(i+1, j)):
                hvg.add_edge(i, j)
    return hvg

def hvg_motifs(hvg):
    motifs = []
    nodes = list(hvg.nodes)
    for i in range(len(nodes) - 3):
        subgraph = hvg.subgraph(nodes[i:i+4])
        motif = nx.to_dict_of_lists(subgraph)
        motifs.append(tuple(sorted((frozenset(motif[k]) for k in motif))))
    return motifs

def entropy(motifs):
    counts = Counter(motifs)
    total = sum(counts.values())
    prob = [count / total for count in counts.values()]
    e = -sum(p * np.log2(p) for p in prob)
    return e
def preprocess_input(data):
    symbol_map = {
    'view': 1,
    'detail': 2,
    'add': 3,
    'remove': 4,
    'purchase': 5,
    'click': 6}
    seq=[symbol_map.get(action, 1) for action in data]
    print(seq)
    hvg = create_hvg(seq)
    motifs = hvg_motifs(hvg)
    e = entropy(motifs)
    nfreq_onegrams=onegrams(seq)
    nfreq_twograms=twograms(seq)
    all_possible_1grams=['1','2','3','4','6']
    all_possible_2grams=['11','12','13','14','16','21','22','23','24','26','31','32','33','34','36','41','42','43','44','46','61','62','63','64','66']
    onegramss = [nfreq_onegrams.get(gram, 0) for gram in all_possible_1grams]
    twogramss = [nfreq_twograms.get(kgram, 0) for kgram in all_possible_2grams]
    mostcommon_motifs=pd.read_csv('data/motiflist.csv')['motifs']
    counts = Counter(motifs)
    X=[]
    motif_cnt = [counts.get(motif, 0)  for motif in mostcommon_motifs]
    X.append(onegramss+twogramss+motif_cnt + [e])
    return X


# Function to display prediction report and analysis for the LR model
def lr_model_report():
    # Unpack results
    mean_f1_lr, std_f1_lr = results['mean_f1_lr'], results['std_f1_lr']
    mean_auc_lr, std_auc_lr = results['mean_auc_lr'], results['std_auc_lr']
    mean_acc_lr, std_acc_lr = results['mean_acc_lr'], results['std_acc_lr']
    mean_prec_lr, std_prec_lr = results['mean_prec_lr'], results['std_prec_lr']
    mean_rec_lr, std_rec_lr = results['mean_rec_lr'], results['std_rec_lr']
    coefficients = results['coefficients']
    feature_names = results['feature_names']
    f1_lr = results['f1_lr']
    auc_lr = results['auc_lr']
    acc_lr = results['acc_lr']
    prec_lr = results['prec_lr']
    rec_lr = results['rec_lr']
    f1_lr_t=results['f1_lr_t']
    auc_lr_t=results['auc_lr_t']
    acc_lr_t=results['acc_lr_t']
    prec_lr_t=results['prec_lr_t']
    rec_lr_t=results['rec_lr_t']
    earliness_values=results['earliness_values'] 
    # Streamlit App Layout
    st.title("Model Results")

    # Display metrics
    st.subheader("Logistic Regression Model Performance Metrics")
    st.write(f'F1 Score: {mean_f1_lr:.4f} ± {std_f1_lr:.4f}')
    st.write(f'AUC: {mean_auc_lr:.4f} ± {std_auc_lr:.4f}')
    st.write(f'Accuracy: {mean_acc_lr:.4f} ± {std_acc_lr:.4f}')
    st.write(f'Precision: {mean_prec_lr:.4f} ± {std_prec_lr:.4f}')
    st.write(f'Recall: {mean_rec_lr:.4f} ± {std_rec_lr:.4f}')
    metrics=results['metrics']
    for model in metrics['f1']:
        st.subheader(f"\n{model.upper()} Model Metrics:")
        for metric_name in metrics:
            mean_metric = np.mean(metrics[metric_name][model])
            std_metric = np.std(metrics[metric_name][model])
            st.write(f"{metric_name.capitalize()} - Mean: {mean_metric:.4f} ± {std_metric:.4f}")
    # Plotting Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coefficients)
    plt.xlabel("Coefficient Value")
    plt.title("Feature Importance")
    st.pyplot(plt)
    st.write("add-add has the largest positive coefficient, indicating a strong positive influence on tconversion. Features like add, view-deatil, and remove-add also have significant positive coefficients, contributing positively to the model's prediction. On the negative side, add-remove  have notable negative coefficients, suggesting that it lower the likelihood of the conversion when their values increase.")
    # Plotting metric trends across iterations
    iterations = list(range(1, 8))
    plt.figure(figsize=(12, 8))
    plt.plot(iterations, f1_lr, label="F1 Score", marker='o')
    plt.plot(iterations, auc_lr, label="AUC", marker='o')
    plt.plot(iterations, acc_lr, label="Accuracy", marker='o')
    plt.plot(iterations, prec_lr, label="Precision", marker='o')
    plt.plot(iterations, rec_lr, label="Recall", marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Performance Metrics Over Iterations")
    plt.legend()
    plt.grid()
    st.pyplot(plt)
    T_values = range(5, 15)
    st.subheader("EARLY PREDICTION")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(T_values, f1_lr_t, marker='o', color='b', label='F1 Score')
    plt.plot(T_values, auc_lr_t, marker='o', color='g', label='AUC')
    plt.title("F1 Score and AUC vs T")
    plt.xlabel("T")
    plt.ylabel("Score")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(T_values, acc_lr_t, marker='o', color='purple', label='Accuracy')
    plt.plot(T_values, prec_lr_t, marker='o', color='orange', label='Precision')
    plt.plot(T_values, rec_lr_t, marker='o', color='red', label='Recall')
    plt.title("Accuracy, Precision, Recall vs T")
    plt.xlabel("T")
    plt.ylabel("Score")
    plt.legend()


    plt.subplot(1, 3, 3)
    plt.plot(T_values, earliness_values, marker='o', color='r', label='Earliness')
    plt.title("Earliness vs T")
    plt.xlabel("T")
    plt.ylabel("Earliness (%)")
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)
    st.write("Low 𝑇 values (e.g., 𝑇=5)  Offer high earliness, allowing predictions early in the session, but with lower model confidence. Higher 𝑇 values (e.g., 𝑇=9) Provide a stable and accurate classification performance with higher F1 and AUC scores, though at the expense of reduced earliness.")
    st.write("the increasing F1 and AUC scores for larger 𝑇 suggest that models indeed benefit from observing more events")
    
with open('lstm_model_results.pkl', 'rb') as f:
    lstm_results = pickle.load(f)
# Function to display prediction report and analysis for the LSTM model
def lstm_model_report():
    st.subheader("LSTM Model")
    st.write("### Model Analysis")
    # Extract metrics
    accuracy = lstm_results['accuracy']
    auc = lstm_results['auc']
    precision_purchase = lstm_results['precision_purchase']
    recall_purchase = lstm_results['recall_purchase']
    f1_purchase = lstm_results['f1_purchase']
    precision_non_purchase = lstm_results['precision_non_purchase']
    recall_non_purchase = lstm_results['recall_non_purchase']
    f1_non_purchase = lstm_results['f1_non_purchase']
    window_sizes = lstm_results['window_sizes']
    accuracies = lstm_results['accuracies']
    auc_scores = lstm_results['auc_scores']
    f1_scores = lstm_results['f1_scores']
    cm = lstm_results['cm']
    fpr=lstm_results['fpr']
    tpr=lstm_results['tpr']
    roc_auc=lstm_results['roc_auc']
    # Streamlit App Layout
    st.title("LSTM Model Results")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Purchase", "Purchase"], yticklabels=["Non-Purchase", "Purchase"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

    # Display metrics
    st.subheader("Model Performance Metrics")
    st.write(f'Accuracy: {accuracy:.4f}')
    st.write(f'AUC: {auc:.4f}')
    st.write(f'Precision (Purchase): {precision_purchase:.4f}')
    st.write(f'Recall (Purchase): {recall_purchase:.4f}')
    st.write(f'F1 Score (Purchase): {f1_purchase:.4f}')
    st.write(f'Precision (Non-Purchase): {precision_non_purchase:.4f}')
    st.write(f'Recall (Non-Purchase): {recall_non_purchase:.4f}')
    st.write(f'F1 Score (Non-Purchase): {f1_non_purchase:.4f}')
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)
with open("models/lr_model_6.pkl", "rb") as f:
    lr_model = pickle.load(f)
lstm_model = tf.keras.models.load_model("models/lstm_model_f.keras")
# Function to make predictions using the selected model
with open('vis_results.pkl', 'rb') as f:
    vis_results = pickle.load(f)


st.title("Prediction and Visualization App")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose an option", ["Prediction", "Visualization"])

if option == "Prediction":
    model_type = st.selectbox("Choose a model", ["Feature Based Classification", "LSTM","MARKOV CHAIN CLASSIFIER"])
    
    if model_type == "Feature Based Classification":
        lr_model_report()
    elif model_type == "MARKOV CHAIN CLASSIFIER":
        st.title("MARKOV CHAIN CLASSIFIER RESULTS")
        mean_f1_m= results1['mean_f1_m']
        std_f1_m= results1['std_f1_m']
        mean_auc_m= results1['mean_auc_m']
        std_auc_m= results1['std_auc_m']
        mean_acc_m= results1['mean_acc_m']
        std_acc_m= results1['std_acc_m']
        mean_prec_m= results1['mean_prec_m']
        std_prec_m= results1['std_prec_m']
        mean_rec_m= results1['mean_rec_m']
        std_rec_m= results1['std_rec_m']
        mean_fpr= results1['mean_fpr']
        tprs=results1['tprs']
        auc_m=results1['auc_m']
        plt.plot([0, 1], [0, 1], 'k--')
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(auc_m)
        std_auc = np.std(auc_m)
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
        plt.fill_between(mean_fpr, np.maximum(mean_tpr - np.std(tprs, axis=0), 0), np.minimum(mean_tpr + np.std(tprs, axis=0), 1), color='blue', alpha=0.2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
        st.write(f"Markov Chain Classifier Results (k={5}):")
        st.write(f'F1 Score: {mean_f1_m:.4f} ± {std_f1_m:.4f}')
        st.write(f'AUC: {mean_auc_m:.4f} ± {std_auc_m:.4f}')
        st.write(f'Accuracy: {mean_acc_m:.4f} ± {std_acc_m:.4f}')
        st.write(f'Precision: {mean_prec_m:.4f} ± {std_prec_m:.4f}')
        st.write(f'Recall: {mean_rec_m:.4f} ± {std_rec_m:.4f}')
    else:
        lstm_model_report()

    # Input for predictions
    
        input_data = st.text_input("Enter your sequence (e.g., 'view, detail, add, view'):")
        input_data = list(map(str, input_data.split(',')))
        if st.button("Predict"):
                symbol_map = {
                'view': 1,
                'detail': 2,
                'add': 3,
                'remove': 4,
                'purchase': 5,
                'click': 6}
                seq=[symbol_map.get(action, 1) for action in input_data]
                print(seq)
                padded_data=pad_sequences([seq], maxlen=100)
                prediction = lstm_model.predict(padded_data) 
                st.text(f"probability of purchase: {prediction}")
                print(prediction)
        
elif option == "Visualization":
    symbol_map = {
    'view': 1,
    'detail': 2,
    'add': 3,
    'remove': 4,
    'purchase': 5,
    'click': 6}
    st.subheader("Visualizations")
    st.title("Session Analysis Visualization")

    # Session Duration Distribution
    st.subheader("Session Duration Distribution")
    plt.figure(figsize=(10, 6))
    plt.hist(session['total_duration'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Session Duration Distribution')
    plt.xlabel('Total Duration (seconds)')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    st.write("Most Sessions Are Short: The majority of session durations are clustered close to the left, indicating that most user sessions are very short")
    st.write("Long-Tail Distribution: There are a few sessions with very long durations, stretching up to 60,000 seconds (about 16 hours)")

    # Event Type Frequencies by Conversion Status
    st.subheader("Event Type Frequencies by Conversion Status")
    action_counts=vis_results['action_counts']
    melted_counts = vis_results['melted_counts']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted_counts, x='event_type', y='count', hue='conversion', ax=ax)
    plt.title('Event Type Frequencies by Conversion Status')
    plt.xlabel('Event Type')
    plt.ylabel('Average Count per Session')
    plt.legend(title='Conversion')
    st.pyplot(fig)
    st.write("Users who exhibit higher engagement with the platform, as reflected by frequent add and detail actions, are more likely to convert.")
    # Symbolized Event Sequence Patterns by Conversion Status
    st.subheader("Symbolized Event Sequence Patterns by Conversion Status")
    counts_df = pd.DataFrame({
        'Conversion': vis_results['conversion_counts'],
        'Non-Conversion': vis_results['non_conversion_counts']
    }, index=['1', '2', '3', '4', '6'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(counts_df, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
    plt.title('Symbolized Event Sequence Patterns by Conversion Status')
    plt.xlabel('Session Type')
    plt.ylabel('Event Symbols')
    st.pyplot(fig)
    st.write("Converted users demonstrate a higher level of commitment by performing more add actions and fewer view or click actions.")
    st.write("Non-converted users engage more in exploratory behaviors, like viewing and clicking, but tend not to progress to actions that lead to purchases.")
    # Conversion Rate by Session Duration
    st.subheader("Conversion Rate by Session Duration")
    conversion_rate = vis_results['conversion_rate']
    plt.figure(figsize=(10, 6))
    conversion_rate.plot(kind='line', marker='o', color='coral')
    plt.title('Conversion Rate by Session Duration')
    plt.xlabel('Session Duration')
    plt.ylabel('Conversion Rate')
    plt.grid(True)
    st.pyplot(plt)
    st.write("This trend indicates that longer sessions are correlated with a higher likelihood of conversion.It suggests that users who spend more time on the site are more engaged, perhaps browsing products, reading details, or adding items to their cart, which may ultimately lead to a purchase.")
    # Conversion Rate by Event Count
    st.subheader("Conversion Rate by Event Count")
    conversion_rate_event_count = vis_results['conversion_rate_event_count']
    plt.figure(figsize=(10, 6))
    conversion_rate_event_count.plot(kind='bar', color='slateblue', edgecolor='black')
    plt.title('Conversion Rate by Event Count')
    plt.xlabel('Event Count')
    plt.ylabel('Conversion Rate')
    st.pyplot(plt)
    st.write("There is a clear positive correlation between the number of events in a session and the conversion rate. Sessions with a higher count of interactions (events) show a significantly higher likelihood of conversion")
     # Transition Matrix for Event Sequences
    st.subheader("Transition Matrix for Event Sequences")
    transition_matrix = vis_results['transition_matrix']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, cmap="coolwarm", xticklabels=['1', '2', '3', '4', '5', '6'], yticklabels=['1', '2', '3', '4', '5', '6'], ax=ax)
    plt.title('Transition Matrix for Event Sequences')
    plt.xlabel('Next Event')
    plt.ylabel('Current Event')
    st.pyplot(fig)
    
    st.write("The events add, remove, and view have relatively high probabilities of transitioning to themselves. For instance, if a user adds a product to the cart, there's an 85% chance they might add another item, indicating repetitive actions in the same category.The view event has a 50% probability of being followed by another view action, suggesting that users often view multiple items in a row without performing other actions.")
    st.write("From view to detail: There’s a high probability (47%) of transitioning from a view to a detail event, which aligns with typical browsing behavior where users are likely to explore item details after an initial view.From detail to add: After viewing details, there is a 13% chance that users add the item to their cart, indicating a moderate level of intent conversion from detail to add.From add to remove: Interestingly, there’s a 13% probability that users will remove items after adding them, which may indicate second thoughts or reconsideration.")

