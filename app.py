from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the trained model and scaler
filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

scaler_filename = 'scaler.sav'
scaler = pickle.load(open(scaler_filename, 'rb'))

# Load the dataset
df = pd.read_csv("Clustered_Customer_Data.csv")

# Set up the Streamlit app
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Customer Cluster Prediction")

with st.form("my_form"):
    # Create input fields
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(label='OneOff Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(label='OneOff Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input(label='Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Trx', step=1)
    purchases_trx = st.number_input(label='Purchases TRX', step=1)
    credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input(label='PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)

    submitted = st.form_submit_button("Submit")

if submitted:
    # Prepare the data
    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases,
             cash_advance, purchases_frequency, oneoff_purchases_frequency,
             purchases_installment_frequency, cash_advance_frequency, cash_advance_trx,
             purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]

    # Apply the saved scaling from the scaler
    data_scaled = scaler.transform(data)

    # Make predictions
    clust = loaded_model.predict(data_scaled)[0]
    st.write(f'Data Belongs to Cluster: {clust}')

    # Plotting
    cluster_df = df[df['Cluster'] == clust]
    plt.rcParams["figure.figsize"] = (20, 3)

    # Creating subplots for each feature in the cluster
    for c in cluster_df.drop(['Cluster'], axis=1):
        fig, ax = plt.subplots()  # Create a new figure and axes for each plot
        sns.histplot(cluster_df[c], ax=ax)  # Use histplot instead of plt.hist
        ax.set_title(f'Distribution of {c} for Cluster {clust}')  # Set a title for clarity
        st.pyplot(fig)






