from model import RegressionModel
from model_spec import specs
import streamlit as st
import torch
import pandas as pd
from preprocessor import (encode_dataframe,
                          normalize_data_zscore)

# Helper function
# Function to switch pages
def switch_page(page_name):
    st.session_state.page = page_name

def main():
    st.title("Student Performance Prediction")

    # Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "form"

    if "df" not in st.session_state:
        st.session_state.df = None

    # Form Page
    if st.session_state.page == "form":
        st.write("Enter the following details to predict the final grade of a student:")

        with st.form("student_form"):
            sex = st.selectbox('student\'s sex (binary: "F" - female or "M" - male)', ["M", "F"])
            age = st.number_input("student's age (numeric: from 15 to 22)", min_value=15, max_value=22, step=1)
            adress = st.selectbox('student\'s home address type (binary: "U" - urban or "R" - rural)', ["U", "R"])
            famsize = st.selectbox('family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)', ["LE3", "GT3"])
            Pstatus = st.selectbox('parent\'s cohabitation status (binary: "T" - living together or "A" - apart)', ["T", "A"])
            Medu = st.selectbox('mother\'s education (numeric: \n0 - none,  \n1 - primary education (4th grade), \n2 – 5th to 9th grade, \n3 – secondary education or \n4 – higher education)',
                                [0, 1, 2, 3, 4])
            Fedu = st.selectbox('father\'s education (numeric: \n0 - none,  \n1 - primary education (4th grade), \n2 – 5th to 9th grade, \n3 – secondary education or \n4 – higher education)',
                                [0, 1, 2, 3, 4])
            Mjob = st.selectbox('mother\'s job (nominal: \n"teacher", \n"health care related", \n"civil services" (e.g. administrative or police), \n"at_home" or \n"other")',
                                ["teacher", "health", "services", "at_home", "other"])
            Fjob = st.selectbox('father\'s job (nominal: \n"teacher", \n"health care related", \n"civil services" (e.g. administrative or police), \n"at_home" or \n"other")',
                                ["teacher", "health", "services", "at_home", "other"])
            reason = st.selectbox('reason to choose this school (nominal: \n"home", \n"reputation", \n"course" preference or \n"other")',
                                ["home", "reputation", "course", "other"])
            traveltime = st.selectbox('home to school travel time (numeric: \n1 - <15 min., \n2 - 15 to 30 min., \n3 - 30 min. to 1 hour, or \n4 - >1 hour)',
                                    [1, 2, 3, 4])
            studytime = st.selectbox('weekly study time (numeric: \n1 - <2 hours, \n2 - 2 to 5 hours, \n3 - 5 to 10 hours, or \n4 - >10 hours)',
                                        [1, 2, 3, 4])
            failures = st.selectbox('number of past class failures (numeric: n if 1<=n<3, else 4)', [0, 1, 2, 3])
            schoolsup = st.selectbox('extra educational support (binary: yes or no)', ["yes", "no"])
            famsup = st.selectbox('family educational support (binary: yes or no)', ["yes", "no"])
            paid = st.selectbox('extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)', ["yes", "no"])
            activities = st.selectbox('extra-curricular activities (binary: yes or no)', ["yes", "no"])
            nursery = st.selectbox('attended nursery school (binary: yes or no)', ["yes", "no"])
            higher = st.selectbox('wants to take higher education (binary: yes or no)', ["yes", "no"])
            internet = st.selectbox('Internet access at home (binary: yes or no)', ["yes", "no"])
            romantic = st.selectbox('with a romantic relationship (binary: yes or no)', ["yes", "no"])
            famrel = st.selectbox('quality of family relationships (numeric: from 1 - very bad to 5 - excellent)', [1, 2, 3, 4, 5])
            freetime = st.selectbox('free time after school (numeric: from 1 - very low to 5 - very high)', [1, 2, 3, 4, 5])
            goout = st.selectbox('going out with friends (numeric: from 1 - very low to 5 - very high)', [1, 2, 3, 4, 5])
            Dalc = st.selectbox('workday alcohol consumption (numeric: from 1 - very low to 5 - very high)', [1, 2, 3, 4, 5])
            Walc = st.selectbox('weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)', [1, 2, 3, 4, 5])
            health = st.selectbox('current health status (numeric: from 1 - very bad to 5 - very good)', [1, 2, 3, 4, 5])
            absences = st.number_input('number of school absences (numeric: from 0 to 93)', min_value=0, max_value=93, step=1)

            G1 = st.number_input('first period grade (numeric: from 0 to 20)', min_value=0, max_value=20, step=1)
            G2 = st.number_input('second period grade (numeric: from 0 to 20)', min_value=0, max_value=20, step=1)

            st.warning("Click twice on the submit button to submit the form.")
            # Submit button
            submitted = st.form_submit_button("Submit")

            if submitted:

                st.session_state.df = {
                            "sex": sex,
                            "age": age,
                            "address": adress,
                            "famsize": famsize,
                            "Pstatus": Pstatus,
                            "Medu": Medu,
                            "Fedu": Fedu,
                            "Mjob": Mjob,
                            "Fjob": Fjob,
                            "reason": reason,
                            "traveltime": traveltime,
                            "studytime": studytime,
                            "failures": failures,
                            "schoolsup": schoolsup,
                            "famsup": famsup,
                            "paid": paid,
                            "activities": activities,
                            "nursery": nursery,
                            "higher": higher,
                            "internet": internet,
                            "romantic": romantic,
                            "famrel": famrel,
                            "freetime": freetime,
                            "goout": goout,
                            "Dalc": Dalc,
                            "Walc": Walc,
                            "health": health,
                            "absences": absences,
                            "G1": G1,
                            "G2": G2
                }

                st.success("Form submitted successfully!")

                print(st.session_state.df)


                switch_page("next")


    # Next Page
    elif st.session_state.page == "next":

        st.write("Please confirm the entered details:")

        if st.session_state.df is not None:

            with st.form("confirm_form"):
                # Display submitted data as a DataFrame
                df = pd.DataFrame([st.session_state.df])

                # Wrap the DataFrame in a div with the custom class
                html = f'''
                <div class="dataframe-container"> 
                {df.to_html(index=False)}
                </div>
                '''
                st.markdown(html, unsafe_allow_html=True)

                # Confirm button
                confirmed = st.form_submit_button("Confirm")

                if confirmed:
                    # Load the model
                    loaded_model = RegressionModel(specs.input_size)
                    loaded_model.load_state_dict(torch.load('./src/regression_model.pth'))
                    loaded_model.eval()
                    print("Model loaded from regression_model.pth")

                    # Preprocess the data
                    # Encode the data
                    df, _ = encode_dataframe(df, 
                                          columns=specs.data_columns_to_encode, 
                                          vocab=specs.encoding_vocab)
                    
                    # Normalize the data
                    df, _, _ = normalize_data_zscore(df, **specs.norm_specs)
                    
                    # Convert DataFrame to torch tensor
                    df = torch.tensor(df.values, dtype=torch.float32)
                    
                    print("Data preprocessed successfully!")
                    print(df)
                    # Make prediction
                    with torch.no_grad():
                        prediction = loaded_model(df).item()

                    prediction = prediction * specs.norm_specs["std_dict"]["G3"] + specs.norm_specs["mean_dict"]["G3"]

                    st.success(f"The predicted final grade is: {prediction:.2f}")



        else:
            st.warning("No data available!")

        # Provide a "Go Back" button to return to the form
        if st.button("Go Back"):
            st.session_state.form_data = None  # Clear form data
            switch_page("form")



if __name__ == "__main__":
    main()