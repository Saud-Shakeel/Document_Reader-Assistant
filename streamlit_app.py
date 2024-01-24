import streamlit as st
import langchain_helper as lch
import os
import textwrap

st.title('Langchain Document Reader/Assistant')

# Create the 'temp' directory if it doesn't exist
os.makedirs("temp", exist_ok=True)

with st.sidebar:
    with st.form(key='form'):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
        user_query = st.sidebar.text_area(label='Ask me About the Document ?', key='query')

        submit_btn = st.form_submit_button(label='submit')

if uploaded_file:
    # Save the uploaded file to a temporary location
    temp_path = os.path.join("temp", uploaded_file.name)
    with open(temp_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())

    lch.vectorDB_for_embeddings(temp_path)
    vec_db = lch.instance_Qdrant()
    response = lch.get_query_from_user(vec_db, user_query)
    st.subheader('Answer:')
    st.text(textwrap.fill(response, width=80))

    # Remove the temporary file after processing (optional)
    os.remove(temp_path)
