from tempfile import NamedTemporaryFile
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import asyncio
import streamlit as st
# from flask import Flask,request, jsonify
import os
import pandas as pd
import nltk
from nltk import word_tokenize
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

Settings.embed_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
evaluator = SemanticSimilarityEvaluator()

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')



def preprocessing(document):
    # preprocessed_text_final=[]
    # for i in document:
    text1 = document.replace('\n', '').replace('\t', '').lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text1)  # Remove non-ASCII characters
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]  # Remove punctuation
    tokens = [token for token in tokens if token]  # Remove empty tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    preprocessed_text = ' '.join(filtered_tokens)
    # preprocessed_text_final.append(preprocessed_text)
    print('preprocess',preprocessed_text)
    return preprocessed_text

async def input(jd,extra,resume):
    score_dict ={}
    try:
        print('jd',jd)
        docc = SimpleDirectoryReader(input_files=[jd]).load_data()[0].text+extra
        # docc = preprocessing(docc[0].text+extra)
        # docc = docc[0].text+extra

        for i in resume:
            file_name = os.path.basename(i)
            print('i',i)
            doccc = SimpleDirectoryReader(input_files=[i]).load_data()[0].text
            # doccc = preprocessing(doccc)
            result = await evaluator.aevaluate(response=doccc,reference=docc)
            print("result: ", result)
            score_dict[file_name]=result.score

        sorted_dict_desc = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(list(sorted_dict_desc.items()), columns=['Resume', 'Score'])
        print(df)
        return df
    except Exception as e:
        print('cant find the file')


def debug_app():
    st.markdown('''# TalentMatch360 🌟''')
                
    st.markdown('''### Nextgen tool for evaluating Job Descriptions and Resumes''')

    left_column, center_column, right_column = st.columns(3)

    with left_column:
        file_path_jd=""
        filepath_jd=""
        JD_folder_path = r"D:\Rohit\jdcv_score_app\jdcv_score_app\temp3"
        
        clear_folder(JD_folder_path)
        uploaded_jd_file = st.file_uploader("Upload your JD here")
        extra = st.text_input("extra details")
        print('extra',extra)
        os.makedirs(JD_folder_path,exist_ok=True)
        print('done')
        if uploaded_jd_file is not None:
            try:
                # JD_embedding = jd_embedding(uploaded_jd_file)[0]
                # JD_embedding1 = jd_embedding(uploaded_jd_file)
                # print('JD 000',JD_embedding)
                file_path_jd = os.path.join(JD_folder_path, uploaded_jd_file.name)
                print('FILE',file_path_jd)
                with open(file_path_jd, "wb") as f:
                    f.write(uploaded_jd_file.getbuffer())
                filepath_jd=uploaded_jd_file.name
                print('u',file_path_jd)
                st.write("JD uploaded")
            except Exception as e:
                st.error(f"Error processing JD: {e}")
    
    with right_column:
        RESUME_folder_path = r'D:/Rohit/jdcv_score_app/jdcv_score_app/temp4/'
        os.makedirs(RESUME_folder_path,exist_ok=True)

        uploaded_resume_files = st.file_uploader(
            "Upload all of the resumes", accept_multiple_files=True)
        
        resume_embeddings = [] 
        if uploaded_resume_files:
            for i in uploaded_resume_files:
                if i is not None:
                    try:
                        file_path = os.path.join(RESUME_folder_path, i.name)
                        print('FILE',file_path)
                        with open(file_path, "wb") as f:   
                            f.write(i.getbuffer()) 
                            resume_embeddings.append(file_path)
                        print('ttt',resume_embeddings)
                        st.write("Resume uploaded")
                    except Exception as e:
                        st.error(f"Error processing resume {i.name}: {e}")
                        st.write("Resume can't upload")
            

    with center_column:
        # score_dict = {}
        # score_dict1={}
        score_df = asyncio.run(input(file_path_jd,extra, resume_embeddings))
        
        st.dataframe(score_df, use_container_width=True, width=1200,hide_index=True)
        # st.dataframe(df1, use_container_width=True, width=1200,hide_index=True)

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=3001, debug=True)
    debug_app()