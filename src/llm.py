import os

import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI



# Load local environment variables

load_dotenv()



def clean_report(text):

    """
    Clean Gemini response formatting
    """


    # Gemini sometimes returns list

    if isinstance(text, list):

        text = " ".join(
            str(item)
            for item in text
        )


    text = str(text)


    replacements = {

        "\\n": "\n",

        "\\": "",

        "**": "",

        "###": "",

        "##": "",

        "#": ""

    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    return text.strip()




def get_api_key():

    """
    Get Gemini API key from:
    
    1. Streamlit Secrets (Cloud)
    2. .env file (Local)
    
    """


    try:

        if "GOOGLE_API_KEY" in st.secrets:

            return st.secrets[
                "GOOGLE_API_KEY"
            ]

    except Exception:

        pass



    return os.getenv(
        "GOOGLE_API_KEY"
    )




def get_llm():

    """
    Initialize Gemini model
    """


    api_key = get_api_key()


    if api_key is None:

        raise ValueError(
            "Google API Key not found"
        )


    llm = ChatGoogleGenerativeAI(

        model="gemini-3.5-flash",

        temperature=0.2,

        google_api_key=api_key

    )


    return llm




def generate_business_insights(
        dataset_summary,
        model_results,
        shap_summary
):


    llm = get_llm()



    prompt = f"""

You are an expert Data Scientist acting as an AI Business Analyst.

Analyze the machine learning results below and create a professional business report.


Dataset Summary:

{dataset_summary}


Model Performance:

{model_results}


SHAP Feature Importance:

{shap_summary}



Create the report using this structure:



EXECUTIVE SUMMARY

Explain the overall findings in 3-4 sentences.



KEY DATA INSIGHTS

- Mention important dataset observations
- Mention important patterns
- Mention useful business observations



MODEL PERFORMANCE

Explain:
- Best performing model
- Performance score
- Why it is suitable



IMPORTANT DRIVERS

Explain the top SHAP features.

For each feature explain:
- What it means
- How it impacts predictions
- Business interpretation



BUSINESS RECOMMENDATIONS

Give practical actions:

- Recommendation 1
- Recommendation 2
- Recommendation 3



FINAL CONCLUSION

Give a short professional conclusion.



Formatting rules:

- Return only the report.
- Do not mention AI instructions.
- Do not mention caching.
- Do not mention system messages.
- Do not use markdown symbols.
- Use clean headings.
- Use simple business language.

"""


    response = llm.invoke(
        prompt
    )


    return clean_report(
        response.content
    )