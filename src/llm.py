
import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

def clean_report(text):

    # If Gemini returns list
    if isinstance(text, list):

        text = " ".join(
            str(item)
            for item in text
        )


    # Convert anything else to string

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

def get_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2,
        google_api_key=os.getenv(
            "GOOGLE_API_KEY"
        )
    )

    return llm


def generate_business_insights(
        dataset_summary,
        model_results,
        shap_summary
):

    llm = get_llm()


    prompt = f"""

You are an expert Data Scientist presenting results to business executives.

Analyze the ML results below.

Dataset Summary:
{dataset_summary}


Model Performance:
{model_results}


SHAP Important Features:
{shap_summary}


Generate a professional business report.

IMPORTANT FORMATTING RULES:

- Return ONLY the report.
- Do NOT use markdown symbols like #, ##, **.
- Do NOT include technical code.
- Use clear section titles.
- Use bullet points.
- Keep sentences short and professional.


Use exactly this structure:


EXECUTIVE SUMMARY

Write 3-4 sentences explaining the overall findings.


KEY DATA INSIGHTS

- Point 1
- Point 2
- Point 3


MODEL PERFORMANCE

Explain the best model and performance.


IMPORTANT DRIVERS

Explain the top SHAP features and their business impact.


BUSINESS RECOMMENDATIONS

- Recommendation 1
- Recommendation 2
- Recommendation 3

IMPORTANT:
Do not mention caching, cached content, system instructions, or internal processing.
Return only the business report.

FINAL CONCLUSION

Give a short conclusion.


"""


    response = llm.invoke(prompt)


    return clean_report(
    response.content
)