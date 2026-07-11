
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.pagesizes import letter

from reportlab.lib.units import inch

import os



def create_pdf_report(
        dataset_summary,
        model_results,
        shap_summary,
        ai_report
):

    """
    Generate AI Data Scientist PDF Report
    """


    file_path = "AI_Data_Scientist_Report.pdf"


    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter
    )


    styles = getSampleStyleSheet()


    story = []


    # Title

    title = Paragraph(
        "AI Data Scientist Report",
        styles["Title"]
    )

    story.append(title)

    story.append(
        Spacer(1,20)
    )


    # Dataset Section

    story.append(
        Paragraph(
            "1. Dataset Overview",
            styles["Heading2"]
        )
    )


    story.append(
        Paragraph(
            str(dataset_summary),
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1,20)
    )


    # Model Performance

    story.append(
        Paragraph(
            "2. Model Performance",
            styles["Heading2"]
        )
    )


    story.append(
        Paragraph(
            str(model_results),
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1,20)
    )


    # SHAP Explanation

    story.append(
        Paragraph(
            "3. Model Explainability (SHAP)",
            styles["Heading2"]
        )
    )


    story.append(
        Paragraph(
            str(shap_summary),
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1,20)
    )


    # Gemini Report

    story.append(
        Paragraph(
            "4. AI Business Analyst Insights",
            styles["Heading2"]
        )
    )


    # Replace line breaks

    ai_report = ai_report.replace(
        "\n",
        "<br/>"
    )


    story.append(
        Paragraph(
            ai_report,
            styles["BodyText"]
        )
    )


    doc.build(
        story
    )


    return file_path