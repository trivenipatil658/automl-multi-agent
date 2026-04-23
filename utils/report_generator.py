from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os


def generate_report(df, ml_result):
    # Create a PDF document at the fixed output path
    doc = SimpleDocTemplate("automl_report.pdf")

    # Load default ReportLab paragraph styles (Title, Heading2, Normal, etc.)
    styles = getSampleStyleSheet()

    # List of content elements to be added to the PDF in order
    content = []

    # Add the report title at the top
    content.append(Paragraph("AutoML Report", styles["Title"]))
    content.append(Spacer(1, 12))  # vertical spacing

    # Add dataset shape information
    content.append(Paragraph(f"Rows: {df.shape[0]}", styles["Normal"]))
    content.append(Paragraph(f"Columns: {df.shape[1]}", styles["Normal"]))

    content.append(Spacer(1, 12))

    # Add a section heading for model results
    content.append(Paragraph("Model Results:", styles["Heading2"]))

    # Add one line per model showing accuracy and F1 score
    for model, metrics in ml_result["results"].items():
        text = f"{model}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
        content.append(Paragraph(text, styles["Normal"]))

    content.append(Spacer(1, 12))

    # Highlight the best model selected by the pipeline
    content.append(Paragraph(f"Best Model: {ml_result['best_model']}", styles["Heading2"]))

    # Embed the model comparison bar chart if it was saved to disk
    if os.path.exists("model_comparison.png"):
        content.append(Image("model_comparison.png", width=400, height=300))

    # Build and write the PDF file to disk
    doc.build(content)

    # Return the file path so the UI can offer it as a download
    return "automl_report.pdf"
