from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os


def generate_report(df, ml_result):
    doc = SimpleDocTemplate("automl_report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AutoML Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Rows: {df.shape[0]}", styles["Normal"]))
    content.append(Paragraph(f"Columns: {df.shape[1]}", styles["Normal"]))

    content.append(Spacer(1, 12))

    content.append(Paragraph("Model Results:", styles["Heading2"]))

    for model, metrics in ml_result["results"].items():
        text = f"{model}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
        content.append(Paragraph(text, styles["Normal"]))

    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Best Model: {ml_result['best_model']}", styles["Heading2"]))

    if os.path.exists("model_comparison.png"):
        content.append(Image("model_comparison.png", width=400, height=300))

    doc.build(content)

    return "automl_report.pdf"