from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak
)

from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

from datetime import datetime
import os
import re


class ThyroGuidePDFReport:

    def __init__(self):

        self.styles = getSampleStyleSheet()

        self.title_style = ParagraphStyle(
            "ReportTitle",
            parent=self.styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=8
        )

        self.subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.grey,
            spaceAfter=15
        )

        self.section_style = ParagraphStyle(
            "Section",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=12,
            spaceAfter=6
        )

        self.body_style = ParagraphStyle(
            "Body",
            parent=self.styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            spaceAfter=8
        )

        self.small_style = ParagraphStyle(
            "Small",
            parent=self.styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.grey
        )


    def _clean_text(self, text):

        if text is None:

            return ""

        text = str(text)

        text = text.replace(
            "\n\n\n",
            "\n\n"
        )

        return text.strip()

    def _calculate_nodule_size(self, cv_json):

        bbox = cv_json.get(
            "bounding_box"
        )

        if not bbox:

            return None


        xmin = bbox.get(
            "xmin"
        )

        ymin = bbox.get(
            "ymin"
        )

        xmax = bbox.get(
            "xmax"
        )

        ymax = bbox.get(
            "ymax"
        )


        if None in [
            xmin,
            ymin,
            xmax,
            ymax
        ]:

            return None


        width = xmax - xmin

        height = ymax - ymin


        return {

            "width_pixels": width,

            "height_pixels": height

        }
    def _create_ai_summary(self, cv_json, gemma_report):

        summary_parts = []

        # ------------------------------------------------
        # Detection summary
        # ------------------------------------------------

        if cv_json.get("nodule_detected", False):

            detection_confidence = (
                cv_json.get(
                    "detection_confidence",
                    0
                ) * 100
            )

            summary_parts.append(

                f"A thyroid nodule was detected "
                f"with a detection confidence of "
                f"{detection_confidence:.2f}%."
            )

        else:

            summary_parts.append(

                "No thyroid nodule was detected "
                "by the computer vision system."
            )

            return " ".join(summary_parts)


    # ------------------------------------------------
    # Classification summary
    # ------------------------------------------------

        classification = cv_json.get(

            "classification",

            "Not available"

        )

        classification_confidence = (

            cv_json.get(

                "classification_confidence",

                0

            ) * 100

        )


        summary_parts.append(

            f"The CNN classification result was "
            f"<b>{classification}</b> with a confidence "
            f"of {classification_confidence:.2f}%."
        )


    # ------------------------------------------------
    # Gemma summary
    # ------------------------------------------------

        if gemma_report:

            gemma_text = str(
                gemma_report
            )


            # Remove Markdown symbols

            gemma_text = re.sub(

                r"[*#]+",

                "",

                gemma_text

            )


            # Extract the first useful section

            important_sections = [

                "Classification Summary",

                "Confidence Interpretation",

                "Patient-Friendly Explanation",

                "Detection Summary"

            ]


            for section_name in important_sections:

                pattern = (

                    re.escape(section_name)

                    + r"\s*:?\s*"

                    + r"(.*?)"

                    + r"(?="

                    + r"Detection Summary|"

                    + r"Classification Summary|"

                    + r"Confidence Interpretation|"

                    + r"Explainability Summary|"

                    + r"Patient-Friendly Explanation|"

                    + r"Important Limitations"

                    + r"|$)"

                )


                match = re.search(

                    pattern,

                    gemma_text,

                    flags=re.DOTALL |

                    re.IGNORECASE

                )


                if match:

                    gemma_summary = (

                        self._clean_text(

                            match.group(1)

                        )

                    )


                    if gemma_summary:

                        summary_parts.append(

                            gemma_summary

                        )

                        break


        return " ".join(
            summary_parts
        )
    def _parse_gemma_report(self, report):

        sections = {}

        if not report:
            return sections

        section_names = [

            "Detection Summary",

            "Classification Summary",

            "Confidence Interpretation",

            "Explainability Summary",

            "Patient-Friendly Explanation",

            "Important Limitations"

        ]

        # --------------------------------------------------------
        # Normalize Markdown formatting
        # --------------------------------------------------------

        report = str(report)

        report = re.sub(
            r"[*#]+",
            "",
            report
        )

        report = report.strip()


        # --------------------------------------------------------
        # Extract each section
        # --------------------------------------------------------

        pattern = (

            r"("

            + "|".join(

                re.escape(name)

                for name in section_names

            )

            + r")"

            r"\s*:?\s*"

            r"(.*?)"

            r"(?="

            + "|".join(

                re.escape(name)

                for name in section_names

            )

            + r"\s*:?"

            + r"|$)"

        )


        matches = re.findall(

            pattern,

            report,

            flags=re.DOTALL | re.IGNORECASE

        )


        for title, content in matches:

            sections[title] = self._clean_text(
                content
            )


        return sections
#
    def generate_pdf(
        self,
        output_path,
        patient_info,
        cv_json,
        gemma_report
    ):

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        document = SimpleDocTemplate(

            output_path,

            pagesize=A4,

            rightMargin=45,

            leftMargin=45,

            topMargin=40,

            bottomMargin=40

        )

        story = []


        # ==================================================
        # HEADER
        # ==================================================

        story.append(
            Paragraph(
                "THYROGUIDE AI",
                self.title_style
            )
        )

        story.append(
            Paragraph(
                "Explainable AI Assistant for Thyroid Ultrasound Analysis",
                self.subtitle_style
            )
        )

        story.append(
            HRFlowable(
                width="100%",
                thickness=1,
                color=colors.black
            )
        )

        story.append(
            Spacer(
                1,
                12
            )
        )


        # ==================================================
        # REPORT INFORMATION
        # ==================================================

        story.append(
            Paragraph(
                "THYROID ULTRASOUND AI-ASSISTED REPORT",
                self.section_style
            )
        )

        report_date = datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )

        metadata = [

            [
                Paragraph(
                    "<b>Report Generated:</b>",
                    self.body_style
                ),

                Paragraph(
                    report_date,
                    self.body_style
                )

            ],

            [

                Paragraph(
                    "<b>Report Generated By:</b>",
                    self.body_style
                ),

                Paragraph(
                    "ThyroGuide AI",
                    self.body_style
                )

            ]

        ]

        metadata_table = Table(
            metadata,
            colWidths=[
                1.7 * inch,
                4.8 * inch
            ]
        )

        metadata_table.setStyle(
            TableStyle(
                [

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),

                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        0
                    ),

                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    )

                ]
            )
        )

        story.append(
            metadata_table
        )


        # ==================================================
        # PATIENT INFORMATION
        # ==================================================

        story.append(
            Paragraph(
                "PATIENT INFORMATION",
                self.section_style
            )
        )

        patient_table_data = [

            [

                Paragraph(
                    "<b>Patient Name</b>",
                    self.body_style
                ),

                Paragraph(
                    str(
                        patient_info.get(
                            "patient_name",
                            "Not provided"
                        )
                    ),

                    self.body_style
                )

            ],

            [

                Paragraph(
                    "<b>Age</b>",
                    self.body_style
                ),

                Paragraph(
                    str(
                        patient_info.get(
                            "age",
                            "Not provided"
                        )
                    ),

                    self.body_style
                )

            ],

            [

                Paragraph(
                    "<b>Gender</b>",
                    self.body_style
                ),

                Paragraph(
                    str(
                        patient_info.get(
                            "gender",
                            "Not provided"
                        )
                    ),

                    self.body_style
                )

            ]

        ]

        patient_table = Table(

            patient_table_data,

            colWidths=[
                1.7 * inch,
                4.8 * inch
            ]

        )

        patient_table.setStyle(

            TableStyle(

                [

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.lightgrey
                    ),

                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.whitesmoke
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    )

                ]

            )

        )

        story.append(
            patient_table
        )


        # ==================================================
        # CLINICAL INFORMATION
        # ==================================================

        clinical_info = patient_info.get(
            "clinical_information",
            "Not provided"
        )

        story.append(
            Paragraph(
                "CLINICAL INFORMATION",
                self.section_style
            )
        )

        story.append(
            Paragraph(
                clinical_info,
                self.body_style
            )
        )


        # ==================================================
        # FINDINGS
        # ==================================================

        story.append(
            Paragraph(
                "AI-ASSISTED FINDINGS",
                self.section_style
            )
        )

        if cv_json.get(
            "nodule_detected",
            False
        ):

            finding_text = (

                "A thyroid nodule was detected in "
                "the uploaded ultrasound image by "
                "the computer vision detection system."
            )

        else:

            finding_text = (

                "No thyroid nodule was detected by "
                "the computer vision detection system."
            )

        story.append(
            Paragraph(
                finding_text,
                self.body_style
            )
        )

        # ==================================================
        # NODULE SIZE
        # ==================================================

        nodule_size = self._calculate_nodule_size(
            cv_json
        )
        if nodule_size:
            story.append(
                Paragraph(
                    "ESTIMATED NODULE SIZE",
                    self.section_style
                )
            )
            size_data = [
                [
                    Paragraph(
                        "<b>Width</b>",
                        self.body_style
                    ),
                    Paragraph(
                        f"{nodule_size['width_pixels']:.1f} pixels",
                        self.body_style
                    )
                ],
                [
                    Paragraph(
                        "<b>Height</b>",
                        self.body_style
                    ),
                    Paragraph(
                        f"{nodule_size['height_pixels']:.1f} pixels",
                        self.body_style
                    )
                ]
            ]
            size_table = Table(

                size_data,

                colWidths=[

                    2.4 * inch,

                    4.1 * inch

                ]

            )
            size_table.setStyle(
                TableStyle(
                    [
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.lightgrey
                        ),
                        (
                            "BACKGROUND",
                            (0, 0),
                            (0, -1),
                            colors.whitesmoke
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP"
                        )
                    ]
                )
            )


            story.append(
                size_table
            )
        # ==================================================
        # AI ANALYSIS
        # ==================================================

        story.append(
            Paragraph(
                "COMPUTER VISION ANALYSIS",
                self.section_style
            )
        )

        analysis_data = [

            [

                Paragraph(
                    "<b>Detection Status</b>",
                    self.body_style
                ),

                Paragraph(
                    (
                        "Detected"
                        if cv_json.get(
                            "nodule_detected",
                            False
                        )
                        else "Not Detected"
                    ),

                    self.body_style
                )

            ],

            [

                Paragraph(
                    "<b>Detection Confidence</b>",
                    self.body_style
                ),

                Paragraph(
                    f"{cv_json.get('detection_confidence', 0) * 100:.2f}%",

                    self.body_style
                )

            ],

            [

                Paragraph(
                    "<b>CNN Classification</b>",
                    self.body_style
                ),

                Paragraph(
                    str(
                        cv_json.get(
                            "classification",
                            "Not available"
                        )
                    ),

                    self.body_style
                )

            ],

            [

                Paragraph(
                    "<b>Classification Confidence</b>",
                    self.body_style
                ),

                Paragraph(
                    f"{cv_json.get('classification_confidence', 0) * 100:.2f}%",

                    self.body_style
                )

            ]

        ]

        analysis_table = Table(

            analysis_data,

            colWidths=[
                2.4 * inch,
                4.1 * inch
            ]

        )

        analysis_table.setStyle(

            TableStyle(

                [

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.lightgrey
                    ),

                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.whitesmoke
                    )

                ]

            )

        )

        story.append(
            analysis_table
        )


        # ==================================================
        # GEMMA REPORT
        # ==================================================
        story.append(Paragraph("AI-ASSISTED SUMMARY",self.section_style))
        ai_summary = self._create_ai_summary(cv_json,gemma_report)
        story.append(Paragraph(ai_summary,self.body_style))
       
        # ==================================================
        # DISCLAIMER
        # ==================================================
        story.append(
            Spacer(
                1,
                10
            )
        )
        story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.grey
            )
        )
        story.append(
            Spacer(
                1,
                8
            )
        )
        disclaimer = (
            "<b>IMPORTANT DISCLAIMER:</b> "
            "This report contains AI-assisted analysis "
            "generated for research and decision-support "
            "purposes. The results are not a definitive "
            "medical diagnosis and should not replace "
            "evaluation by a qualified healthcare professional."
        )
        story.append(
            Paragraph(
                disclaimer,
                self.small_style
            )
        )
        # ==================================================
        # FOOTER
        # ==================================================

        story.append(
            Spacer(
                1,
                15
            )
        )
        story.append(
            Paragraph(
                "Report generated by ThyroGuide AI",
                self.small_style
            )
        )
        story.append(
            Paragraph(
                "Explainable AI Assistant for Thyroid Ultrasound Analysis",
                self.small_style
            )
        )
        document.build(
            story
        )

        return output_path