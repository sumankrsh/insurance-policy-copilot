"""
Generate the four sample policy PDFs in data/policy_documents/.

The documents are fictional. The PDFs are committed, so you only need to run this
(python -m data.policy_document_generator, needs reportlab from requirements-dev.txt)
if you change the wording.
"""

import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class PolicyDocumentGenerator:
    """Generate sample insurance policy PDFs"""

    def __init__(self, output_dir="data/policy_documents"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(
            ParagraphStyle(
                name="PolicyTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1e40af"),
                alignment=TA_CENTER,
                spaceAfter=30,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#1e40af"),
                spaceBefore=20,
                spaceAfter=12,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="PolicyText", parent=self.styles["Normal"], fontSize=11, alignment=TA_JUSTIFY, spaceAfter=12
            )
        )

    def generate_term_life_policy(self):
        """Generate Term Life Insurance Policy"""
        doc = SimpleDocTemplate(
            f"{self.output_dir}/term_life_insurance_policy.pdf",
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        story = []

        # Title
        story.append(Paragraph("TERM LIFE INSURANCE POLICY", self.styles["PolicyTitle"]))
        story.append(Spacer(1, 0.2 * inch))

        # Policy Details Section
        story.append(Paragraph("SECTION 1: POLICY DETAILS", self.styles["SectionHeading"]))

        policy_text = """
        This Term Life Insurance Policy ("Policy") is a contract between the Policyholder and
        Insurance Company Ltd. ("Company"). This Policy provides life insurance coverage for the
        specified term, subject to the terms and conditions stated herein.
        """
        story.append(Paragraph(policy_text, self.styles["PolicyText"]))

        # Coverage Details
        story.append(Paragraph("SECTION 2: COVERAGE DETAILS", self.styles["SectionHeading"]))

        coverage_data = [
            ["Coverage Type", "Term Life Insurance"],
            ["Policy Term", "10, 15, 20, 25, or 30 years"],
            ["Sum Assured", "As specified in Policy Schedule"],
            ["Premium Payment", "Monthly, Quarterly, Semi-Annual, Annual"],
            ["Grace Period", "30 days from premium due date"],
            ["Revival Period", "Within 2 years of lapse"],
        ]

        coverage_table = Table(coverage_data, colWidths=[2.5 * inch, 3.5 * inch])
        coverage_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(coverage_table)
        story.append(Spacer(1, 0.2 * inch))

        # Surrender Value
        story.append(Paragraph("SECTION 3: SURRENDER VALUE AND CHARGES", self.styles["SectionHeading"]))

        surrender_text = """
        <b>3.1 Surrender Charges:</b><br/>
        If you surrender your policy before the completion of the lock-in period,
        surrender charges will apply as follows:
        <br/><br/>
        • Year 1: 10% of total premiums paid<br/>
        • Year 2: 8% of total premiums paid<br/>
        • Year 3: 6% of total premiums paid<br/>
        • Year 4: 4% of total premiums paid<br/>
        • Year 5 onwards: No surrender charge<br/>
        <br/>
        <b>3.2 Lock-in Period:</b><br/>
        The policy has a lock-in period of 5 years. Surrender during this period will
        result in the charges mentioned above.
        <br/><br/>
        <b>3.3 Surrender Value Calculation:</b><br/>
        Surrender Value = Total Premiums Paid - Surrender Charges - Outstanding Loans (if any)
        """
        story.append(Paragraph(surrender_text, self.styles["PolicyText"]))

        # Exclusions
        story.append(Paragraph("SECTION 4: EXCLUSIONS", self.styles["SectionHeading"]))

        exclusions_text = """
        The Company shall not be liable to pay the death benefit if death occurs due to:
        <br/><br/>
        • Suicide within 12 months from policy commencement<br/>
        • Participation in hazardous activities without disclosure<br/>
        • War, terrorism, or nuclear contamination<br/>
        • Criminal activities or illegal acts<br/>
        • Pre-existing conditions not disclosed at the time of application<br/>
        """
        story.append(Paragraph(exclusions_text, self.styles["PolicyText"]))

        # Claims Process
        story.append(Paragraph("SECTION 5: CLAIMS PROCESS", self.styles["SectionHeading"]))

        claims_text = """
        <b>5.1 Notification:</b><br/>
        The nominee must notify the Company within 30 days of the insured's death.
        <br/><br/>
        <b>5.2 Required Documents:</b><br/>
        • Original policy document<br/>
        • Death certificate<br/>
        • Medical records (if applicable)<br/>
        • Police FIR (in case of unnatural death)<br/>
        • Nominee identification proof<br/>
        • Claim form duly filled<br/>
        <br/>
        <b>5.3 Settlement Timeline:</b><br/>
        Claims will be settled within 30 days of receipt of all required documents.
        """
        story.append(Paragraph(claims_text, self.styles["PolicyText"]))

        # Terms and Conditions
        story.append(PageBreak())
        story.append(Paragraph("SECTION 6: TERMS AND CONDITIONS", self.styles["SectionHeading"]))

        terms_text = """
        <b>6.1 Premium Payment:</b><br/>
        Premiums must be paid on or before the due date. A grace period of 30 days is provided.
        <br/><br/>
        <b>6.2 Policy Lapse:</b><br/>
        If premiums are not paid within the grace period, the policy will lapse.
        <br/><br/>
        <b>6.3 Policy Revival:</b><br/>
        Lapsed policies can be revived within 2 years by paying all due premiums with interest.
        <br/><br/>
        <b>6.4 Nomination:</b><br/>
        The policyholder can nominate beneficiaries who will receive the death benefit.
        <br/><br/>
        <b>6.5 Assignment:</b><br/>
        The policy can be assigned to another party with written notice to the Company.
        <br/><br/>
        <b>6.6 Loans:</b><br/>
        Policy loans are available after 3 years at prevailing interest rates.
        <br/><br/>
        <b>6.7 Free-Look Period:</b><br/>
        The policyholder has 15 days from policy receipt to review and return if not satisfied.
        """
        story.append(Paragraph(terms_text, self.styles["PolicyText"]))

        # Build PDF
        doc.build(story)
        print(f"✓ Generated: {self.output_dir}/term_life_insurance_policy.pdf")

    def generate_ulip_policy(self):
        """Generate ULIP Policy Document"""
        doc = SimpleDocTemplate(f"{self.output_dir}/ulip_insurance_policy.pdf", pagesize=letter)

        story = []

        story.append(Paragraph("UNIT LINKED INSURANCE PLAN (ULIP) POLICY", self.styles["PolicyTitle"]))
        story.append(Spacer(1, 0.2 * inch))

        # ULIP Specific Content
        story.append(Paragraph("SECTION 1: ULIP FEATURES", self.styles["SectionHeading"]))

        ulip_text = """
        This Unit Linked Insurance Plan combines investment and insurance, offering:
        <br/><br/>
        • Life cover with investment in market-linked funds<br/>
        • Choice of Equity, Debt, and Balanced funds<br/>
        • Fund switching options (4 free switches per year)<br/>
        • Partial withdrawal after 5 years<br/>
        • Top-up premium facility<br/>
        """
        story.append(Paragraph(ulip_text, self.styles["PolicyText"]))

        # Fund Options
        story.append(Paragraph("SECTION 2: FUND OPTIONS", self.styles["SectionHeading"]))

        fund_data = [
            ["Fund Name", "Risk Level", "Asset Allocation", "Management Fee"],
            ["Equity Growth Fund", "High", "80-100% Equity", "1.35% p.a."],
            ["Balanced Fund", "Medium", "40-60% Equity, 40-60% Debt", "1.25% p.a."],
            ["Debt Fund", "Low", "100% Debt Securities", "1.00% p.a."],
            ["Liquid Fund", "Very Low", "Money Market Instruments", "0.75% p.a."],
        ]

        fund_table = Table(fund_data, colWidths=[2 * inch, 1.2 * inch, 2 * inch, 1.3 * inch])
        fund_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(fund_table)
        story.append(Spacer(1, 0.2 * inch))

        # Charges
        story.append(Paragraph("SECTION 3: CHARGES STRUCTURE", self.styles["SectionHeading"]))

        charges_text = """
        <b>3.1 Premium Allocation Charge:</b><br/>
        • Year 1: 8% of annual premium<br/>
        • Year 2: 6% of annual premium<br/>
        • Year 3: 4% of annual premium<br/>
        • Year 4 onwards: 2% of annual premium<br/>
        <br/>
        <b>3.2 Fund Management Charge:</b> 0.75% to 1.35% p.a. (varies by fund)<br/>
        <b>3.3 Policy Administration Charge:</b> INR 50 per month<br/>
        <b>3.4 Mortality Charge:</b> As per age and sum assured<br/>
        <b>3.5 Surrender Charge:</b> Nil after 5 years<br/>
        <b>3.6 Partial Withdrawal Charge:</b> Nil (after 5 years)<br/>
        <b>3.7 Fund Switch Charge:</b> Free for first 4 switches, INR 100 thereafter<br/>
        """
        story.append(Paragraph(charges_text, self.styles["PolicyText"]))

        # Lock-in and Liquidity
        story.append(Paragraph("SECTION 4: LOCK-IN PERIOD AND LIQUIDITY", self.styles["SectionHeading"]))

        lockin_text = """
        <b>4.1 Lock-in Period:</b> 5 years from policy commencement<br/>
        <b>4.2 Partial Withdrawals:</b> Allowed after 5 years, maximum 20% of fund value per year<br/>
        <b>4.3 Surrender:</b> Allowed after 5 years with no surrender charges<br/>
        <b>4.4 Premium Discontinuance:</b> Policy moves to discontinuance fund earning 4% p.a.<br/>
        """
        story.append(Paragraph(lockin_text, self.styles["PolicyText"]))

        doc.build(story)
        print(f"✓ Generated: {self.output_dir}/ulip_insurance_policy.pdf")

    def generate_health_policy(self):
        """Generate Health Insurance Policy"""
        doc = SimpleDocTemplate(f"{self.output_dir}/health_insurance_policy.pdf", pagesize=letter)

        story = []

        story.append(Paragraph("HEALTH INSURANCE POLICY", self.styles["PolicyTitle"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("SECTION 1: COVERAGE BENEFITS", self.styles["SectionHeading"]))

        benefits_text = """
        • Hospitalization expenses (Room, ICU, Surgeon fees)<br/>
        • Pre-hospitalization (60 days) and Post-hospitalization (90 days)<br/>
        • Day-care procedures (150+ procedures covered)<br/>
        • Ambulance charges (up to INR 5,000 per hospitalization)<br/>
        • Annual health check-up<br/>
        • No Claim Bonus: 10% increase in sum insured for claim-free year<br/>
        """
        story.append(Paragraph(benefits_text, self.styles["PolicyText"]))

        # Waiting Periods
        story.append(Paragraph("SECTION 2: WAITING PERIODS", self.styles["SectionHeading"]))

        waiting_text = """
        • Initial waiting period: 30 days (except accident)<br/>
        • Pre-existing diseases: 3 years<br/>
        • Specific diseases (hernia, cataract, etc.): 2 years<br/>
        • Maternity benefits: 2 years<br/>
        """
        story.append(Paragraph(waiting_text, self.styles["PolicyText"]))

        # Exclusions
        story.append(Paragraph("SECTION 3: EXCLUSIONS", self.styles["SectionHeading"]))

        exclusions_text = """
        Treatment/expenses not covered:<br/>
        • Cosmetic/aesthetic treatments<br/>
        • Dental treatment (unless accident-related)<br/>
        • Alternative treatments (unless specified)<br/>
        • Self-inflicted injuries<br/>
        • Treatment arising from substance abuse<br/>
        • Experimental/unproven treatments<br/>
        """
        story.append(Paragraph(exclusions_text, self.styles["PolicyText"]))

        doc.build(story)
        print(f"✓ Generated: {self.output_dir}/health_insurance_policy.pdf")

    def generate_sample_terms_conditions(self):
        """Generate General Terms and Conditions Document"""
        doc = SimpleDocTemplate(f"{self.output_dir}/general_terms_conditions.pdf", pagesize=letter)

        story = []

        story.append(Paragraph("GENERAL TERMS AND CONDITIONS", self.styles["PolicyTitle"]))
        story.append(Spacer(1, 0.2 * inch))

        sections = [
            (
                "DEFINITIONS",
                """
            • "Company": Insurance Company Ltd<br/>
            • "Policyholder": Person who owns the policy<br/>
            • "Insured": Person whose life/health is covered<br/>
            • "Premium": Amount paid for insurance coverage<br/>
            • "Sum Assured": Amount payable on claim<br/>
            • "Nominee": Person designated to receive benefits<br/>
            """,
            ),
            (
                "PREMIUM PAYMENT CONDITIONS",
                """
            • Premiums must be paid on due date<br/>
            • Grace period of 30 days for all payment modes<br/>
            • Policy lapses if premium not paid within grace period<br/>
            • Lapsed policies can be revived within 2 years<br/>
            • Auto-debit facility available for convenience<br/>
            """,
            ),
            (
                "CLAIM SETTLEMENT PROCESS",
                """
            • Notify within 30 days of claim event<br/>
            • Submit all required documents<br/>
            • Company may investigate claims<br/>
            • Settlement within 30 days of document receipt<br/>
            • Rejection reasons will be communicated in writing<br/>
            """,
            ),
            (
                "POLICY SERVICING",
                """
            • Changes require written request<br/>
            • Address/contact updates within 15 days<br/>
            • Nomination changes allowed anytime<br/>
            • Policy documents available online<br/>
            • Annual statements sent to registered email<br/>
            """,
            ),
            (
                "GRIEVANCE REDRESSAL",
                """
            • Level 1: Customer Service (7 days)<br/>
            • Level 2: Grievance Officer (15 days)<br/>
            • Level 3: Insurance Ombudsman (30 days)<br/>
            • Email: grievance@example.com<br/>
            • Toll-free: 1800-XXX-XXXX<br/>
            """,
            ),
            (
                "REGULATORY COMPLIANCE",
                """
            • Policy governed by Insurance Act and IRDAI regulations<br/>
            • KYC documents mandatory as per regulations<br/>
            • Tax benefits under Section 80C and 80D<br/>
            • GST applicable as per prevailing rates<br/>
            • Subject to jurisdiction of courts in Mumbai<br/>
            """,
            ),
        ]

        for heading, content in sections:
            story.append(Paragraph(f"SECTION: {heading}", self.styles["SectionHeading"]))
            story.append(Paragraph(content, self.styles["PolicyText"]))
            story.append(Spacer(1, 0.1 * inch))

        doc.build(story)
        print(f"✓ Generated: {self.output_dir}/general_terms_conditions.pdf")

    def generate_all_documents(self):
        """Generate all policy documents"""
        print("Generating policy documents...")
        self.generate_term_life_policy()
        self.generate_ulip_policy()
        self.generate_health_policy()
        self.generate_sample_terms_conditions()

        # Create a metadata file for RAG indexing
        metadata = {
            "documents": [
                {
                    "filename": "term_life_insurance_policy.pdf",
                    "title": "Term Life Insurance Policy",
                    "type": "policy",
                    "category": "life_insurance",
                    "keywords": ["term life", "death benefit", "surrender", "premium", "nomination"],
                },
                {
                    "filename": "ulip_insurance_policy.pdf",
                    "title": "Unit Linked Insurance Plan Policy",
                    "type": "policy",
                    "category": "investment_insurance",
                    "keywords": ["ULIP", "funds", "investment", "market-linked", "switches", "charges"],
                },
                {
                    "filename": "health_insurance_policy.pdf",
                    "title": "Health Insurance Policy",
                    "type": "policy",
                    "category": "health_insurance",
                    "keywords": ["health", "hospitalization", "medical", "claims", "cashless"],
                },
                {
                    "filename": "general_terms_conditions.pdf",
                    "title": "General Terms and Conditions",
                    "type": "terms",
                    "category": "general",
                    "keywords": ["terms", "conditions", "grievance", "premium", "claims"],
                },
            ],
            "total_documents": 4,
        }

        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("✓ Generated metadata file")
        print(f"✓ All documents generated in: {self.output_dir}/")
        return metadata


if __name__ == "__main__":
    generator = PolicyDocumentGenerator()
    generator.generate_all_documents()
