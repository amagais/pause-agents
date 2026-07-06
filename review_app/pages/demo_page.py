"""Demo page with synthetic ICU-PAUSE data for presentation purposes.

Mirrors the real review_page layout exactly, reusing the same display
and review widgets so formatting changes propagate automatically.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Synthetic ICU-PAUSE output (generated note)
# ---------------------------------------------------------------------------

DEMO_OUTPUT: dict = {
    "hospitalization_id": "DEMO-2024-00001",
    "generated_at": "2024-01-30T14:32:00Z",
    "sections": {
        "I": (
            "78-year-old male with PMH of atrial fibrillation with a known "
            "cardioversion history, hypertension, type 2 diabetes mellitus, and "
            "chronic obstructive pulmonary disease (COPD) with a history of "
            "frequent exacerbations. Admitted 1/26/2024 for acute onset shortness "
            "of breath and worsening cough, accompanied by a productive yellow "
            "sputum. Initial assessment revealed an oxygen saturation of 88% on "
            "room air, and auscultation demonstrated coarse crackles bilaterally. "
            "He reports a recent increase in sputum volume and thickening, along "
            "with feeling increasingly fatigued over the past 3 days. Labs showed "
            "leukocytosis of 18,000/mm\u00b3 with a left shift, and an elevated "
            "C-reactive protein of 125 mg/L. Chest X-ray demonstrated significant "
            "bilateral infiltrates consistent with pneumonia. Patient was initiated "
            "on intravenous antibiotics (cefepime and azithromycin) and supplemental "
            "oxygen via nasal cannula. ICU course was complicated by a rapid decline "
            "in respiratory status requiring intubation and mechanical ventilation "
            "(high-flow nasal cannula initially, transitioning to ventilator with "
            "SIMV settings). He developed a new onset pleural effusion requiring "
            "thoracentesis which revealed empyema with gram-positive cocci suggestive "
            "of Staphylococcus aureus. Despite aggressive antibiotic therapy and "
            "drainage, patient remained hemodynamically unstable and required "
            "prolonged ventilation. Currently, patient is on ventilator support "
            "(pressure-controlled ventilation), requiring frequent suctioning and a "
            "chest tube for drainage. He is receiving intravenous fluids, and blood "
            "pressure is managed with norepinephrine infusions. Patient remains at "
            "high risk for ventilator-associated pneumonia and requires close monitoring."
        ),
        "C": (
            "Code Status: Not determined\n"
            "DPOA: Not determined\n"
            "ACP: Not documented\n"
            "Currently vent-dependent: Y \u2014 Trach Collar, None"
        ),
        "U_unprescribing": (
            "Changes to home meds: None\n\n"
            "Anticoagulation:\n"
            "VTE Prophylaxis - None \u2013 Reason: Maintaining therapeutic apixaban "
            "for recent pulmonary embolism/deep vein thrombosis.\n"
            "Therapeutic anticoagulation - Apixaban\n\n"
            "Antibiotics:\n"
            "[CONFLICT] Recent sputum cultures positive for Pseudomonas aeruginosa "
            "and Stenotrophomonas maltophilia. Antibiotic plan: Currently, holding "
            "antibiotics to assess response to supportive care. Close monitoring for "
            "worsening pneumonia (increased temperature, increased sputum production, "
            "decreased oxygen saturation) and elevated inflammatory markers will "
            "dictate re-evaluation of antibiotic necessity within 24-48 hours.\n\n"
            "High-Risk Medications:\n"
            "Hydromorphone & Oxycodone (Chronic Pain): Continue home regimen for "
            "pain management. Critical Monitoring: Frequent assessment of mental "
            "status, respiratory rate and effort, and signs of opioid-induced "
            "sedation. Consider a trial reduction in dose if significant drowsiness "
            "or confusion develops, exploring alternative pain management strategies "
            "as appropriate."
        ),
        "P": "No pending tests at the time of transfer.",
        "A": (
            "Social Work: Initial evaluation completed; recommends continued skilled "
            "physical therapy for bed mobility and balance deficits.\n"
            "Physical Therapy: PT: Max assist x2 for bed mobility, poor sitting "
            "balance; recommended continued skilled physical therapy.\n"
            "Occupational Therapy: OT: Not enough information from structured data "
            "to determine appropriate interventions at this time.\n"
            "Speech Therapy: SLP: Not enough information from structured data to "
            "determine appropriate interventions at this time.\n"
            "Wound Care: No active wound care currently required."
        ),
        "U_uncertainty": (
            "Working diagnosis at the time of transfer: chronic respiratory failure "
            "s/p tracheostomy with failed capping trial, though ddx includes "
            "ICU-acquired weakness contributing to wean failure, evolving "
            "ventilator-associated pneumonia (Pseudomonas/Stenotrophomonas on "
            "sputum cultures), and pre-renal AKI of unclear etiology."
        ),
        "S": (
            "- Chronic respiratory failure with tracheostomy: Weaned back to trach collar (FiO2 0.3, 10 LPM) by 1/29 with preserved oxygenation (SpO2 92%), but respiratory wean remains incomplete.\n"
            "- Trach capping intolerance due to upper airway resistance: Failed capping attempt on 1/29 per RT with resultant tachypnea (RR 33) and desaturation to 88%; airway readiness for capping remains unresolved.\n"
            "- Respiratory infection surveillance: Sputum cultures returned Pseudomonas aeruginosa and Stenotrophomonas maltophilia; currently holding antibiotics and monitoring clinically given downtrending WBC (18 \u2192 14.2) and afebrile status.\n"
            "- Acute kidney injury: Creatinine rose from 1.2 to 2.1 over 48 hours, likely pre-renal; nephrotoxins held, monitoring trend.\n"
            "- Asymptomatic bradycardia: Heart rate trending down 61 \u2192 52, patient hemodynamically stable and without symptoms; continue telemetry monitoring.\n"
            "- Persistent tachypnea: Respiratory rate increased from 23 to 33 over the shift, possibly effort-related vs early decompensation; warrants close surveillance.\n"
            "- ICU-acquired weakness with functional decline: Max assist x2 for bed mobility, poor sitting balance, impaired fine motor coordination; baseline prior to admission unknown.\n"
            "- Enteral nutrition via G-tube: Kate Farms 1.4 at 65 ml/hr x 12 hr tolerated without residuals; NPO by mouth pending successful trach capping trial.\n"
            "- Therapeutic anticoagulation: Apixaban 5 mg BID continued for recent PE/DVT; no VTE prophylaxis needed while therapeutically anticoagulated.\n"
            "- Chronic pain management: Home regimen of hydromorphone 2 mg and oxycodone 10 mg continued; requires close monitoring for oversedation given RASS -2 to -3.\n"
            "- Bowel regimen: Miralax and senna administered daily; last documented bowel movement was 1/10, constipation risk elevated.\n"
            "- Infection control: VRE history on record; contact precautions in place throughout hospitalization.\n"
            "- Home support coordination: Independence Plus provides 24/7 private duty nursing and has requested a discharge status update.\n"
            "- Discharge planning: Timeline unknown pending clinical progression; code status and advance care planning remain undocumented.\n"
            "\n"
            "**BEFORE TRANSFER (ICU Team):**\n"
            "- [ ] Continue current enteral nutrition plan and reassess readiness for PO diet advancement only after airway/trach status is re-evaluated\n"
            "- [ ] Confirm trach collar settings (FiO2 0.3, 10 LPM) are clearly documented in the transfer order set with suctioning frequency (q4h)\n"
            "- [ ] Obtain repeat BMP to trend creatinine before transfer; document whether nephrotoxins remain on hold and specify renal follow-up threshold (e.g., Cr > 2.5 triggers nephrology consult)\n"
            "- [ ] Communicate active norepinephrine requirement and current weaning status to receiving team; specify hemodynamic parameters for vasopressor titration and discontinuation\n"
            "- [ ] Contact Independence Plus HHC to provide updated clinical status, anticipated discharge timeline, and equipment needs (trach supplies, suction machine, G-tube supplies)\n"
            "- [ ] Initiate goals-of-care conversation with patient/family given undocumented code status and complex medical course; document outcome in ACP note before transfer\n"
            "\n"
            "**ON THE WARD (Receiving Team):**\n"
            "- [ ] Trend RR and work of breathing q4h; escalate respiratory evaluation (RT assessment, ABG) for sustained tachypnea (RR > 30) or new-onset respiratory distress\n"
            "- [ ] Monitor for signs of worsening pulmonary infection: temperature > 38.3\u00b0C, increased sputum production or purulence, rising WBC, or declining SpO2 below 90% \u2014 if any present, obtain repeat sputum culture and reassess antibiotic necessity within 24 hours\n"
            "- [ ] Continue telemetry monitoring with HR notification parameters: notify provider if HR < 50 or if patient develops symptoms (dizziness, syncope, chest pain) associated with bradycardia\n"
            "- [ ] Recheck BMP in 24 hours post-transfer to trend creatinine; if Cr > 2.5 or rising, assess volume status and consider IV fluid bolus vs nephrology consult\n"
            "- [ ] Perform structured pain and sedation assessment (RASS, pain scale) every 8 hours; if RASS \u2264 -3 on two consecutive checks, hold next opioid dose and reassess need for dose reduction\n"
            "- [ ] Continue bowel regimen (Miralax daily, senna BID); if no BM within 48 hours of transfer, escalate to bisacodyl suppository and consider abdominal X-ray to rule out ileus\n"
            "- [ ] Enforce contact precautions for VRE; ensure isolation signage is posted and all providers are aware of precaution status on handoff\n"
            "- [ ] Coordinate PT/OT evaluation within 24 hours of transfer to establish ward-level mobility goals and reassess ICU-acquired weakness trajectory\n"
            "\n"
            "**AT DISCHARGE (Case Manager/Team):**\n"
            "- [ ] Arrange transportation home accounting for tracheostomy equipment, suction machine, and need for medical escort if patient remains on supplemental oxygen\n"
            "- [ ] Finalize tracheostomy management plan with pulmonology: document weaning protocol, decannulation readiness criteria, and outpatient follow-up appointment within 2 weeks\n"
            "- [ ] Verify code status and ACP documentation is complete; if still undocumented at discharge, flag as critical outstanding item in discharge summary\n"
            "- [ ] Confirm Independence Plus HHC has received updated care plan including trach care protocol, G-tube feeding schedule, medication reconciliation, and 24/7 nursing requirements\n"
            "- [ ] Schedule outpatient follow-up appointments: pulmonology (2 weeks), PCP (1 week), nephrology if AKI unresolved, and infectious disease if cultures remain positive at discharge"
        ),
        "E": (
            "Physical exam: Patient is supine, requires complete assistance for all "
            "transfers and positioning. Lower extremity strength significantly "
            "diminished bilaterally, with inability to resist gravity in both legs. "
            "Neurological: Alert and oriented to person, place, and time, but "
            "exhibits significant difficulty with purposeful movement. Respiratory: "
            "Laryngectomy performed 6 months ago; currently utilizing a custom-fitted "
            "esophageal stent with a tracheostomy tube in place. Stent is patent and "
            "functioning well, demonstrating adequate granulation tissue formation. "
            "Current device: Tracheostomy tube (size 6.0), with humidified oxygen "
            "via nasal cannula at 2L/min. Arterial line in place, providing "
            "continuous monitoring of blood pressure. Most recent RR 24, SpO2 92% "
            "on room air. Potential for aspiration? Risk assessment initiated; "
            "suctioning performed every 4 hours. Lines/drains assessed for removal? "
            "Not applicable \u2013 esophageal stent and tracheostomy tube are currently "
            "stable. Active lines/drains/airways: Cuffless 6.0 tracheostomy, "
            "esophageal stent with tracheostomy tube, arterial line."
        ),
    },
    "todo_checklist": [],
    "warnings": [
        # Clinician-facing — these reach the bedside panel.
        {
            "category": "cross_domain_conflict",
            "severity": "safety_critical",
            "message": "Case Manager labels patient 'Currently vent-dependent: Y' but respiratory_support data shows daytime trach-collar weaning with ongoing nocturnal ventilatory dependence; reconcile before ward transfer.",
            "source_agent": "resident",
            "source_section": "C",
        },
        {
            "category": "safety_flag",
            "severity": "clinical",
            "message": "High-risk sedative/opioid interaction risk: prior fentanyl, midazolam, propofol, and hydromorphone exposure carries additive respiratory depression and hypotension risk if resumed after transfer.",
            "source_agent": "pharmacy",
            "source_section": "U_unprescribing",
        },
        {
            "category": "data_gap",
            "severity": "clinical",
            "message": "Isolation precautions/status not confirmed in available data; verify at bedside before ward transfer.",
            "source_agent": "case_manager",
            "source_section": "A",
        },
        # Audit-only — visible in dev mode, not in the clinician panel.
        {
            "category": "editorial_revision",
            "severity": "info",
            "message": "Revised '#Respiratory' summary to remove specific SIMV settings (hallucinated numeric values).",
            "source_agent": "respiratory",
            "source_section": "S",
        },
        {
            "category": "editorial_revision",
            "severity": "info",
            "message": "Standardized opioid medication item (hydromorphone/oxycodone) to avoid unsupported details such as 'dose escalation on 1/28'.",
            "source_agent": "pharmacy",
            "source_section": "U_unprescribing",
        },
        {
            "category": "editorial_revision",
            "severity": "info",
            "message": "Removed 'patient tolerating trach collar for 48 hours' (duration not documented in source data).",
            "source_agent": "respiratory",
            "source_section": "S",
        },
        {
            "category": "editorial_revision",
            "severity": "info",
            "message": "Revised '#Renal' to remove pre-renal etiology causal attribution (inference not in structured data); retained objective creatinine trend.",
            "source_agent": "nurse",
            "source_section": "S",
        },
        {
            "category": "editorial_revision",
            "severity": "info",
            "message": "Removed fabricated chest tube output detail ('150ml serosanguinous over 24h') — not in source data.",
            "source_agent": "nurse",
            "source_section": "E",
        },
    ],
    "qa_issues": [
        "Pharmacy lists 'Antibiotics: N/A - no planned antimicrobials' while nurse reports microbiology flags with '>100,000 CFU/ml Pseudomonas aeruginosa' and '>100,000 CFU/ml Stenotrophomonas maltophilia' plus prior VRE positivity; reconcile whether these are colonization vs active infection and whether antimicrobial plan/isolation implications need explicit documentation.",
        "Case manager documents 'Code Status: Not documented' and 'ACP: Not documented' despite a complex ICU stay with tracheostomy and ongoing respiratory support; this is a critical omission that should be clarified before transfer/discharge planning.",
        "Respiratory therapist notes 'Capping trial failed - upper airway resistance' but the I section states patient is on 'ventilator support (pressure-controlled ventilation)'; these are inconsistent — patient is on trach collar, not ventilator — and the current respiratory status in I should be reconciled with the S section and RT assessment.",
        "Intensivist documents norepinephrine infusion at 0.08 mcg/kg/min in the medication administration record, but the disposition section states 'Discharge Timeline Unknown - Pending Clinical Progression' without flagging active vasopressor use as a barrier to transfer; active pressors should be explicitly listed as a transfer-readiness criterion.",
        "Physical therapy documents 'Max assist x2 for bed mobility' and 'poor sitting balance' while the U_uncertainty section attributes this to 'ICU-acquired weakness (prolonged sedation RASS -2/-3 for >48h)'; however, baseline functional status prior to admission is not documented anywhere in the note — clarify whether current deficits represent new ICU-acquired weakness or pre-existing functional limitations.",
    ],
    "section_confidences": {},
    "metadata": {},
}

# ---------------------------------------------------------------------------
# Synthetic source bundle (structured clinical data)
# ---------------------------------------------------------------------------

DEMO_SOURCE: dict = {
    "demographics": {
        "age_at_admission": 78,
        "sex_category": "Male",
        "admission_type_category": "Emergency",
        "icu_admission_dttm": "2024-01-26T08:15:00",
        "reference_dttm": "2024-01-30T10:00:00",
        "icu_los_hours": 97.75,
    },
    "vitals_summary": [
        {"time_bucket": "2024-01-29 00:00–08:00", "vital_category": "HR", "vital_value": 61},
        {"time_bucket": "2024-01-29 08:00–16:00", "vital_category": "HR", "vital_value": 55},
        {"time_bucket": "2024-01-29 16:00–00:00", "vital_category": "HR", "vital_value": 52},
        {"time_bucket": "2024-01-29 00:00–08:00", "vital_category": "RR", "vital_value": 23},
        {"time_bucket": "2024-01-29 08:00–16:00", "vital_category": "RR", "vital_value": 28},
        {"time_bucket": "2024-01-29 16:00–00:00", "vital_category": "RR", "vital_value": 33},
        {"time_bucket": "2024-01-29 00:00–08:00", "vital_category": "SpO2", "vital_value": 94},
        {"time_bucket": "2024-01-29 08:00–16:00", "vital_category": "SpO2", "vital_value": 93},
        {"time_bucket": "2024-01-29 16:00–00:00", "vital_category": "SpO2", "vital_value": 92},
        {"time_bucket": "2024-01-29 00:00–08:00", "vital_category": "MAP", "vital_value": 72},
        {"time_bucket": "2024-01-29 08:00–16:00", "vital_category": "MAP", "vital_value": 68},
        {"time_bucket": "2024-01-29 16:00–00:00", "vital_category": "MAP", "vital_value": 70},
        {"time_bucket": "2024-01-29 00:00–08:00", "vital_category": "Temp", "vital_value": 37.2},
        {"time_bucket": "2024-01-29 08:00–16:00", "vital_category": "Temp", "vital_value": 37.8},
        {"time_bucket": "2024-01-29 16:00–00:00", "vital_category": "Temp", "vital_value": 37.5},
    ],
    "labs_recent": [
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "WBC", "lab_value_numeric": 14.2, "lab_unit": "K/uL"},
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "Hemoglobin", "lab_value_numeric": 10.1, "lab_unit": "g/dL"},
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "Platelets", "lab_value_numeric": 185, "lab_unit": "K/uL"},
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "Creatinine", "lab_value_numeric": 2.1, "lab_unit": "mg/dL"},
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "BUN", "lab_value_numeric": 38, "lab_unit": "mg/dL"},
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "Sodium", "lab_value_numeric": 139, "lab_unit": "mEq/L"},
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "Potassium", "lab_value_numeric": 4.2, "lab_unit": "mEq/L"},
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "Albumin", "lab_value_numeric": 3.1, "lab_unit": "g/dL"},
        {"lab_result_dttm": "2024-01-29T06:00:00", "lab_category": "CRP", "lab_value_numeric": 125, "lab_unit": "mg/L"},
        {"lab_result_dttm": "2024-01-27T06:00:00", "lab_category": "Creatinine", "lab_value_numeric": 1.2, "lab_unit": "mg/dL"},
    ],
    "meds_continuous": [
        {"admin_dttm": "2024-01-29T12:00:00", "med_category": "Norepinephrine", "med_dose": 0.08, "med_dose_unit": "mcg/kg/min"},
    ],
    "meds_intermittent": [
        {"admin_dttm": "2024-01-29T08:00:00", "med_category": "Apixaban", "med_dose": 5, "med_dose_unit": "mg", "med_status": "Given"},
        {"admin_dttm": "2024-01-29T08:00:00", "med_category": "Hydromorphone", "med_dose": 2, "med_dose_unit": "mg", "med_status": "Given"},
        {"admin_dttm": "2024-01-29T12:00:00", "med_category": "Oxycodone", "med_dose": 10, "med_dose_unit": "mg", "med_status": "Given"},
        {"admin_dttm": "2024-01-29T08:00:00", "med_category": "Miralax", "med_dose": 17, "med_dose_unit": "g", "med_status": "Given"},
        {"admin_dttm": "2024-01-29T08:00:00", "med_category": "Senna", "med_dose": 8.6, "med_dose_unit": "mg", "med_status": "Given"},
    ],
    "respiratory_support": [
        {"recorded_dttm": "2024-01-29T08:00:00", "device_category": "Trach Collar", "mode_category": "Spontaneous", "fio2_set": 0.30, "peep_set": None},
    ],
    "assessments": [
        {"recorded_dttm": "2024-01-29T08:00:00", "assessment_type": "RASS", "assessment_value": "-2"},
        {"recorded_dttm": "2024-01-29T08:00:00", "assessment_type": "GCS", "assessment_value": "11 (E3V3M5)"},
        {"recorded_dttm": "2024-01-29T12:00:00", "assessment_type": "RASS", "assessment_value": "-3"},
        {"recorded_dttm": "2024-01-29T16:00:00", "assessment_type": "RASS", "assessment_value": "-2"},
    ],
    "code_status": [],
    "diagnoses": [
        {"diagnosis_name": "Pneumonia, bilateral", "icd_code": "J18.1"},
        {"diagnosis_name": "Empyema", "icd_code": "J86.9"},
        {"diagnosis_name": "Acute respiratory failure with hypoxia", "icd_code": "J96.01"},
        {"diagnosis_name": "Atrial fibrillation", "icd_code": "I48.91"},
        {"diagnosis_name": "Hypertension", "icd_code": "I10"},
        {"diagnosis_name": "Type 2 diabetes mellitus", "icd_code": "E11.9"},
        {"diagnosis_name": "COPD with acute exacerbation", "icd_code": "J44.1"},
        {"diagnosis_name": "Acute kidney injury", "icd_code": "N17.9"},
        {"diagnosis_name": "Pulmonary embolism (recent)", "icd_code": "I26.99"},
        {"diagnosis_name": "Deep vein thrombosis (recent)", "icd_code": "I82.409"},
    ],
    "microbiology": [
        {"specimen_type": "Sputum", "organism": "Pseudomonas aeruginosa", "collect_dttm": "2024-01-28T10:00:00"},
        {"specimen_type": "Sputum", "organism": "Stenotrophomonas maltophilia", "collect_dttm": "2024-01-28T10:00:00"},
        {"specimen_type": "Pleural fluid", "organism": "Staphylococcus aureus (gram-positive cocci)", "collect_dttm": "2024-01-27T14:00:00"},
    ],
    "procedures": [
        {"procedure_name": "Endotracheal intubation", "procedure_date": "2024-01-26"},
        {"procedure_name": "Thoracentesis", "procedure_date": "2024-01-27"},
        {"procedure_name": "Chest tube insertion", "procedure_date": "2024-01-27"},
        {"procedure_name": "Tracheostomy", "procedure_date": "2024-01-28"},
    ],
    "clinical_notes": {
        "nursing_note": [
            {
                "creation_dttm": "2024-01-29T16:00:00",
                "note_text": (
                    "Patient on trach collar, FiO2 0.3, 10 LPM. Capping trial attempted "
                    "this AM but failed due to upper airway resistance — patient became "
                    "tachypneic (RR 33) and desatted to 88%. Returned to trach collar. "
                    "Suctioning q4h with moderate thick yellow secretions. HR trending "
                    "down 61→52, patient asymptomatic, MD aware. Max assist x2 for bed "
                    "mobility. Sitting balance poor. Patient alert and oriented x3 but "
                    "fatigues quickly. G-tube feeds (Kate Farms 1.4, 65 ml/hr x 12hr) "
                    "tolerated without residuals. NPO otherwise — awaiting successful "
                    "capping trial. Last BM 1/10 — Miralax and Senna given. VRE history "
                    "— contact precautions in place. Arterial line in L radial, site clean "
                    "and dry. Trach site clean with minimal erythema."
                ),
            },
        ],
        "progress_note": [
            {
                "creation_dttm": "2024-01-29T10:00:00",
                "note_text": (
                    "Day 4 ICU. 78M PMH AFib, HTN, DM2, COPD admitted for bilateral PNA "
                    "c/b empyema and respiratory failure requiring tracheostomy. Current: "
                    "trach collar FiO2 30%. Capping trial failed today — significant upper "
                    "airway resistance noted. AKI developing: Cr 1.2 → 2.1 over 48h, likely "
                    "pre-renal in setting of reduced PO intake and norepinephrine use. Will "
                    "monitor, hold nephrotoxins. Sputum cultures returned Pseudomonas and "
                    "Stenotrophomonas — decision to hold antibiotics and monitor clinically "
                    "given patient afebrile and WBC trending down (18→14.2). Will reassess "
                    "in 24-48h. Bradycardia: HR 52, asymptomatic, likely medication-related. "
                    "Continue home apixaban for recent PE/DVT. Continue hydromorphone/oxycodone "
                    "per home regimen with close monitoring for oversedation given RASS -2 to -3. "
                    "PT/OT: max assist x2, poor sitting balance, impaired fine motor coordination. "
                    "Concern for ICU-acquired weakness vs baseline deconditioning. Social work "
                    "consulted for discharge planning. Code status not yet addressed — will "
                    "discuss with family today."
                ),
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Synthetic claims (atomic verifiable statements)
# ---------------------------------------------------------------------------

DEMO_CLAIMS: list[dict] = [
    # I section — every sentence as a separate claim
    {"claim_id": "I_claim_1", "section": "I", "text": "78-year-old male with PMH of atrial fibrillation with a known cardioversion history, hypertension, type 2 diabetes mellitus, and chronic obstructive pulmonary disease (COPD) with a history of frequent exacerbations."},
    {"claim_id": "I_claim_2", "section": "I", "text": "Admitted 1/26/2024 for acute onset shortness of breath and worsening cough, accompanied by a productive yellow sputum."},
    {"claim_id": "I_claim_3", "section": "I", "text": "Initial assessment revealed an oxygen saturation of 88% on room air, and auscultation demonstrated coarse crackles bilaterally."},
    {"claim_id": "I_claim_4", "section": "I", "text": "He reports a recent increase in sputum volume and thickening, along with feeling increasingly fatigued over the past 3 days."},
    {"claim_id": "I_claim_5", "section": "I", "text": "Labs showed leukocytosis of 18,000/mm\u00b3 with a left shift, and an elevated C-reactive protein of 125 mg/L."},
    {"claim_id": "I_claim_6", "section": "I", "text": "Chest X-ray demonstrated significant bilateral infiltrates consistent with pneumonia."},
    {"claim_id": "I_claim_7", "section": "I", "text": "Patient was initiated on intravenous antibiotics (cefepime and azithromycin) and supplemental oxygen via nasal cannula."},
    {"claim_id": "I_claim_8", "section": "I", "text": "ICU course was complicated by a rapid decline in respiratory status requiring intubation and mechanical ventilation (high-flow nasal cannula initially, transitioning to ventilator with SIMV settings)."},
    {"claim_id": "I_claim_9", "section": "I", "text": "He developed a new onset pleural effusion requiring thoracentesis which revealed empyema with gram-positive cocci suggestive of Staphylococcus aureus."},
    {"claim_id": "I_claim_10", "section": "I", "text": "Despite aggressive antibiotic therapy and drainage, patient remained hemodynamically unstable and required prolonged ventilation."},
    {"claim_id": "I_claim_11", "section": "I", "text": "Currently, patient is on ventilator support (pressure-controlled ventilation), requiring frequent suctioning and a chest tube for drainage."},
    {"claim_id": "I_claim_12", "section": "I", "text": "He is receiving intravenous fluids, and blood pressure is managed with norepinephrine infusions."},
    {"claim_id": "I_claim_13", "section": "I", "text": "Patient remains at high risk for ventilator-associated pneumonia and requires close monitoring."},
    # C section
    {"claim_id": "C_claim_1", "section": "C", "text": "Code status is not determined."},
    {"claim_id": "C_claim_2", "section": "C", "text": "DPOA is not determined."},
    {"claim_id": "C_claim_3", "section": "C", "text": "ACP is not documented."},
    {"claim_id": "C_claim_4", "section": "C", "text": "Patient is currently vent-dependent on Trach Collar."},
    # U (unprescribing) section
    {"claim_id": "U_unprescribing_claim_1", "section": "U_unprescribing", "text": "No changes to home medications."},
    {"claim_id": "U_unprescribing_claim_2", "section": "U_unprescribing", "text": "VTE prophylaxis is not given because patient is maintaining therapeutic apixaban for recent pulmonary embolism/deep vein thrombosis."},
    {"claim_id": "U_unprescribing_claim_3", "section": "U_unprescribing", "text": "Therapeutic anticoagulation is apixaban."},
    {"claim_id": "U_unprescribing_claim_4", "section": "U_unprescribing", "text": "Recent sputum cultures are positive for Pseudomonas aeruginosa and Stenotrophomonas maltophilia."},
    {"claim_id": "U_unprescribing_claim_5", "section": "U_unprescribing", "text": "Antibiotics are currently being held to assess response to supportive care."},
    {"claim_id": "U_unprescribing_claim_6", "section": "U_unprescribing", "text": "Patient is on hydromorphone and oxycodone for chronic pain as a home regimen."},
    # P section
    {"claim_id": "P_claim_1", "section": "P", "text": "No pending tests at the time of transfer."},
    # A section
    {"claim_id": "A_claim_1", "section": "A", "text": "Social Work: Initial evaluation completed; recommends continued skilled physical therapy for bed mobility and balance deficits."},
    {"claim_id": "A_claim_2", "section": "A", "text": "Physical Therapy: Max assist x2 for bed mobility, poor sitting balance."},
    {"claim_id": "A_claim_3", "section": "A", "text": "No active wound care currently required."},
    # U (uncertainty) section
    {"claim_id": "U_uncertainty_claim_1", "section": "U_uncertainty", "text": "Max assist x2 required for bed mobility, poor sitting balance, impaired fine motor coordination."},
    {"claim_id": "U_uncertainty_claim_2", "section": "U_uncertainty", "text": "Differential includes risk for ICU-acquired weakness from prolonged sedation (RASS -2/-3 for >48h)."},
    {"claim_id": "U_uncertainty_claim_3", "section": "U_uncertainty", "text": "Certainty level is 2."},
    # S section
    {"claim_id": "S_claim_1", "section": "S", "text": "Trach collar: FiO2 0.3, 10 LPM; Capping Trial Failed."},
    {"claim_id": "S_claim_2", "section": "S", "text": "AKI with creatinine rising from 1.2 to 2.1 over 48 hours."},
    {"claim_id": "S_claim_3", "section": "S", "text": "Heart rate decreased from 61 to 52, patient is asymptomatic."},
    {"claim_id": "S_claim_4", "section": "S", "text": "Enteral nutrition: Kate Farms 1.4, 65 ml/hr x 12 hr via G-Tube, tolerated."},
    {"claim_id": "S_claim_5", "section": "S", "text": "VRE History - Contact Precautions."},
    # E section
    {"claim_id": "E_claim_1", "section": "E", "text": "Patient is supine, requires complete assistance for all transfers and positioning."},
    {"claim_id": "E_claim_2", "section": "E", "text": "Lower extremity strength significantly diminished bilaterally, with inability to resist gravity in both legs."},
    {"claim_id": "E_claim_3", "section": "E", "text": "Alert and oriented to person, place, and time, but exhibits significant difficulty with purposeful movement."},
    {"claim_id": "E_claim_4", "section": "E", "text": "Laryngectomy performed 6 months ago; currently utilizing a custom-fitted esophageal stent with a tracheostomy tube in place."},
    {"claim_id": "E_claim_5", "section": "E", "text": "Current device: Tracheostomy tube (size 6.0), with humidified oxygen via nasal cannula at 2L/min."},
    {"claim_id": "E_claim_6", "section": "E", "text": "Arterial line in place, providing continuous monitoring of blood pressure."},
    {"claim_id": "E_claim_7", "section": "E", "text": "Most recent RR 24, SpO2 92% on room air."},
    {"claim_id": "E_claim_8", "section": "E", "text": "Suctioning performed every 4 hours."},
    {"claim_id": "E_claim_9", "section": "E", "text": "Active lines/drains/airways: Cuffless 6.0 tracheostomy, esophageal stent with tracheostomy tube, arterial line."},
]


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------

def render_demo_page() -> None:
    """Render the demo page mirroring the real review page layout."""
    import streamlit as st

    from display.note_renderer import render_note
    from display.source_renderer import render_source
    from review.hallucination_widget import render_hallucination_widget
    from review.omissions_widget import render_omissions_widget
    from review.pdsqi9_widget import render_pdsqi9_widget

    # --- Header ---
    demo = DEMO_SOURCE["demographics"]
    age = demo["age_at_admission"]
    sex = demo["sex_category"]
    admit = demo["admission_type_category"]
    icu_adm = demo["icu_admission_dttm"]
    los = demo["icu_los_hours"]

    col_header, col_back = st.columns([5, 1])
    with col_header:
        st.markdown("### Demo Case `DEMO-2024-00001`")
        st.caption(
            f"{age}{sex[0]} | {admit} | ICU admission: {icu_adm} | ICU LOS: {los:.1f}h"
        )
    with col_back:
        if st.button("Back to dashboard"):
            st.session_state["page"] = "dashboard"
            st.rerun()

    st.info(
        "This is a **synthetic patient case** for demonstration purposes only. "
        "No real patient data is displayed. Responses are not saved.",
        icon="\u2139\ufe0f",
    )
    st.divider()

    # --- Two-column layout (mirrors review_page) ---
    left, right = st.columns([55, 45])

    with left:
        st.markdown("#### Source Data")
        st.caption("Review the source data and generated note below.")
        render_source(DEMO_SOURCE)

        st.divider()
        st.markdown("#### Generated ICU-PAUSE Note")
        render_note(DEMO_OUTPUT)

    with right:
        st.markdown("#### Review Form")
        st.caption(
            "Complete all four steps below, then submit. "
            "You can save a draft at any point and return later."
        )
        with st.container(height=900):

            # Step 1: Hallucination / accuracy check
            with st.expander("**Step 1 \u2014 Accuracy Check (Claim Verification)**", expanded=True):
                render_hallucination_widget(DEMO_CLAIMS, existing=None)

            # Step 2: Omissions check
            with st.expander("**Step 2 \u2014 Completeness Check (Critical Omissions)**", expanded=True):
                render_omissions_widget(DEMO_SOURCE, existing=None)

            # Step 3: PDSQI-9
            with st.expander("**Step 3 \u2014 PDSQI-9 Quality Scores**", expanded=True):
                render_pdsqi9_widget(existing=None)

            # Step 4: Comments
            with st.expander("**Step 4 \u2014 Overall Comments (optional)**", expanded=False):
                st.text_area(
                    "Any additional observations about this note? (max 500 characters)",
                    value="",
                    max_chars=500,
                    key="demo_overall_comment",
                    height=120,
                )

        # Note: demo page does not save responses
        st.caption("Demo mode \u2014 responses are not saved.")
