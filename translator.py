import json

# Mock database of military-to-civilian skill mappings
MILITARY_SKILL_DB = {
    "11B": {"civilian_title": "Infantry Team Leader", "skills": ["Team Leadership", "Risk Management", "Operational Planning"]},
    "25B": {"civilian_title": "IT Specialist", "skills": ["Network Administration", "Cybersecurity", "Hardware Troubleshooting"]},
    "68W": {"civilian_title": "Emergency Medical Technician", "skills": ["CPR Certified", "Trauma Care", "Patient Assessment"]},
    "1N4X1": {"civilian_title": "Intelligence Analyst", "skills": ["Data Analysis", "Threat Assessment", "Briefing Skills"]},
}

def translate_military_skill(military_code):
    """Convert military job code to civilian skills."""
    if military_code in MILITARY_SKILL_DB:
        return MILITARY_SKILL_DB[military_code]
    else:
        return {"error": "Military code not found in database."}

# Example Usage
if __name__ == "__main__":
    military_code = input("Enter your Military MOS/Rate/AFSC: ").strip().upper()
    result = translate_military_skill(military_code)
    print(json.dumps(result, indent=2))
