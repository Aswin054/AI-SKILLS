INTERVIEW_QUESTIONS = {
    "general": [
        "How does your military experience prepare you for this role?",
        "Describe a time you led a team under pressure.",
    ],
    "technical": [
        "Explain how you troubleshoot network issues (for IT roles).",
        "How would you handle a medical emergency? (for EMT roles).",
    ]
}

def get_interview_questions(job_role):
    """Fetch relevant interview questions based on job."""
    questions = INTERVIEW_QUESTIONS["general"]
    if "IT" in job_role:
        questions += INTERVIEW_QUESTIONS["technical"][0]
    elif "EMT" in job_role:
        questions += INTERVIEW_QUESTIONS["technical"][1]
    return questions

# Example Usage
if __name__ == "__main__":
    job_role = input("Target Job Title: ")
    questions = get_interview_questions(job_role)
    print("\nPractice These Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
