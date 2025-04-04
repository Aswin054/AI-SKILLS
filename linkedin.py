LINKEDIN_TIPS = [
    "Optimize your LinkedIn headline: 'Transitioning [Military Role] | Seeking [Civilian Role]'",
    "Connect with veteran-friendly employers like Amazon, Boeing, and USAA.",
    "Join groups: 'Military Veterans in Tech' or 'Veteran Transition Network'.",
]

def get_linkedin_tips():
    """Simulate fetching tips from LinkedIn Learning API."""
    return LINKEDIN_TIPS

# Example Usage
if __name__ == "__main__":
    print("LinkedIn Networking Tips:")
    for tip in get_linkedin_tips():
        print(f"- {tip}")
