import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_response(state):
    master_prompt = {
    "bored/looking": "You are an expert in ADHD and pedagogy. You know how to work with children with ADHD. The child is looking at the screen, but their EEG signal indicates boredom. They are likely daydreaming. Suggest to the caregiver what they can do in this situation.",
    "bored/not_looking": "You are an expert in ADHD and pedagogy. You know how to work with children with ADHD. The child is not looking at the screen, and their EEG signal indicates boredom. They are likely daydreaming and not attempting to focus. Suggest to the caregiver what they can do in this situation.",
    "focused/looking": "You are an expert in ADHD and pedagogy. You know how to work with children with ADHD. The child is looking at the screen, and their EEG signal indicates focus. This is okay.",
    "focused/not_looking": "You are an expert in ADHD and pedagogy. You know how to work with children with ADHD. The child is not looking at the screen, and their EEG signal indicates focus. They are likely being distracted by something. Suggest to the caregiver what they can do in this situation."
}
    # Wybierz odpowiedni prompt
    prompt = master_prompt.get(state, "Stan nieznany, spróbuj podać inną wartość.")

    # Generuj odpowiedź
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please suggest educational solutions that could help in this situation."}
        ]
    )

    # Pobierz treść odpowiedzi
    message = response.choices[0].message.content
    return message

# Przykład użycia
state = "bored/not_looking"
response1 = generate_response(state)
print(response1)

state = "bored/looking"
response2 = generate_response(state)
print(response2)

state = "focused/not_looking"
response3 = generate_response(state)
print(response3)

state = "focused/looking"
response4 = generate_response(state)
print(response4)