from openai import OpenAI
import os
from github_requests import generate_user_content

class ChatGPT():
    def __init__(self):
        self.client = OpenAI()
    
    def feed_user_content(self, user_content, additional_context=None):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You have to read all the content from the github repository and automatically generate a readme file, with clear isntructions on how to use the code." + (additional_context or "")},
                {"role": "user", "content": user_content},
            ]
        )

        self.content = completion.choices[0].message.content
        
        return self.content
    
    def save_as_md(self, filename):
        path = os.path.join("content", "readme_content", filename)

        with open(path, 'w', encoding='utf-8') as md_file:
            md_file.write(self.content)
            print(f"Content saved to {filename}")


if __name__ == "__main__":

    owner = "pr1u20"  # Replace with the repository owner's username
    repo = "pyaocs"  # Replace with the repository name

    user_content = generate_user_content(owner, repo, save_as_md=True)

    chatgpt = ChatGPT()
    chatgpt.feed_user_content(user_content,
                              additional_context="The repository can be installed as pip install pyaocs.")
    
    chatgpt.save_as_md(f"{repo}_readme.md")