from openai import OpenAI
import os
from github_requests import generate_user_content
import markdown
from markdown.extensions import Extension

class ChatGPT():
    def __init__(self):
        self.client = OpenAI()
        self.md_converter = markdown.Markdown(extensions=[YourCustomMarkdownExtension()])
        self.content_md = ""
        self.content_html = ""
    
    def feed_user_content(self, user_content, additional_context=None):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You have to read all the content from the github repository and automatically generate a readme file, with clear isntructions on how to use the code." + (additional_context or "")},
                {"role": "user", "content": user_content},
            ]
        )

        self.content_md = completion.choices[0].message.content
        
        return self.content_md
    
    def feed_user_content_streaming(self, user_content, additional_context=None):
        
        response_stream = self.get_stream(user_content, additional_context)

        # Process the stream as it arrives.
        for completion in response_stream:
            
            self.data_processing(completion)

    def data_processing(self, completion):

        # Assuming each `completion` here represents a chunk of text.
        # You might receive it word by word, line by line, or in small groups of words.
        content = completion.choices[0].delta.content
        if not content is None:
            self.content_md += content
            print(content, end ="")

            # Convert the Markdown chunk to HTML
            html_chunk = self.md_converter.convert(self.content_md)
            # Append or merge the HTML chunk to your existing HTML content
            self.content_html = html_chunk

            # Optionally, reset the internal state of the Markdown converter if needed
            self.md_converter.reset()


    def get_stream(self, user_content, additional_context=None):
        # Assume `stream=True` enables streaming completions.
        # This is a conceptual example; actual implementation details may vary.
        response_stream = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You have to read all the content from the github repository and automatically generate a readme file, with clear instructions on how to use the code." + (additional_context or "")},
                {"role": "user", "content": user_content},
            ],
            stream=True  # This is a hypothetical parameter for enabling streaming.
        )

        return response_stream
    
    def save_as_md(self, filename):
        path = os.path.join("content", "readme_content", filename)

        with open(path, 'w', encoding='utf-8') as md_file:
            md_file.write(self.content_md)
            print(f"Content saved to {filename}")


class YourCustomMarkdownExtension(Extension):
    def extendMarkdown(self, md):
        # Here you can add custom processors, patterns, or postprocessors to the Markdown converter
        pass


if __name__ == "__main__":

    owner = "pr1u20"  # Replace with the repository owner's username
    repo = "pyaocs"  # Replace with the repository name

    user_content, tree_structure = generate_user_content(owner, repo, save_as_md=True)

    chat_bot = ChatGPT()
    chat_bot.feed_user_content_streaming(user_content,
                              additional_context=None)
    
    #chatgpt.save_as_md(f"{repo}_readme.md")