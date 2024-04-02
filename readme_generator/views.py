from django.shortcuts import render
from django.http import HttpResponse
import os
print(os.getcwd())
from github_requests import generate_user_content  # Adjust the import path as necessary
from chatgpt_requests import ChatGPT
import markdown

def index(request):
    readme_content_html = ""
    if request.method == 'POST':
        repo_url = request.POST.get('repo_url', '')
        if repo_url:
            owner, repo = repo_url.split('/')[-2:]  # Extract owner and repo name from URL
            
            user_content = generate_user_content(owner, repo)
            chatgpt = ChatGPT()
            readme_content_md = chatgpt.feed_user_content(user_content,
                                                       additional_context="The repository can be installed as pip install pyaocs.")
            
            # Convert the readme content to HTML
            readme_content_html = markdown.markdown(readme_content_md)
    
    return render(request, 'index.html', {'readme_content': readme_content_html})
