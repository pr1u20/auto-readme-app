from django.shortcuts import render
from django.http import HttpResponse
import os
from github_requests import generate_user_content  # Adjust the import path as necessary
from chatgpt_requests import ChatGPT
import markdown
from django.http import JsonResponse

def index(request):
    chatgpt = ChatGPT()

    if request.method == 'POST':
        repo_url = request.POST.get('repo_url', '')
        if repo_url:
            owner, repo = repo_url.split('/')[-2:]  # Extract owner and repo name from URL
            
            user_content = generate_user_content(owner, repo)

            response_stream = chatgpt.get_stream(user_content, 
                                              additional_context = None)

        # Process the stream as it arrives.
        for completion in response_stream:
            
            chatgpt.data_processing(completion)

            # In your index view
            request.session['readme_content_html'] = chatgpt.content_html
            request.session.modified = True
    
    return render(request, 'index.html', {'readme_content': chatgpt.content_html})

def get_updated_content(request):
    # Here, generate or retrieve your updated HTML content
    
    updated_html_content = request.session.get('readme_content_html', 'No new content.')

    return JsonResponse({"html_content": updated_html_content})
