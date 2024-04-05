from django.shortcuts import render
from django.http import HttpResponse
import os
from github_requests import generate_user_content  # Adjust the import path as necessary


def index(request):
    return render(request, 'index.html')

"""
class RealTimeChatGPT(ChatGPT):
        
    def __init__(self):

        self.chatgpt = ChatGPT()
        self.response_stream = None

    def index(self, request):

        if request.method == 'POST':
            repo_url = request.POST.get('repo_url', '')
            if repo_url:
                owner, repo = repo_url.split('/')[-2:]  # Extract owner and repo name from URL
                
                user_content = generate_user_content(owner, repo)

                self.response_stream = self.chatgpt.get_stream(user_content, 
                                                additional_context = None)

        return render(request, 'index.html', {'readme_content': self.chatgpt.content_html})
    
    async def update_content(self, request):
        # Here, generate or retrieve your updated HTML content

        if self.response_stream is not None:
            # Process the stream as it arrives.
            for completion in self.response_stream:
                
                self.chatgpt.data_processing(completion)

        return render(request, 'index.html', {'readme_content': self.chatgpt.content_html})


def get_updated_content(request):
    # Here, generate or retrieve your updated HTML content
    
    try:
        latest_content = ProcessedContent.objects.latest('timestamp')
        updated_html_content = latest_content.content_html
    except ProcessedContent.DoesNotExist:
        updated_html_content = "No content available yet."

    print(updated_html_content)

    return JsonResponse({"html_content": updated_html_content})
"""