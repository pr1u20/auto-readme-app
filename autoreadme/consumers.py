from channels.generic.websocket import AsyncWebsocketConsumer
import json
from github_requests import generate_user_content  # Adjust the import path as necessary
from chatgpt_requests import ChatGPT, markdown_to_html
import asyncio
import markdown
import pypandoc


class ChatGPTConsumer(AsyncWebsocketConsumer):
    
    async def connect(self):
        self.chatgpt = ChatGPT()  # Assuming ChatGPT can be initialized like this
        await self.accept()

    async def disconnect(self, close_code):
        # Handle disconnection
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        repo_url = text_data_json['repo_url']
        
        if repo_url:
            owner, repo = repo_url.split('/')[-2:]
            user_content, tree_structure = generate_user_content(owner, repo)
            #tree_html = markdown.markdown(tree_structure)
            tree_html = markdown_to_html(tree_structure)

            # Assuming get_stream is an async generator
            for completion in self.chatgpt.get_stream(user_content, additional_context=None):
                self.chatgpt.data_processing(completion)
                # Send update to the client
                await self.send(text_data=json.dumps({'message': self.chatgpt.content_html,
                                                      'tree': tree_html}))
                await asyncio.sleep(0.02)  # Adjust the delay as needed