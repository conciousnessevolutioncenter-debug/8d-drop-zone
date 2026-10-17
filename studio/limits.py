"""Limit upload bytes before multipart parsing can spool an unlimited body."""
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse

class UploadLimit:
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') != '/api/projects':
            return await self.app(scope, receive, send)
        total = 0
        exceeded = False
        async def bounded_receive():
            nonlocal total, exceeded
            message = await receive()
            if message['type'] == 'http.request':
                total += len(message.get('body', b''))
                if total > self.max_bytes:
                    exceeded = True
                    # Starlette's parser closes any temporary files on this exception.
                    raise MultiPartException('Upload exceeds the allowed size.')
            return message
        async def bounded_send(message):
            if not exceeded:
                await send(message)
        try:
            await self.app(scope, bounded_receive, bounded_send)
        except MultiPartException:
            if not exceeded:
                raise
        if exceeded:
            await JSONResponse({'detail':'Upload exceeds the allowed size.'}, status_code=413)(scope, receive, send)
