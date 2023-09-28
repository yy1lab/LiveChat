import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests

# Your client ID and secret
CLIENT_ID = "6zpgq9gdbeo0lnc3an0a7uxoioeja4"
CLIENT_SECRET = "d1jlinyc4o8a59698tzdgujcg5vkh7"

# The URL to request the token
REDIRECT_URI = "http://localhost:3000/auth/twitch/callback"

# The scopes you want to request access to
SCOPES = ["analytics:read:games"]

# The URL to request the token
TOKEN_URL = f"https://id.twitch.tv/oauth2/token?client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&redirect_uri={REDIRECT_URI}&grant_type=authorization_code"

# The HTML response to send to the user when the token is received
HTML_RESPONSE = """
<!doctype html>
<html>
<head>
    <title>Twitch API Token Received</title>
</head>
<body>
    <h1>Twitch API Token Received</h1>
    <p>You can close this tab now.</p>
</body>
</html>
"""

# The handler for the HTTP server
class TwitchTokenHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(HTML_RESPONSE.encode())
        code = self.path.split("=")[1]
        self.server.access_token = self.request_token(code)

    def request_token(self, code):
        response = requests.post(TOKEN_URL + f"&code={code}")
        if response.status_code == 200:
            data = response.json()
            return data.get("access_token")

# Start the temporary server and open the authorization URL
server_address = ("", 3000)
httpd = HTTPServer(server_address, TwitchTokenHandler)
webbrowser.open_new(f"https://id.twitch.tv/oauth2/authorize?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&response_type=code&scope={'+'.join(SCOPES)}")

# Wait for the user to authorize the application and receive the token
httpd.handle_request()

# The access token is now available as a property of the server object
access_token = httpd.access_token
